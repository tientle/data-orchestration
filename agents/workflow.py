
"""LangGraph-based ML workflow for Instacart reorder prediction.

Agents:
1. Data Agent: Load and validate cleaned data from PostgreSQL.
2. Split Agent: Create user-disjoint training and test sets.
3. Training Agent: Train a logistic regression model.
4. Evaluation Agent: Evaluate reorder predictions.
5. Reporting Agent: Save metrics and workflow status.

Each agent is a specialized deterministic worker.
"""

import json
from typing import TypedDict

import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from langgraph.graph import StateGraph, START, END

from scripts.pipeline import OUTPUTS, database_url


# --------------------------------------------------
# 1. Define model features
# --------------------------------------------------

FEATURES = [
    "times_purchased",
    "last_purchase_order",
    "mean_cart_position",
    "historical_reorder_rate",
    "previous_order_count",
    "orders_since_last_purchase",
    "next_order_dow",
    "next_order_hour",
    "days_since_prior_order",
    "aisle_id",
    "department_id",
]

CATEGORICAL = [
    "aisle_id",
    "department_id",
    "next_order_dow",
]

NUMERIC = [
    column
    for column in FEATURES
    if column not in CATEGORICAL
]


# --------------------------------------------------
# 2. Define LangGraph state
# --------------------------------------------------

class State(TypedDict, total=False):
    status: str
    rows: int
    train_rows: int
    test_rows: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    message: str


# Store large Python objects outside LangGraph state.
context = {}


def save_status(state, node):
    """Save the latest agent status for visualization."""

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    payload = {
        "last_node": node,
        **state,
    }

    status_path = OUTPUTS / "agent_status.json"

    status_path.write_text(
        json.dumps(payload, indent=2)
    )


# --------------------------------------------------
# 3. Data Agent
# --------------------------------------------------

def data_agent(state: State):
    """Load cleaned Instacart data from PostgreSQL."""

    engine = create_engine(database_url())

    try:
        df = pd.read_sql_table(
            "instacart_reorder_features",
            engine
        )
    finally:
        engine.dispose()

    required_columns = FEATURES + [
        "user_id",
        "target_reordered",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "The Instacart feature table is empty."
        )

    if df["target_reordered"].nunique() != 2:
        raise ValueError(
            "Target must contain two classes."
        )

    context["data"] = df

    update = {
        **state,
        "status": "data validated",
        "rows": len(df),
    }

    save_status(update, "data_agent")

    return update


# --------------------------------------------------
# 4. Split Agent
# --------------------------------------------------

def split_agent(state: State):
    """Split data by user to prevent user overlap."""

    df = context["data"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["target_reordered"],
            groups=df["user_id"],
        )
    )

    context["train"] = df.iloc[train_idx].copy()
    context["test"] = df.iloc[test_idx].copy()

    update = {
        **state,
        "status": "user-disjoint train/test split",
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
    }

    save_status(update, "split_agent")

    return update


# --------------------------------------------------
# 5. Feature preparation
# --------------------------------------------------

def prepare_features(df):
    """Convert model features to numeric NumPy inputs."""

    X = df[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X.to_numpy()


# --------------------------------------------------
# 6. Training Agent
# --------------------------------------------------

def training_agent(state: State):
    """Train a logistic regression reorder classifier."""

    df = context["train"]

    X_train = prepare_features(df)
    y_train = df["target_reordered"]

    # Select columns by position to avoid
    # ColumnTransformer feature-name issues.
    numeric_indices = [
        FEATURES.index(column)
        for column in NUMERIC
    ]

    categorical_indices = [
        FEATURES.index(column)
        for column in CATEGORICAL
    ]

    # Numeric features:
    # 1. Fill missing values with the median.
    # 2. Standardize features to mean 0 and variance 1.
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical features:
    # 1. Fill missing values with the most common value.
    # 2. One-hot encode categories.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_indices,
            ),
            (
                "category",
                categorical_transformer,
                categorical_indices,
            ),
        ],
        remainder="drop",
    )

    # Build the full ML pipeline.
    model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocess,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train the model.
    model.fit(X_train, y_train)

    context["model"] = model

    update = {
        **state,
        "status": "model trained",
    }

    save_status(update, "training_agent")

    return update


# --------------------------------------------------
# 7. Evaluation Agent
# --------------------------------------------------

def evaluation_agent(state: State):
    """Evaluate predictions on held-out users."""

    df = context["test"]

    X_test = prepare_features(df)
    y_test = df["target_reordered"]

    model = context["model"]

    # Predict reorder probabilities.
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Convert probabilities to binary predictions.
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    update = {
        **state,
        "status": "model evaluated",
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
    }

    save_status(update, "evaluation_agent")

    return update


# --------------------------------------------------
# 8. Reporting Agent
# --------------------------------------------------

def reporting_agent(state: State):
    """Save final evaluation results to JSON."""

    update = {
        **state,
        "status": "complete",
        "message": (
            "Model evaluated on held-out users."
        ),
    }

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUTS / "model_metrics.json"

    metrics_path.write_text(
        json.dumps(update, indent=2)
    )

    save_status(update, "reporting_agent")

    return update


# --------------------------------------------------
# 9. Build LangGraph workflow
# --------------------------------------------------

def build_graph():
    """Connect the five agents into a workflow."""

    graph = StateGraph(State)

    agents = [
        ("data_agent", data_agent),
        ("split_agent", split_agent),
        ("training_agent", training_agent),
        ("evaluation_agent", evaluation_agent),
        ("reporting_agent", reporting_agent),
    ]

    for name, function in agents:
        graph.add_node(name, function)

    graph.add_edge(
        START,
        "data_agent",
    )

    graph.add_edge(
        "data_agent",
        "split_agent",
    )

    graph.add_edge(
        "split_agent",
        "training_agent",
    )

    graph.add_edge(
        "training_agent",
        "evaluation_agent",
    )

    graph.add_edge(
        "evaluation_agent",
        "reporting_agent",
    )

    graph.add_edge(
        "reporting_agent",
        END,
    )

    return graph.compile()


# --------------------------------------------------
# 10. Run workflow
# --------------------------------------------------

if __name__ == "__main__":

    result = build_graph().invoke(
        {"status": "started"}
    )

    print(
        json.dumps(result, indent=2)
    )
