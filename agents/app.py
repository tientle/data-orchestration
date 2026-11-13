
"""Interactive dashboard for the Instacart reorder prediction workflow."""

import json

import streamlit as st

from agents.workflow import build_graph
from scripts.pipeline import OUTPUTS


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Instacart Reorder Prediction",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Instacart Reorder Prediction")
st.caption(
    "An end-to-end data orchestration and machine learning project "
    "using Apache Airflow, PostgreSQL, LangGraph, and Streamlit."
)


# --------------------------------------------------
# Project overview
# --------------------------------------------------

st.header("Project Overview")

st.write(
    "Can we predict which grocery products a customer will purchase again "
    "based on their previous shopping behavior?"
)

st.write(
    "This project processes Instacart transaction data, builds customer-product "
    "features, and trains a machine learning model to predict future reorders."
)

st.info(
    "**Business application:** Reorder predictions could help grocery platforms "
    "personalize shopping recommendations, remind customers about frequently "
    "purchased products, and make repeat shopping more convenient."
)


# --------------------------------------------------
# Data orchestration overview
# --------------------------------------------------

st.header("How the Pipeline Works")

st.write(
    "The project uses two connected workflows. Apache Airflow prepares "
    "the transaction data, while LangGraph coordinates model training "
    "and evaluation."
)

st.markdown(
    """
    **1. Apache Airflow — Data Engineering**

    - Ingest six Instacart CSV tables.
    - Clean and validate transaction and product records.
    - Join customer orders with product information.
    - Engineer features describing historical purchasing behavior.
    - Load the processed data into PostgreSQL.

    **2. LangGraph — Machine Learning**

    - Retrieve and validate the processed data.
    - Split observations into training and testing sets.
    - Train a logistic regression model.
    - Evaluate reorder predictions on held-out customers.
    - Generate model evaluation reports.
    """
)


# --------------------------------------------------
# Agent descriptions
# --------------------------------------------------

AGENTS = {
    "data_agent": {
        "label": "Data Validation",
        "description": (
            "Retrieves engineered features from PostgreSQL and "
            "checks that the dataset contains valid target classes."
        ),
    },
    "split_agent": {
        "label": "Train/Test Split",
        "description": (
            "Separates customers into training and testing groups "
            "so that the same customer does not appear in both."
        ),
    },
    "training_agent": {
        "label": "Model Training",
        "description": (
            "Preprocesses numerical and categorical features "
            "and trains a logistic regression classifier."
        ),
    },
    "evaluation_agent": {
        "label": "Model Evaluation",
        "description": (
            "Evaluates predictions using precision, recall, "
            "F1 score, accuracy, and average precision."
        ),
    },
    "reporting_agent": {
        "label": "Reporting",
        "description": (
            "Saves model evaluation metrics and workflow status "
            "for reproducibility and monitoring."
        ),
    },
}


# --------------------------------------------------
# LangGraph workflow
# --------------------------------------------------

st.header("Agent Workflow")

st.write(
    "Each node represents a specialized processing step. "
    "The agents execute sequentially, passing workflow status "
    "between stages."
)

if st.button("Run Agent Workflow", type="primary"):
    try:
        with st.spinner("Running machine learning workflow..."):
            result = build_graph().invoke({"status": "started"})

        st.success("Agent workflow completed successfully!")
        with st.expander("View workflow output"):
            st.json(result)

    except Exception as error:
        st.error(f"Workflow failed: {error}")


# --------------------------------------------------
# Read workflow status
# --------------------------------------------------

status_file = OUTPUTS / "agent_status.json"

last_node = ""
workflow_status = {}

if status_file.exists():
    try:
        workflow_status = json.loads(status_file.read_text())
        last_node = workflow_status.get("last_node", "")

    except (OSError, json.JSONDecodeError):
        st.warning("Unable to read the latest workflow status.")

if last_node:
    st.write(
        "**Most recently completed step:**",
        AGENTS.get(last_node, {}).get("label", last_node),
    )


# --------------------------------------------------
# Workflow visualization
# --------------------------------------------------

nodes = list(AGENTS.keys())

lines = [
    "digraph G {",
    "rankdir=LR;",
    'graph [bgcolor="transparent", pad="0.3", nodesep="0.5"];',
    'node [shape=box, style="rounded,filled", '
    'fillcolor="#F1E8FA", color="#A78BBF", '
    'fontname="Helvetica", fontsize=12, margin="0.2"];',
    'edge [color="#999999", arrowsize=0.7];',
]

for node in nodes:
    label = AGENTS[node]["label"]

    if node == last_node:
        lines.append(
            f'{node} [label="{label}", '
            'fillcolor="#C7EFD7", color="#68A884"];'
        )
    else:
        lines.append(f'{node} [label="{label}"];')

for first, second in zip(nodes, nodes[1:]):
    lines.append(f"{first} -> {second};")

lines.append("}")

st.graphviz_chart("\n".join(lines), width="stretch")

st.subheader("What Does Each Agent Do?")

for agent in AGENTS.values():
    with st.expander(agent["label"]):
        st.write(agent["description"])


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

st.header("Model Evaluation")

metrics_file = OUTPUTS / "model_metrics.json"

if metrics_file.exists():

    try:
        metrics = json.loads(metrics_file.read_text())

    except (OSError, json.JSONDecodeError):
        metrics = {}
        st.warning("Unable to read model evaluation metrics.")

    if metrics:

        st.write(
            "The model is evaluated on customers excluded from "
            "the training dataset. This helps estimate how well "
            "it generalizes to unseen customers."
        )

        if "rows" in metrics:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Observations",
                f"{metrics.get('rows', 0):,}",
            )

            col2.metric(
                "Training Observations",
                f"{metrics.get('train_rows', 0):,}",
            )

            col3.metric(
                "Testing Observations",
                f"{metrics.get('test_rows', 0):,}",
            )

        st.subheader("Classification Performance")

        metric_columns = st.columns(4)

        metric_definitions = [
            (
                "Precision",
                "precision",
                "Of the products predicted to be reordered, "
                "how many actually were?",
            ),
            (
                "Recall",
                "recall",
                "Of all products actually reordered, "
                "how many did the model identify?",
            ),
            (
                "F1 Score",
                "f1",
                "A measure balancing precision and recall.",
            ),
            (
                "Average Precision",
                "average_precision",
                "Summarizes precision-recall performance "
                "across different prediction thresholds.",
            ),
        ]

        for column, (label, key, description) in zip(
            metric_columns, metric_definitions
        ):
            value = metrics.get(key)

            if value is not None:
                column.metric(label, f"{value:.1%}")
            else:
                column.metric(label, "N/A")

            column.caption(description)

        with st.expander("View all evaluation metrics"):
            st.json(metrics)


        # --------------------------------------------------
        # Interpret model results
        # --------------------------------------------------

        st.subheader("Interpreting the Results")

        precision = metrics.get("precision")
        recall = metrics.get("recall")

        if precision is not None and recall is not None:

            st.write(
                f"At the current classification threshold, "
                f"the model identifies approximately "
                f"**{recall:.1%} of actual reorders**."
            )

            st.write(
                f"However, only **{precision:.1%} of products "
                f"predicted to be reordered** were actually reordered."
            )

            if recall > precision:
                st.info(
                    "**Interpretation:** The model prioritizes "
                    "finding potential reorders, but it also produces "
                    "many false positives. This could be useful for "
                    "generating broad product suggestions, although "
                    "the recommendations may need additional filtering."
                )

            elif precision > recall:
                st.info(
                    "**Interpretation:** The model is more selective "
                    "about predicting reorders, but may miss products "
                    "customers would purchase again."
                )

            else:
                st.info(
                    "The model has similar precision and recall "
                    "at the current classification threshold."
                )

else:
    st.warning(
        "No evaluation results found. Run the agent workflow "
        "to generate model metrics."
    )


# --------------------------------------------------
# Business application
# --------------------------------------------------

st.header("Business Application")

st.write(
    "The model estimates whether a customer is likely to reorder "
    "a product based on their historical purchasing behavior."
)

st.subheader("How Could Instacart Use These Predictions?")

st.markdown(
    """
    **Personalized Recommendations**

    Suggest products customers may want to purchase again
    based on their previous orders.

    **Shopping Convenience**

    Help customers quickly find frequently purchased items
    instead of searching for them manually.

    **Customer Engagement**

    Surface relevant reorder reminders that may encourage
    repeat purchases.
    """
)

st.subheader("Limitations and Next Steps")

st.markdown(
    """
    - **False positives:** The model may recommend products that
      customers do not actually intend to reorder.
    - **Limited features:** Predictions rely on historical
      transactions rather than real-time customer preferences.
    - **Classification threshold:** Adjusting the prediction
      threshold could improve the balance between precision
      and recall.
    - **Model comparison:** Future experiments could compare
      logistic regression against random forests or gradient
      boosting models.
    - **Business evaluation:** Recommendation quality should
      eventually be measured through customer engagement
      and purchasing behavior.
    """
)

st.divider()

st.caption(
    "Instacart Reorder Prediction | "
    "Apache Airflow · PostgreSQL · LangGraph · scikit-learn · Streamlit"
)
