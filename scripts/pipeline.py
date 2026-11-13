"""Instacart ingestion, feature preparation, database loading, and basic analysis.

All steps are deterministic and safe to rerun for the same source files.
"""
from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
OUTPUTS = ROOT / 'outputs'
FILES = ['orders', 'order_products__prior', 'order_products__train', 'products', 'aisles', 'departments']


def database_url():
    return os.getenv('DATABASE_URL', 'postgresql+psycopg2://instacart:instacart@localhost:5432/instacart')


def ingest(table):
    if table not in FILES:
        raise ValueError(f'Unexpected table: {table}')
    file = RAW / f'{table}.csv'
    if not file.is_file():
        raise FileNotFoundError(f'Missing {file}. Download Instacart CSVs to data/raw first.')
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f'Validated source: {file} ({file.stat().st_size:,} bytes)')


def clean_orders():
    orders = pd.read_csv(RAW / 'orders.csv')
    required = {'order_id', 'user_id', 'eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'}
    assert required.issubset(orders.columns)
    orders = orders.drop_duplicates('order_id')
    # Use customers with a labeled next order; reproducible sample by sorted user ID.
    eligible = sorted(orders.loc[orders.eval_set.eq('train'), 'user_id'].unique())
    sample_n = int(os.getenv('SAMPLE_USERS', '1500'))
    selected = set(eligible[:sample_n])
    orders = orders[orders.user_id.isin(selected)].copy()
    orders.to_csv(PROCESSED / 'orders.csv', index=False)
    print(f'Cleaned {len(orders):,} orders from {len(selected):,} users')


def clean_products():
    products = pd.read_csv(RAW / 'products.csv').drop_duplicates('product_id')
    aisles = pd.read_csv(RAW / 'aisles.csv').drop_duplicates('aisle_id')
    departments = pd.read_csv(RAW / 'departments.csv').drop_duplicates('department_id')
    products = products.merge(aisles, on='aisle_id', validate='many_to_one')
    products = products.merge(departments, on='department_id', validate='many_to_one')
    products.to_csv(PROCESSED / 'products.csv', index=False)
    print(f'Cleaned {len(products):,} products')


def clean_purchases(kind):
    if kind not in ('prior', 'train'):
        raise ValueError(kind)
    orders = pd.read_csv(PROCESSED / 'orders.csv', usecols=['order_id', 'eval_set'])
    relevant_ids = set(orders.loc[orders.eval_set.eq(kind), 'order_id'])
    source = RAW / f'order_products__{kind}.csv'
    output = PROCESSED / f'purchases_{kind}.csv'
    # Chunking avoids loading millions of unrelated rows into memory.
    first = True
    count = 0
    for chunk in pd.read_csv(source, chunksize=250_000):
        subset = chunk[chunk.order_id.isin(relevant_ids)].dropna(subset=['order_id', 'product_id'])
        subset.to_csv(output, mode='w' if first else 'a', index=False, header=first)
        first = False
        count += len(subset)
    if first:
        pd.DataFrame(columns=['order_id', 'product_id', 'add_to_cart_order', 'reordered']).to_csv(output, index=False)
    print(f'Cleaned {count:,} {kind} order-product rows')


def build_features():
    """One candidate row per user/product purchased before the labeled next order.

    Features come ONLY from prior orders; labels come from the next train order.
    Products never previously purchased are outside this model's candidate universe.
    """
    orders = pd.read_csv(PROCESSED / 'orders.csv')
    prior = pd.read_csv(PROCESSED / 'purchases_prior.csv')
    train = pd.read_csv(PROCESSED / 'purchases_train.csv')
    products = pd.read_csv(PROCESSED / 'products.csv')
    prior_orders = orders.loc[orders.eval_set.eq('prior'), ['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day']]
    history = prior.merge(prior_orders, on='order_id', validate='many_to_one')
    features = history.groupby(['user_id', 'product_id'], as_index=False).agg(
        times_purchased=('order_id', 'count'),
        last_purchase_order=('order_number', 'max'),
        mean_cart_position=('add_to_cart_order', 'mean'),
        historical_reorder_rate=('reordered', 'mean'),
    )
    user_history = prior_orders.groupby('user_id', as_index=False).agg(
        previous_order_count=('order_id', 'count'),
        last_order_number=('order_number', 'max'),
    )
    features = features.merge(user_history, on='user_id', validate='many_to_one')
    features['orders_since_last_purchase'] = features['last_order_number'] - features['last_purchase_order']
    next_orders = orders.loc[orders.eval_set.eq('train'), ['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
    next_orders = next_orders.rename(columns={'order_dow': 'next_order_dow', 'order_hour_of_day': 'next_order_hour'})
    features = features.merge(next_orders.drop(columns='order_id'), on='user_id', validate='many_to_one')
    labels = train.merge(next_orders[['order_id', 'user_id']], on='order_id', validate='many_to_one')[['user_id', 'product_id']].drop_duplicates()
    labels['target_reordered'] = 1
    features = features.merge(labels, on=['user_id', 'product_id'], how='left', validate='one_to_one')
    features['target_reordered'] = features['target_reordered'].fillna(0).astype('int8')
    features = features.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', validate='many_to_one')
    features['days_since_prior_order'] = features['days_since_prior_order'].fillna(0)
    if features.empty or features.target_reordered.nunique() != 2:
        raise ValueError('Not enough candidate rows or class diversity; increase SAMPLE_USERS')
    features.to_csv(PROCESSED / 'model_features.csv', index=False)
    print(f'Feature table: {len(features):,} candidate products, positive rate {features.target_reordered.mean():.3f}')


def load_database():
    engine = create_engine(database_url())
    df = pd.read_csv(PROCESSED / 'model_features.csv')
    # Replace makes repeated DAG runs idempotent for this small teaching dataset.
    df.to_sql('instacart_reorder_features', engine, if_exists='replace', index=False, chunksize=3000, method='multi')
    print(f'Loaded {len(df):,} rows to PostgreSQL')


def analyze():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url())
    query = text('''SELECT department_id, COUNT(*) AS candidates,
        AVG(target_reordered::float) AS reorder_rate
        FROM instacart_reorder_features
        GROUP BY department_id ORDER BY candidates DESC LIMIT 15''')
    with engine.connect() as conn:
        summary = pd.read_sql(query, conn)
    summary.to_csv(OUTPUTS / 'department_reorder_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['ingest', 'clean_orders', 'clean_products', 'clean_purchases', 'build_features', 'load_database', 'analyze'])
    parser.add_argument('--table')
    args = parser.parse_args()
    if args.step == 'ingest': ingest(args.table)
    elif args.step == 'clean_orders': clean_orders()
    elif args.step == 'clean_products': clean_products()
    elif args.step == 'clean_purchases': clean_purchases(args.table)
    elif args.step == 'build_features': build_features()
    elif args.step == 'load_database': load_database()
    elif args.step == 'analyze': analyze()
