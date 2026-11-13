"""Scheduled DAG with parallel ingestion and cleaning branches."""
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from scripts import pipeline

@dag(
    dag_id='instacart_reorder_pipeline',
    schedule='@weekly',
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 2, 'retry_delay': timedelta(minutes=1)},
    tags=['instacart', 'data-engineering'],
)
def instacart_reorder_pipeline():
    @task
    def ingest(name):
        pipeline.ingest(name)

    @task
    def clean_orders():
        pipeline.clean_orders()

    @task
    def clean_products():
        pipeline.clean_products()

    @task
    def clean_purchases(kind):
        pipeline.clean_purchases(kind)

    @task
    def merge_and_engineer():
        pipeline.build_features()

    @task
    def load():
        pipeline.load_database()

    @task
    def analyze():
        pipeline.analyze()

    # Parallel source validation (actual concurrency depends on executor).
    raw_orders = ingest.override(task_id='ingest_orders')('orders')
    raw_prior = ingest.override(task_id='ingest_prior')('order_products__prior')
    raw_train = ingest.override(task_id='ingest_train')('order_products__train')
    raw_products = ingest.override(task_id='ingest_products')('products')
    raw_aisles = ingest.override(task_id='ingest_aisles')('aisles')
    raw_departments = ingest.override(task_id='ingest_departments')('departments')

    cleaned_orders = clean_orders()
    raw_orders >> cleaned_orders
    cleaned_products = clean_products()
    [raw_products, raw_aisles, raw_departments] >> cleaned_products
    cleaned_prior = clean_purchases.override(task_id='clean_prior')('prior')
    cleaned_train = clean_purchases.override(task_id='clean_train')('train')
    [raw_prior, cleaned_orders] >> cleaned_prior
    [raw_train, cleaned_orders] >> cleaned_train
    features = merge_and_engineer()
    [cleaned_orders, cleaned_products, cleaned_prior, cleaned_train] >> features
    loaded = load()
    features >> loaded >> analyze()

instacart_reorder_pipeline()
