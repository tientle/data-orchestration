FROM apache/airflow:2.10.5-python3.11

COPY requirements.txt /tmp/project-requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==2.10.5" \
    -r /tmp/project-requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
