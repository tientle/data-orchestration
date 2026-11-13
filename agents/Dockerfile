FROM python:3.11-slim

WORKDIR /app

COPY agents/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY agents/ /app/agents/

ENV PYTHONPATH=/app

CMD ["sleep", "infinity"]