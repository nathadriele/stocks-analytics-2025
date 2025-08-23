# ==============================
# Stock Market Analytics - Docker
# Base: Python 3.11 slim
# ==============================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY LICENSE /app/LICENSE

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -e .

COPY . /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

RUN mkdir -p /app/data/raw /app/data/processed /app/data/analytics /app/data/signals /app/data/backtests \
    /app/models /app/reports/img /app/storage

ENTRYPOINT ["python", "-m", "src.app.run_daily_pipeline"]
