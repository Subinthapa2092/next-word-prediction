# # Slim CPU image: training happens in notebooks/CI, this image only serves predictions.
# FROM python:3.11-slim AS base

# WORKDIR /app
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# RUN apt-get update && apt-get install -y --no-install-recommends curl \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY config.py app.py ./
# COPY src ./src
# COPY static ./static
# COPY templates ./templates
# # models/ is a bind mount in docker-compose.yml, not baked into the image (weights change per training run)

# RUN useradd --create-home appuser && mkdir -p models data/raw data/processed outputs \
#     && chown -R appuser:appuser /app
# USER appuser

# EXPOSE 8000
# HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
#     CMD curl -f http://localhost:8000/health || exit 1

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Slim CPU image: training happens in notebooks/CI, this image only serves predictions.
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY config.py app.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates

# Include the trained model and tokenizer in the Docker image
COPY models/gru.keras ./models/gru.keras
COPY models/tokenizer.json ./models/tokenizer.json

RUN useradd --create-home appuser \
    && mkdir -p models data/raw data/processed outputs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]