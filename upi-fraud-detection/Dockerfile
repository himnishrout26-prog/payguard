# PayGuard fraud-scoring API
FROM python:3.11-slim

WORKDIR /app

# system deps for lightgbm/xgboost wheels that need libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/

# artifacts/ must already contain model.joblib, calibrator.joblib,
# feature_columns.json, metrics.json -- i.e. run train.py (or
# train_compare.py) BEFORE building this image. The image serves a trained
# model, it doesn't train one at build time.

WORKDIR /app/src
EXPOSE 8000

# PAYGUARD_ARTIFACT_DIR lets you point at a different artifacts folder at
# runtime (e.g. a volume mount) without rebuilding the image -- see config.py
ENV PAYGUARD_ARTIFACT_DIR=/app/artifacts

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
