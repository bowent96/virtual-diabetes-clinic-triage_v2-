# builder stage: install deps and train model artifact
FROM python:3.11-slim as builder

WORKDIR /app

# avoid cached apt lists issues
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements-pin.txt requirements-pin.txt
RUN pip install --no-cache-dir -r requirements-pin.txt

# copy training code and train model artifact
COPY training/ training/
COPY training/datasets.py training/datasets.py
WORKDIR /app/training

# produce artifacts in ../artifacts
RUN python train.py --version v0.1 --seed 42

# runtime stage
FROM python:3.11-slim

WORKDIR /app

COPY requirements-pin.txt requirements-pin.txt
RUN pip install --no-cache-dir -r requirements-pin.txt

# copy app and artifacts from builder
COPY app/ app/
COPY --from=builder /app/artifacts/ artifacts/

ENV MODEL_PATH=/app/artifacts/model.joblib
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
