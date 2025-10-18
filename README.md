# Virtual Diabetes Clinic — Triage ML Service

Minimal reproducible project for the assignment.

## Overview
- v0.1: baseline StandardScaler + LinearRegression
- v0.2: improved model (RandomForest + SelectKBest)
- Docker image bakes the trained model into `artifacts/model.joblib`.
- API: FastAPI
  - `GET /health` -> `{"status":"ok", "model_version":"v0.1-42"}`
  - `POST /predict` -> `{"prediction": <float>, "model_version": "..."}`

## Exact request body (10 features)
```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}

