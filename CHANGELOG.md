# CHANGELOG

## v0.2 — IMPROVED (release candidate)
- Date: YYYY-MM-DD
- Training: RandomForestRegressor with SelectKBest(k=8) + StandardScaler, seed=42.
- Why: added non-linear model + simple feature selection to reduce noise and capture non-linear feature interactions.
- Metrics (validation):
  - RMSE: **<replace_after_training>**
  - High-risk threshold: 75th percentile on training set
  - Precision @ 75th pct: **<replace_after_training>**
  - Recall @ 75th pct: **<replace_after_training>**

## v0.1 — Baseline
- Date: YYYY-MM-DD
- Training: StandardScaler + LinearRegression (seed=42)
- Metrics (validation):
  - RMSE: **<replace_after_training>**
  - High-risk precision / recall not applicable for baseline (or see metrics above)

**How to reproduce metrics locally**
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements-pin.txt`
3. `python training/train.py --version v0.1 --seed 42`
4. View `artifacts/metrics.json` for RMSE and other metrics.
