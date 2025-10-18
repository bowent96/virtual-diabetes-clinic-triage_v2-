"""
Train script.
Saves:
 - artifacts/model.joblib
 - artifacts/metrics.json
 - artifacts/model_version.txt
Deterministic: seed set, pinned scikit-learn versions assumed.
"""

import json
import os
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

from datasets import load_diabetes_df

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def train(args):
    X, y = load_diabetes_df()
    # deterministic split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    if args.version == "v0.1":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    else:
        # v0.2: feature selection + scaling + RandomForest (or Ridge if chosen)
        if args.model == "rf":
            model = RandomForestRegressor(
                n_estimators=args.n_estimators, random_state=args.seed, n_jobs=1
            )
        elif args.model == "ridge":
            model = Ridge(random_state=args.seed)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=1)

        pipeline = Pipeline(
            [
                ("select", SelectKBest(score_func=f_regression, k=args.k)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    pipeline.fit(X_train, y_train)
    preds_val = pipeline.predict(X_val)
    rmse = float(mean_squared_error(y_val, preds_val, squared=False))

    # If asked, compute binary high-risk flag at threshold = 75th percentile on train
    threshold = float(np.percentile(y_train, 75))
    high_pred = (pipeline.predict(X_val) >= threshold).astype(int)
    high_true = (y_val.values >= threshold).astype(int)
    precis, recall, f1, _ = precision_recall_fscore_support(
        high_true, high_pred, average="binary", zero_division=0
    )
    metrics = {
        "rmse": rmse,
        "threshold_75pct_train": threshold,
        "highrisk_precision": float(precis),
        "highrisk_recall": float(recall),
        "highrisk_f1": float(f1),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }

    model_version = f"{args.version}-{args.seed}"
    joblib.dump(pipeline, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACT_DIR / "model_version.txt").write_text(model_version)

    print("Saved model and metrics to artifacts/")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, choices=["v0.1", "v0.2"], default="v0.1")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", type=str, choices=["rf", "ridge"], default="rf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    train(args)
