from pathlib import Path
import joblib
from typing import Dict
import numpy as np

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
VERSION_FILE = Path(__file__).resolve().parents[1] / "artifacts" / "model_version.txt"

class ModelWrapper:
    def __init__(self, path: str = None):
        path = path or str(ARTIFACT)
        self._load(path)

    def _load(self, path):
        try:
            self.pipeline = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
        try:
            self.version = VERSION_FILE.read_text().strip()
        except Exception:
            self.version = "unknown"

    def predict(self, X_dict: Dict[str, float]) -> float:
        # order features exactly
        feature_order = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        try:
            x = np.array([X_dict[f] for f in feature_order], dtype=float).reshape(1, -1)
        except KeyError as e:
            raise KeyError(f"Missing feature in request: {e}")
        pred = float(self.pipeline.predict(x)[0])
        return pred
