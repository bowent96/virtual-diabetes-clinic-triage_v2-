from fastapi import FastAPI, HTTPException
from app.schema import PredictRequest, PredictResponse
from app.model import ModelWrapper
import os

MODEL_PATH = os.environ.get("MODEL_PATH")  # optional override

app = FastAPI(title="Virtual Diabetes Triage API")

# load model at startup
try:
    model = ModelWrapper(path=MODEL_PATH)
except Exception as e:
    # we still start the server but health will indicate model load error
    model = None
    load_error = str(e)
else:
    load_error = None


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "model_version": None, "error": load_error}
    return {"status": "ok", "model_version": model.version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        pred = model.predict(req.dict())
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # provide JSON error with traceback for observability while in dev (strip or reduce in prod)
        raise HTTPException(status_code=500, detail=f"internal error: {e}")
    return {"prediction": pred, "model_version": model.version}
