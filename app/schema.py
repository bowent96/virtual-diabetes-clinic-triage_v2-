from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # exact feature names from sklearn.load_diabetes
    age: float = Field(..., description="standardized age")
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class PredictResponse(BaseModel):
    prediction: float
    model_version: str
