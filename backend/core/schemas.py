from pydantic import BaseModel


class TrainResult(BaseModel):
    model_name: str
    model_uri: str
    accuracy: float
    f1: float
    params: dict
    run_id: str


class PredictionResult(BaseModel):
    symbol: str
    prediction: int  # 1 = up, 0 = down
    confidence: float
    horizon_days: int
