import pandas as pd
import structlog

from backend.core.config import settings
from backend.core.prediction.mlflow_predict import MLflowPredictor
from backend.core.schemas import PredictionResult
from backend.infra.database import get_data_store

logger = structlog.get_logger(__name__)


def predict_latest() -> PredictionResult:
    """Load latest features, predict with the registered model, store result."""
    store = get_data_store()
    df = store.load_features()

    last_row = df.drop("Target", axis=1, errors="ignore").tail(1)
    logger.info("predicting", date=str(last_row.index[0]))

    predictor = MLflowPredictor()
    pred = predictor.predict(last_row)[0]
    proba = predictor.predict_proba(last_row)[0]
    confidence = float(max(proba))

    # Grab closing price at prediction date for persistence
    try:
        raw_df = store.load_raw()
        entry_price = float(raw_df.loc[last_row.index[0], "Close"])
    except Exception:
        entry_price = None

    result = PredictionResult(
        symbol=settings.stock.symbol,
        prediction=int(pred),
        confidence=confidence,
        horizon_days=settings.stock.prediction_horizon_days,
    )

    pred_df = pd.DataFrame(
        {
            "Date": last_row.index,
            "prediction": [result.prediction],
            "confidence": [result.confidence],
            "symbol": [result.symbol],
            "horizon_days": [result.horizon_days],
            "entry_price": [entry_price],
        }
    )
    pred_df.set_index("Date", inplace=True)

    # Append to existing predictions if available
    try:
        existing = store.load_predictions(days=10000)
        existing.index = pd.to_datetime(existing.index).normalize()
        pred_df.index = pd.to_datetime(pred_df.index).normalize()
        pred_df = pd.concat([existing, pred_df])
        pred_df = pred_df[~pred_df.index.duplicated(keep="last")]
        pred_df = pred_df.sort_index()
    except FileNotFoundError:
        pass

    store.save_predictions(pred_df)

    direction = "UP" if result.prediction == 1 else "DOWN"
    logger.info(
        "prediction_complete",
        direction=direction,
        confidence=f"{result.confidence:.2%}",
        horizon=result.horizon_days,
    )

    return result


if __name__ == "__main__":
    predict_latest()
