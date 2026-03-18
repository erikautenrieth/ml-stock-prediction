import mlflow
import numpy as np
import pandas as pd
import structlog

from backend.core.config import settings
from backend.core.prediction.base import Predictor

logger = structlog.get_logger(__name__)


class MLflowPredictor(Predictor):
    """Load a model from the DagsHub MLflow model registry and predict."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.mlflow.model_name
        self._model = None

    def load_model(self) -> None:
        model_uri = f"models:/{self._model_name}/latest"
        logger.info("loading_model", uri=model_uri)
        self._model = mlflow.sklearn.load_model(model_uri)
        logger.info("model_loaded", model_name=self._model_name)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            self.load_model()
        return self._model.predict(features.values)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            self.load_model()
        return self._model.predict_proba(features.values)
