from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Predictor(ABC):
    """Abstract base for making predictions with a trained model."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...
