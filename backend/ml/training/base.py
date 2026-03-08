from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split


class Trainer(ABC):
    """Abstract base for model training."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build(self, params: dict | None = None):
        """Return an unfitted sklearn-compatible estimator."""
        ...

    @abstractmethod
    def default_params(self) -> dict: ...

    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> dict:
        """Define Optuna search space for hyperparameter tuning."""
        ...

    def prepare(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = df["Target"]
        features = df.drop("Target", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(
            features.values,
            target.values,
            test_size=test_size,
            shuffle=False,
        )
        self._feature_names = features.columns.tolist()
        return x_train, x_test, y_train, y_test
