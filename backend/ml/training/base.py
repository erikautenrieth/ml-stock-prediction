from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from backend.core.config import settings

logger = structlog.get_logger(__name__)


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
        embargo: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
        """Split data into train/test sets with rolling-window cutoff and sample weights.

        Rolling window: if ``settings.training.training_years`` is set, rows older than
        that many years from today are discarded before splitting.  This removes
        ancient market regimes (dot-com, 2008 crisis) that may hurt model accuracy.

        Sample weights: exponential decay so recent observations receive more weight.
        The half-life is expressed as a fraction of the training set length via
        ``settings.training.sample_weight_halflife`` (default 0.33 → the oldest third of
        training rows have ≤ 50 % the weight of the newest rows).

        embargo: number of trailing training rows to drop.
        The label for row t is computed as (close[t+h] > close[t]), so the last
        `h` training rows look forward into the test period — their labels are
        contaminated by future test-set prices.  Dropping these rows ("purging"
        in López de Prado 2018) removes this lookahead bias.

        Returns (x_train, x_test, y_train, y_test, sample_weights).
        """
        df = df.dropna(subset=["Target"])

        # --- Rolling-window cutoff ---
        training_years = settings.training.training_years
        if training_years is not None and hasattr(df.index, "year"):
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=training_years)
            rows_before = len(df)
            df = df[df.index >= cutoff]
            logger.info(
                "rolling_window_applied",
                training_years=training_years,
                cutoff=str(cutoff.date()),
                rows_dropped=rows_before - len(df),
                rows_kept=len(df),
            )

        target = df["Target"].astype(int)
        features = df.drop("Target", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            shuffle=False,
        )
        if embargo > 0:
            x_train = x_train.iloc[:-embargo]
            y_train = y_train.iloc[:-embargo]
        self._feature_names = features.columns.tolist()

        # --- Exponential sample weights ---
        sample_weights = self._compute_sample_weights(len(x_train))

        return x_train, x_test, y_train, y_test, sample_weights

    @staticmethod
    def _compute_sample_weights(n: int) -> np.ndarray:
        """Exponential-decay weights: recent rows get more influence.

        half_life = n * halflife_fraction  →  the oldest ``half_life`` rows
        carry ≤ 50 % the weight of the newest rows.
        """
        halflife_frac = settings.training.sample_weight_halflife
        half_life = max(int(n * halflife_frac), 1)
        weights = np.exp(np.linspace(-n / half_life, 0, n))
        # Normalise so mean weight ≈ 1 (doesn't change the fit, but keeps
        # metrics comparable with unweighted baselines).
        weights /= weights.mean()
        return weights
