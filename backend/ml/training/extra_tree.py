import optuna
from sklearn.ensemble import ExtraTreesClassifier

from backend.ml.training.base import Trainer


class ExtraTreesTrainer(Trainer):
    def name(self) -> str:
        return "ExtraTreesClassifier"

    def default_params(self) -> dict:
        return {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }

    def search_space(self, trial: optuna.Trial) -> dict:
        limit_depth = trial.suggest_categorical("limit_depth", [True, False])
        class_weight = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", "none"]
        )
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50) if limit_depth else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": None if class_weight == "none" else class_weight,
        }

    def build(self, params: dict | None = None) -> ExtraTreesClassifier:
        p = {**self.default_params(), **(params or {})}
        # Always enforce these
        p["random_state"] = 42
        p["n_jobs"] = -1
        # Clean optuna meta-params that aren't model params
        p.pop("limit_depth", None)
        return ExtraTreesClassifier(**p)
