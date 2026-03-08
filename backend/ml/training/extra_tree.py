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

    def build(self, params: dict | None = None) -> ExtraTreesClassifier:
        p = {**self.default_params(), **(params or {})}
        # Always enforce these
        p["random_state"] = 42
        p["n_jobs"] = -1
        return ExtraTreesClassifier(**p)
