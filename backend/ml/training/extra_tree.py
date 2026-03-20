import optuna
from sklearn.ensemble import ExtraTreesClassifier

from backend.ml.training.base import Trainer


class ExtraTreesTrainer(Trainer):
    def name(self) -> str:
        return "ExtraTreesClassifier"

    def default_params(self) -> dict:
        return {
            # Oshiro et al. (2012): variance saturates around 128–512 trees;
            # 300 is a well-supported sweet spot.
            "n_estimators": 300,
            # Geurts et al. (2006): fully-grown trees (None) are the theoretical default;
            # the extreme randomization already acts as regularization.
            "max_depth": None,
            "min_samples_split": 2,  # sklearn default; low impact (Probst et al. 2018)
            "min_samples_leaf": 1,  # sklearn default; Extra Trees' randomization handles overfitting  # noqa: E501
            "max_features": "sqrt",  # Geurts et al. (2006): sqrt(p) recommended for classification
            "random_state": 42,
            "n_jobs": -1,
        }

    def search_space(self, trial: optuna.Trial) -> dict:
        # --- limit_depth ---
        # Geurts et al. (2006): no depth limit is default, but depth restriction can
        # regularize on noisy financial time-series data.
        limit_depth = trial.suggest_categorical("limit_depth", [True, False])

        # --- bootstrap ---
        # Extra Trees by design uses the full dataset (bootstrap=False, Geurts 2006).
        # Enabling it adds sub-sampling variance reduction at the cost of bias.
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        return {
            # n_estimators — log-scale search justified by Probst et al. (2018):
            # gains are large at the low end (100→200) and negligible at the high end.
            # Range 100–1000 covers the full useful regime (Oshiro et al. 2012).
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
            # max_depth — if constrained: range 3–40 captures shallow (high-bias)
            # to near-unlimited (low-bias) trees.
            "max_depth": trial.suggest_int("max_depth", 3, 40) if limit_depth else None,
            # min_samples_split — moderate importance (Probst et al. 2018); wider
            # range 2–20 avoids artificially restricting the search.
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            # min_samples_leaf — TOP-2 most important parameter (Geurts 2006;
            # Probst et al. 2018). Range 1–20 spans from no regularization to
            # heavy leaf-smoothing, which is critical for noisy stock data.
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            # max_features — THE most important hyperparameter for Extra Trees
            # (Geurts et al. 2006; Probst et al. 2018). Log-uniform sampling
            # concentrates trials near smaller fractions (where sqrt/log2 lie)
            # while still exploring higher values. Range 0.05–1.0 covers all
            # practically relevant sub-space sizes.
            "max_features": trial.suggest_float("max_features", 0.05, 1.0, log=True),
            # criterion — gini vs entropy makes negligible empirical difference
            # (Fernández-Delgado et al. 2014, JMLR). Kept for completeness.
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "bootstrap": bootstrap,
            # max_samples — only valid with bootstrap=True. Lower bound 0.6
            # avoids too few samples per tree; Breiman (2001) suggests 0.632.
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0) if bootstrap else None,
        }

    def build(self, params: dict | None = None) -> ExtraTreesClassifier:
        p = {**self.default_params(), **(params or {})}
        # Always enforce these
        p["random_state"] = 42
        p["n_jobs"] = -1
        p["class_weight"] = "balanced"
        # Clean optuna meta-params that aren't model params
        p.pop("limit_depth", None)
        # max_samples only valid with bootstrap=True
        if not p.get("bootstrap", False):
            p.pop("max_samples", None)
        return ExtraTreesClassifier(**p)
