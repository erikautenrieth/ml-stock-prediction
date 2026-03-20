import lightgbm as lgb
import optuna

from backend.ml.training.base import Trainer


class LightGBMTrainer(Trainer):
    """LightGBM gradient-boosted trees.

    Why LightGBM over tree bagging (ExtraTrees / RandomForest):
    - Boosting corrects residuals → better on structured/tabular data
    - Built-in L1 / L2 regularisation prevents overfitting on noisy financial data
    - Native handling of missing values (no imputation needed)
    - Much faster training (histogram-based, leaf-wise growth)
    - ``sample_weight`` supported natively in ``fit()``
    """

    def name(self) -> str:
        return "LightGBMClassifier"

    def default_params(self) -> dict:
        return {
            # --- Boosting budget ---
            # Ke et al. (2017): more iterations + lower LR generalises better.
            # 500 trees at lr=0.03 is a conservative starting point.
            "n_estimators": 500,
            "learning_rate": 0.03,
            # --- Tree structure ---
            # LightGBM grows leaf-wise; num_leaves is THE primary complexity
            # control (Ke et al. 2017). 31 is the default; keep max_depth as
            # a safety cap so depth ≤ log2(num_leaves).
            "num_leaves": 31,
            "max_depth": -1,
            # --- Regularisation ---
            # Gu et al. (2020, "Empirical Asset Pricing via ML"): strong
            # regularisation is critical for noisy financial targets.
            "min_child_samples": 20,
            "min_gain_to_split": 0.01,  # suppress noise-splits (Ke et al. 2017)
            "reg_alpha": 0.1,  # L1 — promotes sparsity in noisy feature space
            "reg_lambda": 1.0,  # L2 — shrinks leaf values; key for financial data
            "path_smooth": 1.0,  # smooth leaf output when samples are few
            # --- Stochastic regularisation ---
            # Row + column sub-sampling per iteration adds variance which
            # reduces overfitting (Ke et al. 2017).
            "subsample": 0.8,
            "subsample_freq": 1,  # MUST be >0 for subsample to take effect
            "colsample_bytree": 0.8,
            # --- Histogram binning ---
            # Lower max_bin adds implicit regularisation by discretising
            # feature values more coarsely (Ke et al. 2017). 127 is a good
            # balance for medium-noise tabular data.
            "max_bin": 127,
            # --- Class balance ---
            # S&P500 has slight UP bias in long bull markets (Ampomah 2020).
            "class_weight": "balanced",
            # --- Infra ---
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def search_space(self, trial: optuna.Trial) -> dict:
        # --- Tree depth + leaves ---
        # Constraint: num_leaves ≤ 2^max_depth (Ke et al. 2017).
        # Without this, LightGBM silently clips leaves and Optuna wastes trials.
        max_depth = trial.suggest_int("max_depth", 3, 12)
        max_leaves_for_depth = 2**max_depth
        min_leaves_for_depth = min(15, max_leaves_for_depth)
        num_leaves = trial.suggest_int(
            "num_leaves",
            min_leaves_for_depth,
            min(max_leaves_for_depth, 255),
        )

        # --- Subsample ---
        # subsample_freq MUST be > 0 for subsample to take effect; many
        # tutorials miss this (Ke et al. 2017, §3.2 GOSS).
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        subsample_freq = 1 if subsample < 1.0 else 0

        return {
            # Boosting budget — log-scale: gains diminish at high end
            # (Probst et al. 2019 "Tunability").
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            # Learning rate — log-uniform: small LR + many trees generalises
            # better on noisy targets (Gu et al. 2020).
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            # min_child_samples — primary overfitting guard for leaf-wise
            # growth (Ke et al. 2017). Higher = more conservative.
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            # min_gain_to_split — prunes splits that barely reduce loss.
            # Critical for noisy financial features where many splits are
            # just fitting noise (Leippold et al. 2022).
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split",
                1e-4,
                1.0,
                log=True,
            ),
            # Row & column sub-sampling
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            # L1 + L2 regularisation — log-uniform covers both "nearly off"
            # and "heavy" regimes (Gu et al. 2020; Rasekhschaffe & Jones 2019).
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            # path_smooth — smooths leaf predictions; helpful when some
            # leaves have few samples (LightGBM docs; analogous to
            # Catboost's leaf estimation method).
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
            # max_bin — lower = more regularisation via coarser histograms
            # (Ke et al. 2017, §2 histogram-based algorithm).
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        }

    def build(self, params: dict | None = None) -> lgb.LGBMClassifier:
        p = {**self.default_params(), **(params or {})}
        # Always enforce infra settings
        p["random_state"] = 42
        p["n_jobs"] = -1
        p["verbosity"] = -1
        return lgb.LGBMClassifier(**p)
