import mlflow
import numpy as np
import optuna
import structlog
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from backend.core.config import settings
from backend.ml.training.base import Trainer

logger = structlog.get_logger(__name__)


def tune(
    trainer: Trainer,
    x_train,
    y_train,
    *,
    sample_weights: np.ndarray | None = None,
    n_trials: int = 50,
) -> dict:
    """Run Optuna hyperparameter search, logging every trial to MLflow via DagsHub.

    Uses TimeSeriesSplit cross-validation on the training data instead of a single
    hold-out evaluation. This prevents overfitting hyperparameters to one specific
    test period and yields more robust parameter estimates on financial time series.
    (Raffinot 2017; López de Prado 2018 "Advances in Financial Machine Learning").

    gap=prediction_horizon_days is passed to TimeSeriesSplit so that val fold labels
    do not overlap with the latest training fold labels — a.k.a. "embargo" (López de
    Prado 2018).  Labels at day t reach forward h days; omitting h rows between the
    end of train and start of val eliminates this serial label correlation.

    sample_weights: exponential-decay weights produced by ``Trainer.prepare()``.
    Passed through to ``model.fit()`` in every CV fold so the tuner evaluates
    parameter sets under the same weighting scheme used for final training.
    """
    # Keep Optuna's internal trial logs out of console; we emit compact progress logs below.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective_name = "cv_roc_auc"
    horizon = settings.stock.prediction_horizon_days
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)

    def objective(trial: optuna.Trial) -> float:
        params = trainer.search_space(trial)
        trial.set_user_attr("processed_params", params)
        model = trainer.build(params)

        # Walk-forward CV on training data — each fold strictly respects time order.
        fold_acc, fold_prec, fold_rec, fold_auc = [], [], [], []
        for train_idx, val_idx in tscv.split(x_train):
            fold_weights = sample_weights[train_idx] if sample_weights is not None else None
            x_fold_train = x_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            x_fold_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            model.fit(x_fold_train, y_fold_train, sample_weight=fold_weights)
            preds = model.predict(x_fold_val)
            proba = model.predict_proba(x_fold_val)[:, 1]
            fold_acc.append(accuracy_score(y_val, preds))
            fold_prec.append(precision_score(y_val, preds, zero_division=0))
            fold_rec.append(recall_score(y_val, preds, zero_division=0))
            # roc_auc_score needs both classes present in y_val
            if len(np.unique(y_val)) > 1:
                fold_auc.append(roc_auc_score(y_val, proba))

        # Store all metrics as trial attributes so they appear in MLflow
        trial.set_user_attr("cv_precision", float(np.mean(fold_prec)))
        trial.set_user_attr("cv_recall", float(np.mean(fold_rec)))
        mean_auc = float(np.mean(fold_auc)) if fold_auc else 0.5
        trial.set_user_attr("cv_roc_auc", mean_auc)

        # Optimise on ROC-AUC — immune to class imbalance, measures ranking
        # quality across all thresholds.  A model that predicts "always UP"
        # scores exactly 0.5 → no shortcut possible.
        return mean_auc

    def trial_log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = float(trial.value) if trial.value is not None else None
        best_value = float(study.best_value) if study.best_trial is not None else None
        # Keep trial history in the current MLflow run without creating nested runs.
        step = trial.number
        mlflow.log_metrics(
            {
                "trial_cv_roc_auc": round(float(trial.user_attrs.get("cv_roc_auc", 0.0)), 6),
                "trial_cv_precision": round(float(trial.user_attrs.get("cv_precision", 0.0)), 6),
                "trial_cv_recall": round(float(trial.user_attrs.get("cv_recall", 0.0)), 6),
            },
            step=step,
        )
        is_best = study.best_trial is not None and study.best_trial.number == trial.number
        # Log first, improved, and every 5th trial to keep console readable.
        should_log = trial.number == 0 or is_best or (trial.number + 1) % 5 == 0
        if should_log:
            logger.info(
                "tuning_progress",
                trial_number=trial.number,
                objective_name=objective_name,
                objective_value=round(value, 6) if value is not None else None,
                best_trial=study.best_trial.number if study.best_trial is not None else None,
                best_objective_value=round(best_value, 6) if best_value is not None else None,
                improved=is_best,
            )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[trial_log_callback])

    best_params = study.best_trial.user_attrs["processed_params"]

    best = study.best_trial
    logger.info(
        "tuning_complete",
        objective_name=objective_name,
        best_objective_value=round(float(best.value), 6),
        best_cv_precision=round(float(best.user_attrs.get("cv_precision", 0.0)), 6),
        best_cv_recall=round(float(best.user_attrs.get("cv_recall", 0.0)), 6),
        best_trial=best.number,
        best_params_logged_to_mlflow=True,
        n_trials=len(study.trials),
    )

    return best_params
