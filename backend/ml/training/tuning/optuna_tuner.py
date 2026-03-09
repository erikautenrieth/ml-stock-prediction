import mlflow
import numpy as np
import optuna
import structlog
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from backend.core.config import settings
from backend.ml.training.base import Trainer

logger = structlog.get_logger(__name__)


def tune(
    trainer: Trainer,
    x_train,
    y_train,
    x_test,
    y_test,
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
    """
    horizon = settings.stock.prediction_horizon_days
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)

    def objective(trial: optuna.Trial) -> float:
        params = trainer.search_space(trial)
        trial.set_user_attr("processed_params", params)
        model = trainer.build(params)

        # Walk-forward CV on training data — each fold strictly respects time order.
        fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
        for train_idx, val_idx in tscv.split(x_train):
            model.fit(x_train[train_idx], y_train[train_idx])
            preds = model.predict(x_train[val_idx])
            y_val = y_train[val_idx]
            fold_acc.append(accuracy_score(y_val, preds))
            fold_prec.append(precision_score(y_val, preds, zero_division=0))
            fold_rec.append(recall_score(y_val, preds, zero_division=0))
            fold_f1.append(f1_score(y_val, preds, zero_division=0))

        # Store all metrics as trial attributes so they appear in MLflow
        trial.set_user_attr("cv_precision", float(np.mean(fold_prec)))
        trial.set_user_attr("cv_recall", float(np.mean(fold_rec)))
        trial.set_user_attr("cv_f1", float(np.mean(fold_f1)))

        # Optimise on F1 — more robust than accuracy when classes are slightly imbalanced
        # (UP days slightly outnumber DOWN days in a long bull market like S&P500).
        return float(np.mean(fold_f1))

    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="cv_f1",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    best_params = study.best_trial.user_attrs["processed_params"]

    best = study.best_trial
    logger.info(
        "tuning_complete",
        best_cv_f1=best.value,
        best_cv_precision=best.user_attrs.get("cv_precision"),
        best_cv_recall=best.user_attrs.get("cv_recall"),
        best_params=best_params,
        n_trials=len(study.trials),
    )

    return best_params
