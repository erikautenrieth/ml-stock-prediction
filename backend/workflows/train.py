import mlflow
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from backend.core.config import settings
from backend.core.schemas import TrainResult
from backend.infra.database import get_data_store
from backend.ml.registry.model_registry import log_and_register
from backend.ml.training import get_trainer, list_trainers
from backend.ml.training.base import Trainer
from backend.ml.training.tuning.optuna_tuner import tune

logger = structlog.get_logger(__name__)


def _fit_calibrated_model(
    trainer: Trainer,
    params: dict,
    x_train,
    y_train,
    sample_weights=None,
):
    """Fit a calibrated classifier with a robust isotonic-first strategy."""
    n_splits = 3 if len(y_train) >= 40 else 2
    max_safe_gap = max(0, (len(y_train) // (n_splits + 1)) - 1)
    gap = min(settings.stock.prediction_horizon_days, max_safe_gap)
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    for method in ("isotonic", "sigmoid"):
        calibrated = CalibratedClassifierCV(
            estimator=trainer.build(params),
            method=method,
            cv=cv,
        )
        try:
            calibrated.fit(x_train, y_train, sample_weight=sample_weights)
            return calibrated, method
        except ValueError as exc:
            if method == "isotonic":
                logger.warning("isotonic_calibration_failed_fallback", error=str(exc))
            else:
                raise


def _log_feature_importance(model, trainer: Trainer) -> None:
    """Extract and log top-15 feature importances to MLflow."""
    try:
        # CalibratedClassifierCV wraps the real estimator
        inner = getattr(model, "estimator", model)
        # sklearn tree ensembles
        importances = getattr(inner, "feature_importances_", None)
        if importances is None and hasattr(inner, "calibrated_classifiers_"):
            # calibrated model: grab from first sub-estimator
            first = inner.calibrated_classifiers_[0].estimator
            importances = getattr(first, "feature_importances_", None)
        if importances is None:
            return

        names = getattr(trainer, "_feature_names", None)
        if names is None:
            names = [f"f{i}" for i in range(len(importances))]

        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        top = ranked[:15]
        for feat, imp in top:
            mlflow.log_metric(f"fi_{feat}", round(float(imp), 6))
        logger.info(
            "feature_importance_top15",
            features={f: round(float(v), 4) for f, v in top},
        )
    except Exception as exc:
        logger.warning("feature_importance_failed", error=str(exc))


def _model_registry_name(trainer: Trainer, experiment: str | None) -> str:
    """Derive the DagsHub model-registry name for this trainer.

    Each trainer type gets its own registry entry so the best model
    *per algorithm* is always preserved::

        best_ExtraTreesClassifier_model
        best_LightGBMClassifier_model
    """
    if experiment:
        return f"{experiment}_{trainer.name()}"
    return f"best_{trainer.name()}_model"


def train_model(
    trainer: Trainer | None = None,
    do_tuning: bool = True,
    n_trials: int = 60,
    force: bool = False,
    experiment: str | None = None,
) -> TrainResult:
    """Full training pipeline: load features → tune → train → log to DagsHub.

    Args:
        trainer: Any ``Trainer`` subclass. Defaults to ExtraTrees.
        force: Register model even if it doesn't beat the current best.
        experiment: Optional experiment name. Sets a separate MLflow experiment
            and model registry name so different approaches don't collide.
    """
    if trainer is None:
        trainer = get_trainer("extra_trees")

    model_name = _model_registry_name(trainer, experiment)

    # Experiment isolation: each experiment gets its own MLflow experiment
    # and model name so models from different approaches never overwrite.
    if experiment:
        mlflow.set_experiment(experiment)
    logger.info("train_start", model=trainer.name(), registry_name=model_name)

    store = get_data_store()
    df = store.load_features()
    logger.info("features_loaded", rows=len(df), columns=len(df.columns))
    horizon = settings.stock.prediction_horizon_days
    x_train, x_test, y_train, y_test, sample_weights = trainer.prepare(df, embargo=horizon)

    # --- Class distribution sanity check ---
    n_up_train = int(y_train.sum())
    n_down_train = len(y_train) - n_up_train
    n_up_test = int(y_test.sum())
    n_down_test = len(y_test) - n_up_test
    logger.info(
        "class_distribution",
        train_up=n_up_train,
        train_down=n_down_train,
        train_pct_up=round(n_up_train / len(y_train) * 100, 1),
        test_up=n_up_test,
        test_down=n_down_test,
        test_pct_up=round(n_up_test / len(y_test) * 100, 1),
    )

    with mlflow.start_run(run_name=f"train_{trainer.name()}") as run:
        if do_tuning:
            logger.info("starting_tuning", n_trials=n_trials)
            best_params = tune(
                trainer,
                x_train,
                y_train,
                sample_weights=sample_weights,
                n_trials=n_trials,
            )
        else:
            best_params = trainer.default_params()

        logger.info(
            "training_final_model",
            model=trainer.name(),
            tuned=do_tuning,
            n_params=len(best_params),
        )
        model, calibration_method = _fit_calibrated_model(
            trainer=trainer,
            params=best_params,
            x_train=x_train,
            y_train=y_train,
            sample_weights=sample_weights,
        )
        training_params = {
            **best_params,
            "probability_calibration": calibration_method,
        }

        preds = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        # --- Feature importance (top 15) ---
        _log_feature_importance(model, trainer)

        model_uri = log_and_register(
            model=model,
            x_train=x_train,
            y_train=y_train,
            params=training_params,
            metrics=metrics,
            model_name=model_name,
            force=force,
        )

        result = TrainResult(
            model_name=trainer.name(),
            model_uri=model_uri,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"],
            params=training_params,
            run_id=run.info.run_id,
        )
        logger.info(
            "training_complete",
            run_id=run.info.run_id,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"],
        )

    return result


if __name__ == "__main__":
    import argparse

    available = list_trainers()
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument(
        "--model",
        "-m",
        choices=available,
        default="extra_trees",
        help=f"Trainer to use (default: extra_trees). Available: {available}",
    )
    parser.add_argument("--force", action="store_true", help="Register model even if worse")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment name (isolates MLflow experiment + model registry)",
    )
    parser.add_argument("--no-tuning", action="store_true", help="Skip Optuna tuning, use defaults")
    parser.add_argument("--trials", type=int, default=60, help="Number of Optuna trials")
    args = parser.parse_args()

    train_model(
        trainer=get_trainer(args.model),
        do_tuning=not args.no_tuning,
        n_trials=args.trials,
        force=args.force,
        experiment=args.experiment,
    )
