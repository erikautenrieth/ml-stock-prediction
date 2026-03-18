import mlflow
import structlog

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from backend.core.config import settings
from backend.core.schemas import TrainResult
from backend.infra.dagshub_init import init_dagshub
from backend.infra.database import get_data_store
from backend.ml.registry.model_registry import log_and_register
from backend.ml.training.base import Trainer
from backend.ml.training.extra_tree import ExtraTreesTrainer
from backend.ml.training.tuning.optuna_tuner import tune

logger = structlog.get_logger(__name__)


def train_model(
    trainer: Trainer | None = None,
    do_tuning: bool = True,
    n_trials: int = 60,
) -> TrainResult:
    """Full training pipeline: load features → tune → train → log to DagsHub."""
    init_dagshub()

    if trainer is None:
        trainer = ExtraTreesTrainer()

    store = get_data_store()
    df = store.load_features()
    logger.info("features_loaded", rows=len(df), columns=len(df.columns))
    horizon = settings.stock.prediction_horizon_days
    x_train, x_test, y_train, y_test = trainer.prepare(df, embargo=horizon)

    with mlflow.start_run(run_name=f"train_{trainer.name()}") as run:
        if do_tuning:
            logger.info("starting_tuning", n_trials=n_trials)
            best_params = tune(trainer, x_train, y_train, x_test, y_test, n_trials=n_trials)
        else:
            best_params = trainer.default_params()

        logger.info("training_final_model", params=best_params)
        model = trainer.build(best_params)
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        logger.info("evaluation", **metrics)

        model_uri = log_and_register(
            model=model,
            x_train=x_train,
            y_train=y_train,
            params=best_params,
            metrics=metrics,
        )

        result = TrainResult(
            model_name=trainer.name(),
            model_uri=model_uri,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            roc_auc=metrics["roc_auc"],
            params=best_params,
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
    train_model()
