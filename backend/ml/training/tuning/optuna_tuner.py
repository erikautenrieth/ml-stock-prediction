import mlflow
import optuna
import structlog
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import accuracy_score

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
    """Run Optuna hyperparameter search, logging every trial to MLflow via DagsHub."""

    def objective(trial: optuna.Trial) -> float:
        params = trainer.search_space(trial)
        trial.set_user_attr("processed_params", params)
        model = trainer.build(params)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        return accuracy_score(y_test, preds)

    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="accuracy",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    best_params = study.best_trial.user_attrs["processed_params"]

    logger.info(
        "tuning_complete",
        best_accuracy=study.best_value,
        best_params=best_params,
        n_trials=len(study.trials),
    )

    return best_params
