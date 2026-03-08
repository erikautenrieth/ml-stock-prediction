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
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", "none"]),
        }
        if params["class_weight"] == "none":
            params["class_weight"] = None

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

    logger.info(
        "tuning_complete",
        best_accuracy=study.best_value,
        best_params=study.best_params,
        n_trials=len(study.trials),
    )

    return study.best_params
