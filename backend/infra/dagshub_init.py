import os

import dagshub
import mlflow
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

_dagshub_initialized = False


def init_dagshub() -> None:
    """Initialize DagsHub + MLflow connection. Safe to call multiple times."""
    global _dagshub_initialized
    if _dagshub_initialized:
        return

    cfg = settings.dagshub
    token = cfg.token.get_secret_value()

    if not token:
        logger.warning("dagshub_skipped", reason="No DAGSHUB_USER_TOKEN set")
        return

    os.environ["DAGSHUB_USER_TOKEN"] = token

    dagshub.init(
        repo_owner=cfg.repo_owner,
        repo_name=cfg.repo_name,
        mlflow=True,
    )

    # Override tracking URI if explicitly set, otherwise DagsHub sets it
    if settings.mlflow.tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    mlflow.set_experiment(settings.mlflow.experiment_name)

    _dagshub_initialized = True
    logger.info(
        "dagshub_initialized",
        repo=cfg.repo_full,
        experiment=settings.mlflow.experiment_name,
        tracking_uri=mlflow.get_tracking_uri(),
    )
