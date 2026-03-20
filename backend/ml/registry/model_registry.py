import mlflow
import structlog
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from backend.core.config import settings

logger = structlog.get_logger(__name__)


def _get_production_info(client: MlflowClient, model_name: str) -> tuple[float | None, int | None]:
    """Return (roc_auc, n_features) of the latest registered model version."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        return None, None

    if not versions:
        return None, None

    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    roc_auc = run.data.metrics.get("roc_auc")
    n_features = run.data.params.get("n_features")
    n_features = int(n_features) if n_features is not None else None
    logger.info(
        "existing_model_found",
        version=latest.version,
        roc_auc=roc_auc,
        n_features=n_features,
        run_id=latest.run_id,
    )
    return roc_auc, n_features


def _archive_old_versions(client: MlflowClient, model_name: str, new_run_id: str) -> None:
    """Tag old model versions with archive metadata so their history is preserved."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        return

    for v in versions:
        if v.run_id == new_run_id:
            continue
        old_run = client.get_run(v.run_id)
        old_metrics = old_run.data.metrics
        old_params = old_run.data.params

        # Store old model metadata as tags on the version before it gets superseded
        client.set_model_version_tag(model_name, v.version, "archived", "true")
        client.set_model_version_tag(
            model_name, v.version, "archived_accuracy", str(old_metrics.get("accuracy", ""))
        )
        client.set_model_version_tag(
            model_name, v.version, "archived_f1", str(old_metrics.get("f1", ""))
        )
        client.set_model_version_tag(model_name, v.version, "archived_params", str(old_params))

        # Also log as tags on the NEW run so you can see history in one place
        mlflow.set_tag(f"prev_v{v.version}_accuracy", old_metrics.get("accuracy", ""))
        mlflow.set_tag(f"prev_v{v.version}_f1", old_metrics.get("f1", ""))
        mlflow.set_tag(f"prev_v{v.version}_run_id", v.run_id)

        # Delete old version artifact to free DagsHub storage
        client.delete_model_version(model_name, v.version)
        logger.info(
            "old_version_archived_and_deleted",
            version=v.version,
            old_accuracy=old_metrics.get("accuracy"),
        )


def log_and_register(
    model,
    x_train,
    y_train,
    params: dict,
    metrics: dict,
    model_name: str | None = None,
    force: bool = False,
) -> str:
    """Log model to MLflow. Only register if it beats the current best.

    Set *force=True* to register regardless of performance (useful when
    experimenting with new settings/data that aren't directly comparable).

    Old model versions are deleted to save storage, but their metrics/params
    are preserved as tags on the archived version and on the new run.
    Returns the model URI.
    """
    model_name = model_name or settings.mlflow.model_name
    client = MlflowClient()

    current_roc_auc, current_n_features = _get_production_info(client, model_name)
    new_roc_auc = metrics.get("roc_auc", 0)
    new_n_features = x_train.shape[1]

    mlflow.log_params({**params, "n_features": new_n_features})
    mlflow.log_metrics(metrics)

    signature = infer_signature(x_train, model.predict(x_train))

    # If the old model has no n_features param (registered before this logging was added),
    # we cannot verify compatibility → treat it as changed to force promotion.
    feature_count_changed = (
        current_n_features is None and current_roc_auc is not None  # old model, no n_features
    ) or (
        current_n_features is not None and current_n_features != new_n_features  # known mismatch
    )
    if feature_count_changed:
        logger.info(
            "feature_set_changed",
            old_n_features=current_n_features,
            new_n_features=new_n_features,
            reason="forced promotion — old model incompatible (n_features unknown or changed)",
        )

    metric_worse = current_roc_auc is not None and new_roc_auc <= current_roc_auc
    if not feature_count_changed and metric_worse and not force:
        # Log model as artifact but do NOT register it
        info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )
        logger.info(
            "model_not_promoted",
            new_roc_auc=new_roc_auc,
            current_roc_auc=current_roc_auc,
            reason="not better than current",
        )
        return info.model_uri

    if force and metric_worse:
        logger.warning(
            "force_promoting_worse_model",
            new_roc_auc=new_roc_auc,
            current_roc_auc=current_roc_auc,
        )

    # New model is better — register and clean up old versions
    info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name,
    )

    run_id = mlflow.active_run().info.run_id
    _archive_old_versions(client, model_name, run_id)

    if current_roc_auc is not None:
        mlflow.set_tag("replaced_roc_auc", current_roc_auc)
        logger.info(
            "model_promoted",
            model_name=model_name,
            new_roc_auc=new_roc_auc,
            previous_roc_auc=current_roc_auc,
            improvement=f"{new_roc_auc - current_roc_auc:+.4f}",
        )
    else:
        logger.info(
            "model_registered_first",
            model_name=model_name,
            roc_auc=new_roc_auc,
        )

    return info.model_uri
