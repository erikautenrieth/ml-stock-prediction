import structlog

from backend.core.config import settings
from backend.infra.database.duckdb_store import DuckDBStore

logger = structlog.get_logger(__name__)


def get_data_store() -> DuckDBStore:
    remote = None
    if settings.dagshub.token.get_secret_value():
        from backend.infra.storage import get_remote_storage

        remote = get_remote_storage()
        logger.info("data_store_remote_enabled")

    return DuckDBStore(settings.database.data_dir, remote=remote)
