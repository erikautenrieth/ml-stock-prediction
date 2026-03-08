from backend.core.config import settings
from backend.infra.database.duckdb_store import DuckDBStore


def get_data_store() -> DuckDBStore:
    return DuckDBStore(settings.database.data_dir)
