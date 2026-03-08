from pathlib import Path

import duckdb
import pandas as pd
import structlog

from backend.infra.database.base import DataStore

logger = structlog.get_logger(__name__)


class DuckDBStore(DataStore):
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "raw" / "sp500_ohlcv.parquet"
        self.features_path = self.data_dir / "features" / "sp500_features.parquet"
        self.predictions_path = self.data_dir / "predictions" / "sp500_predictions.parquet"

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)
        logger.info("saved", path=str(path), rows=len(df))

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        df = duckdb.query(f"SELECT * FROM '{path}'").to_df()
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        return df

    def save_raw(self, df: pd.DataFrame) -> None:
        self._write_parquet(df, self.raw_path)

    def save_features(self, df: pd.DataFrame) -> None:
        self._write_parquet(df, self.features_path)

    def save_predictions(self, df: pd.DataFrame) -> None:
        self._write_parquet(df, self.predictions_path)

    def load_raw(self) -> pd.DataFrame:
        return self._read_parquet(self.raw_path)

    def load_features(self) -> pd.DataFrame:
        return self._read_parquet(self.features_path)

    def load_predictions(self, days: int = 30) -> pd.DataFrame:
        df = self._read_parquet(self.predictions_path)
        return df.tail(days)
