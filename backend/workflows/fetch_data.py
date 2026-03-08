from datetime import datetime

import structlog
import yfinance as yf

from backend.core.config import settings
from backend.core.features.indicators import calc_indicators
from backend.core.features.manifest import FeatureManifest
from backend.core.features.preprocessing import (
    calc_target,
    fetch_extra_market_data,
    transform_to_returns,
)
from backend.infra.dagshub_init import init_dagshub
from backend.infra.database import get_data_store

logger = structlog.get_logger(__name__)


def fetch_and_store(
    symbol: str | None = None,
    start_date: str | None = None,
) -> tuple:
    init_dagshub()

    symbol = symbol or settings.stock.symbol
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")
    store = get_data_store()

    logger.info("downloading raw data", symbol=symbol, start=start_date, end=end)
    raw_df = yf.download(symbol, start=start_date, end=end)
    if hasattr(raw_df.columns, "nlevels") and raw_df.columns.nlevels > 1:
        raw_df.columns = raw_df.columns.droplevel(1)
    logger.info("raw data downloaded", rows=len(raw_df))

    store.save_raw(raw_df)

    logger.info("computing target")
    df = calc_target(raw_df.copy())

    logger.info("computing indicators")
    df = calc_indicators(df)

    logger.info("fetching extra market data")
    df = fetch_extra_market_data(df, start_date=start_date)

    df["Volume"] = df["Volume"].astype(float)
    df = df.ffill()

    logger.info("transforming to returns")
    df = transform_to_returns(df)

    manifest = FeatureManifest.from_dataframe(df)
    logger.info(
        "features ready",
        rows=manifest.row_count,
        columns=len(manifest.columns),
        hash=manifest.column_hash,
    )

    store.save_features(df)

    return df, manifest


if __name__ == "__main__":
    fetch_and_store()
