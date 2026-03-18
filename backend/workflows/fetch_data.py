from datetime import datetime

import pandas as pd
import structlog
import yfinance as yf

from backend.core.config import settings
from backend.core.features.manifest import FeatureManifest
from backend.core.features.preprocessing import build_features
from backend.infra.database import get_data_store

logger = structlog.get_logger(__name__)


def download_extra_market_data(
    start_date: str | None = None,
) -> pd.DataFrame:
    """Download extra market tickers (VIX, Treasury yields, etc.).

    Returns a DataFrame with columns like ``'^VIX Close'``.
    """
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")
    tickers = settings.stock.extra_tickers

    combined = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if "Close" in data.columns:
            combined[f"{ticker} Close"] = data["Close"]

    logger.info("extra_market_data_downloaded", tickers=tickers, rows=len(combined))
    return combined


def fetch_and_store(
    symbol: str | None = None,
    start_date: str | None = None,
) -> tuple:
    symbol = symbol or settings.stock.symbol
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")
    store = get_data_store()

    logger.info("downloading_raw_data", symbol=symbol, start=start_date, end=end)
    raw_df = yf.download(symbol, start=start_date, end=end)
    if hasattr(raw_df.columns, "nlevels") and raw_df.columns.nlevels > 1:
        raw_df.columns = raw_df.columns.droplevel(1)
    store.save_raw(raw_df)

    extra_df = download_extra_market_data(start_date=start_date)

    logger.info("building_features")
    df = build_features(raw_df, extra_market_df=extra_df)

    manifest = FeatureManifest.from_dataframe(df)
    logger.info(
        "features_ready",
        rows=manifest.row_count,
        columns=len(manifest.columns),
        hash=manifest.column_hash,
    )

    store.save_features(df)

    return df, manifest


def get_data(
    symbol: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download & process data in one call. Used by exploration notebooks."""
    symbol = symbol or settings.stock.symbol
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")

    raw = yf.download(symbol, start=start_date, end=end)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    extra_df = download_extra_market_data(start_date=start_date)
    df = build_features(raw, extra_market_df=extra_df)

    last_day_df = df.drop("Target", axis=1).tail(1)
    training_df = df.iloc[:-10]

    return training_df, last_day_df


if __name__ == "__main__":
    fetch_and_store()
