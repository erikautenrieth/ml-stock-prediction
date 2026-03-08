from datetime import datetime

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from backend.core.config import settings
from backend.core.features.indicators import calc_indicators

logger = structlog.get_logger(__name__)


def build_features(raw_df: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Process raw OHLCV data into ML-ready features."""
    df = calc_target(raw_df.copy())
    df = calc_indicators(df)
    df = fetch_extra_market_data(df, start_date=start_date)
    df["Volume"] = df["Volume"].astype(float)
    df = df.ffill()
    df = transform_to_returns(df)
    return df


def get_data(
    symbol: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download & process data in one call. Used by exploration notebooks."""
    symbol = symbol or settings.stock.symbol
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = build_features(df, start_date=start_date)

    last_day_df = df.drop("Target", axis=1).tail(1)
    training_df = df.iloc[:-10]

    return training_df, last_day_df


def calc_target(df: pd.DataFrame) -> pd.DataFrame:
    horizon = settings.stock.prediction_horizon_days
    df["Target"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)
    return df


def fetch_extra_market_data(
    df: pd.DataFrame,
    start_date: str | None = None,
) -> pd.DataFrame:
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

    combined.interpolate(method="polynomial", order=3, inplace=True, axis=0)
    combined.bfill(inplace=True)

    combined.index = pd.to_datetime(combined.index)
    df_idx = pd.to_datetime(df.index)
    combined = combined.reindex(df_idx)

    out = df.copy()
    for col in combined.columns:
        out[col] = combined[col].values

    nan_before = out.isna().sum().sum()
    out.interpolate(method="polynomial", order=3, inplace=True, axis=0)
    out.bfill(inplace=True)
    nan_after = out.isna().sum().sum()
    if nan_before > 0:
        logger.info(
            "fetch_extra_market_data: filled %d NaN values, %d remaining",
            nan_before - nan_after,
            nan_after,
        )

    return out


def transform_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute prices to returns/relative values for stationarity."""
    df = df.copy()
    close = df["Close"].copy()

    # Moving averages & bands → relative distance to close price (%)
    relative_cols = [
        c
        for c in df.columns
        if any(k in c for k in ["SMA", "EMA", "WMA", "SAR", "up_band", "mid_band", "low_band"])
    ]
    for col in relative_cols:
        df[col] = (df[col] - close) / close

    # Momentum/MACD → normalize by close (scale-independent)
    normalize_cols = [
        c
        for c in df.columns
        if any(c.startswith(k) for k in ["Momentum", "MACD", "MACD_SIGNAL", "MACD_HIST"])
    ]
    for col in normalize_cols:
        df[col] = df[col] / close

    # Interest rates → diff instead of pct_change (near-zero values cause extreme %)
    rate_tickers = ["^TNX", "^IRX", "^FVX", "^TYX"]
    rate_cols = [f"{t} Close" for t in rate_tickers if f"{t} Close" in df.columns]
    for col in rate_cols:
        df[col] = df[col].diff()

    # All other prices, volume, OBV → daily returns via pct_change
    pct_cols = ["Close", "Volume", "OBV"]
    extra_close_cols = [c for c in df.columns if c.endswith(" Close") and c not in rate_cols]
    pct_cols.extend(extra_close_cols)
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].pct_change()

    # Volume can have 0 values → pct_change produces inf
    df.replace([np.inf, -np.inf], 0, inplace=True)

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info(
            "transform_to_returns: %d rows dropped (NaN from pct_change), %d remaining",
            rows_dropped,
            len(df),
        )

    return df
