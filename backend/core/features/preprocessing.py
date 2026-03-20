import numpy as np
import pandas as pd
import structlog

from backend.core.config import settings
from backend.core.features.indicators import calc_indicators

logger = structlog.get_logger(__name__)


def build_features(
    raw_df: pd.DataFrame,
    extra_market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Process raw OHLCV data into ML-ready features.

    Args:
        raw_df: Raw OHLCV DataFrame.
        extra_market_df: Pre-downloaded extra market close prices
            (e.g. VIX, Treasury yields). If *None*, the step is skipped.
    """
    df = calc_target(raw_df.copy())
    df = calc_indicators(df)
    if extra_market_df is not None and not extra_market_df.empty:
        df = merge_extra_market_data(df, extra_market_df)
    df["Volume"] = df["Volume"].astype(float)
    df = df.ffill()
    df = transform_to_returns(df)
    return df


def calc_target(df: pd.DataFrame) -> pd.DataFrame:
    horizon = settings.stock.prediction_horizon_days
    threshold = settings.features.target_threshold
    ret = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    df["Target"] = np.where(ret > threshold, 1, np.where(ret < -threshold, 0, np.nan))
    n_dead = df["Target"].isna().sum()
    logger.info(
        "calc_target: threshold=%.4f, dead_zone=%d rows (%.1f%%)",
        threshold, n_dead, 100 * n_dead / len(df),
    )
    return df


def merge_extra_market_data(
    df: pd.DataFrame,
    extra_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align pre-downloaded extra market data to *df*'s index and merge."""
    extra_df = extra_df.copy()
    extra_df.interpolate(method="polynomial", order=3, inplace=True, axis=0)
    extra_df.bfill(inplace=True)

    extra_df.index = pd.to_datetime(extra_df.index)
    df_idx = pd.to_datetime(df.index)
    extra_df = extra_df.reindex(df_idx)

    out = df.copy()
    for col in extra_df.columns:
        out[col] = extra_df[col].values

    nan_before = out.isna().sum().sum()
    out.interpolate(method="polynomial", order=3, inplace=True, axis=0)
    out.bfill(inplace=True)
    nan_after = out.isna().sum().sum()
    if nan_before > 0:
        logger.info(
            "merge_extra_market_data: filled %d NaN values, %d remaining",
            nan_before - nan_after,
            nan_after,
        )

    return out


def transform_to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute prices to returns/relative values for stationarity."""
    df = df.copy()
    close = df["Close"].copy()

    # Moving averages & SAR → relative distance to close price (%)
    relative_cols = [
        c
        for c in df.columns
        if any(k in c for k in ["EMA", "SAR"])
    ]
    for col in relative_cols:
        df[col] = (df[col] - close) / close

    # Momentum/MACD → normalize by close (scale-independent)
    # Note: MACD_SIGNAL and MACD_HIST already start with "MACD", so they are captured
    # by the "MACD" prefix check — no need to list them separately.
    normalize_cols = [c for c in df.columns if any(c.startswith(k) for k in ["Momentum", "MACD"])]
    for col in normalize_cols:
        df[col] = df[col] / close

    # Interest rates → diff instead of pct_change (near-zero values cause extreme %)
    rate_tickers = ["^TNX", "^IRX"]
    rate_cols = [f"{t} Close" for t in rate_tickers if f"{t} Close" in df.columns]
    for col in rate_cols:
        df[col] = df[col].diff()

    # All other prices and volume → daily returns via pct_change
    pct_cols = ["Close", "Volume"]
    extra_close_cols = [c for c in df.columns if c.endswith(" Close") and c not in rate_cols]
    pct_cols.extend(extra_close_cols)
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].pct_change()

    # OBV is cumulative and can be negative → pct_change across zero is meaningless.
    # Use daily diff normalized by close price instead (Sehgal & Garhyan 2002; OBV literature).
    # This gives a scale-independent "volume flow rate" feature.
    if "OBV" in df.columns:
        df["OBV"] = df["OBV"].diff() / close

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
