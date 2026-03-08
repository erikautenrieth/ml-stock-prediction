import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from datetime import datetime

from backend.core.config import settings
from backend.core.features.indicators import calc_indicators

logger = structlog.get_logger(__name__)


def get_data(
    symbol: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol = symbol or settings.stock.symbol
    start_date = start_date or settings.stock.start_date
    end = datetime.now().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = calc_target(df)
    df = calc_indicators(df)
    df = fetch_extra_market_data(df, start_date=start_date)

    df["Volume"] = df["Volume"].astype(float)

    nan_before = df.isna().sum().sum()
    df = df.ffill()
    nan_after = df.isna().sum().sum()
    if nan_before > 0:
        logger.info(
            "get_data: ffill filled %d NaN values, %d remaining", nan_before - nan_after, nan_after
        )

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


def scale_data(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x, scaler
