import warnings

import pandas as pd
import structlog
import ta

from backend.core.config import settings

logger = structlog.get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="ta")


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].astype(float)

    indicator_window = settings.stock.indicator_window

    df[f"SMA {indicator_window}"] = ta.trend.sma_indicator(close, window=indicator_window)
    df[f"EMA {indicator_window}"] = ta.trend.ema_indicator(close, window=indicator_window)
    df["EMA 20"] = ta.trend.ema_indicator(close, window=20)
    df[f"WMA {indicator_window}"] = ta.trend.wma_indicator(close, window=indicator_window)
    df[f"Momentum {indicator_window}"] = close - close.shift(indicator_window)
    df["SAR"] = ta.trend.PSARIndicator(high, low, close).psar()

    df["RSI"] = ta.momentum.rsi(close, window=14)
    df["ROC"] = ta.momentum.roc(close, window=10)
    df["%R"] = ta.momentum.williams_r(high, low, close, lbp=14)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    df["CCI"] = ta.trend.cci(high, low, close, window=14)
    df["ADOSC"] = ta.volume.chaikin_money_flow(high, low, close, volume, window=10)

    df["%K"] = (close - low) * 100 / (high - low)
    df["%D"] = df["%K"].rolling(3).mean()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["+DMI"] = adx.adx_pos()
    df["-DMI"] = adx.adx_neg()
    df["ADX"] = adx.adx()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["up_band"] = bb.bollinger_hband()
    df["mid_band"] = bb.bollinger_mavg()
    df["low_band"] = bb.bollinger_lband()

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info("calc_indicators: %d rows dropped (NaN), %d remaining", rows_dropped, len(df))

    df.drop(["High", "Low", "Open"], axis=1, inplace=True, errors="ignore")

    return df
