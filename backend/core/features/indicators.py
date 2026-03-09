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
    # Momentum removed: after normalization by close it is nearly identical to ROC.
    # ROC = (close - close.shift(n)) / close.shift(n) * 100 already captures this signal cleanly.
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

    # Correct stochastic oscillator: compares close to the 14-period high/low range
    # (George Lane, 1950s; confirmed by Vaiz & Ramaswami 2016, Investopedia, Patel et al. 2015)
    # Previous code only used intraday high/low (single day) — wrong formula.
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["+DMI"] = adx.adx_pos()
    df["-DMI"] = adx.adx_neg()
    df["ADX"] = adx.adx()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["up_band"] = bb.bollinger_hband()
    df["mid_band"] = bb.bollinger_mavg()
    df["low_band"] = bb.bollinger_lband()
    # BB %B: position within the bands (0 = lower band, 1 = upper band).
    # Already bounded ~ [0, 1], no further transformation required.
    df["BB_pband"] = bb.bollinger_pband()
    # BB width: (upper - lower) / middle — normalized volatility measure.
    # Already scale-independent, no further transformation required.
    df["BB_width"] = bb.bollinger_wband()

    # ATR normalized by close: scale-independent volatility measure.
    # Plain ATR is in price units; dividing by close makes it stationary.
    df["ATR_norm"] = ta.volatility.average_true_range(high, low, close, window=14) / close

    # Lagged daily returns (t-1 to t-20): capture short-term momentum/mean-reversion.
    # Pre-computed as returns so transform_to_returns leaves them untouched.
    # Literature: Patel et al. (2015); Malla et al. (2026) validated lags up to 30 days.
    daily_ret = close.pct_change()
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag{lag}"] = daily_ret.shift(lag)

    # Rolling realized volatility (multi-scale): std of daily returns over 5/10/20/60 days.
    # One of the most consistently validated features in quantitative ML literature:
    # Moews et al. (2019 Expert Systems), Malla et al. (2026), Pabuccu & Barbu (2024).
    # Captures volatility regime (calm vs stressed market) which modulates signal reliability.
    for w in [5, 10, 20, 60]:
        df[f"vol_roll{w}"] = daily_ret.rolling(w).std()

    # Overnight gap return: (open_t - close_{t-1}) / close_{t-1}.
    # Captures information arrival during closed-market hours.
    # Bisdoulis (2025): open-to-EMA ratios among highest feature importance for GBDTs.
    df["gap_ret"] = (df["Open"] - close.shift(1)) / close.shift(1)

    # Intraday range normalized by close: (high - low) / close.
    # Scale-independent within-day volatility proxy; already stationary.
    # Moews et al. (2019): intraday range consistently useful for direction prediction.
    df["intraday_range"] = (high - low) / close

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info("calc_indicators: %d rows dropped (NaN), %d remaining", rows_dropped, len(df))

    df.drop(["High", "Low", "Open"], axis=1, inplace=True, errors="ignore")

    return df
