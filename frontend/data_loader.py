from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from backend.core.config import settings
from backend.infra.database import get_data_store


@st.cache_data(ttl=3600)
def load_price_data(months: int = 6) -> pd.DataFrame:
    """Download recent OHLCV data for the configured stock symbol."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = pd.Timestamp.now() - pd.DateOffset(months=months)
    df = yf.download(settings.stock.symbol, start=start.strftime("%Y-%m-%d"), end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


@st.cache_data(ttl=600)
def load_predictions() -> pd.DataFrame:
    """Load stored predictions from the data store."""
    store = get_data_store()
    try:
        df = store.load_predictions(days=10000)
    except FileNotFoundError:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    """Load engineered features (for feature importance display)."""
    store = get_data_store()
    try:
        return store.load_features()
    except FileNotFoundError:
        return pd.DataFrame()
