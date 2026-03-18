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


@st.cache_data(ttl=3600)
def load_model_info() -> dict:
    """Load model metadata from MLflow, falling back to config."""
    info: dict = {
        "model_name": "ExtraTreesClassifier",
        "model_short": "ExtraTrees",
        "accuracy": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "roc_auc": None,
        "n_features": None,
        "trained_at": None,
        "horizon": settings.stock.prediction_horizon_days,
        "symbol": settings.stock.symbol_display,
        "trained": False,
    }
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_name = settings.mlflow.model_name
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            run = client.get_run(latest.run_id)
            m = run.data.metrics
            p = run.data.params
            ts = run.info.start_time
            info["accuracy"] = m.get("accuracy")
            info["f1"] = m.get("f1")
            info["precision"] = m.get("precision")
            info["recall"] = m.get("recall")
            info["roc_auc"] = m.get("roc_auc")
            info["n_features"] = p.get("n_features")
            if p.get("model_name"):
                info["model_name"] = p["model_name"]
                short = p["model_name"]
                for suffix in ("Classifier", "Regressor"):
                    short = short.replace(suffix, "")
                info["model_short"] = short
            if ts:
                info["trained_at"] = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            info["trained"] = True
    except Exception:
        pass
    return info
