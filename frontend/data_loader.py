import logging
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from backend.core.config import settings
from backend.infra.database import get_data_store

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def load_price_data(months: int = 6) -> pd.DataFrame:
    """Download live OHLCV from yfinance, falling back to store on error."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)

    # Primary: live data from Yahoo Finance
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(
            settings.stock.symbol,
            start=cutoff.strftime("%Y-%m-%d"),
            end=end,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty:
            logger.info("load_price_data: fetched %d rows from yfinance", len(df))
            return df
    except Exception as exc:
        logger.warning("load_price_data: yfinance failed (%s), trying store", exc)

    # Fallback: stored data (may be stale)
    try:
        store = get_data_store()
        df = store.load_raw()
        df.index = pd.to_datetime(df.index)
        df = df[df.index >= cutoff]
        if not df.empty:
            logger.info("load_price_data: fallback to store, %d rows", len(df))
            return df
    except Exception as exc:
        logger.warning("load_price_data: store also failed (%s)", exc)

    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_predictions() -> pd.DataFrame:
    """Load stored predictions from the data store."""
    store = get_data_store()
    try:
        df = store.load_predictions(days=10000)
        logger.info("load_predictions: loaded %d rows", len(df))
    except FileNotFoundError:
        logger.warning("load_predictions: no predictions file found")
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
    """Load model metadata from MLflow, falling back to config defaults."""
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

        from backend.infra.dagshub_init import init_dagshub

        init_dagshub()
        client = MlflowClient()
        model_name = settings.mlflow.model_name
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return info

        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        _update_info_from_run(info, run)
    except Exception as exc:
        logger.warning("load_model_info failed: %s", exc)
    return info


def _update_info_from_run(info: dict, run: object) -> None:
    """Fill *info* dict from an MLflow Run object."""
    metrics = run.data.metrics
    params = run.data.params

    for key in ("accuracy", "f1", "precision", "recall", "roc_auc"):
        info[key] = metrics.get(key)

    info["n_features"] = params.get("n_features")

    name = params.get("model_name")
    if name:
        info["model_name"] = name
        for suffix in ("Classifier", "Regressor"):
            name = name.replace(suffix, "")
        info["model_short"] = name

    ts = run.info.start_time
    if ts:
        info["trained_at"] = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")

    info["trained"] = True
