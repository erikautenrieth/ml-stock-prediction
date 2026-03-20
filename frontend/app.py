import os

import pandas as pd
import streamlit as st

# Bridge Streamlit Cloud secrets → env vars (before config import)
try:
    for key in ("DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME", "DAGSHUB_TOKEN"):
        if key not in os.environ and key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # No secrets.toml — running locally with env vars

from backend.core.config import settings
from frontend.charts import (
    candlestick_chart,
    compute_prediction_outcomes,
    confidence_chart,
    prediction_line_chart,
    prediction_performance_chart,
    price_line_chart,
)
from frontend.data_loader import load_predictions, load_price_data
from frontend.model_card import render_model_card


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _confidence_label(conf: float) -> tuple[str, str]:
    """Return (label, icon) for a confidence value."""
    if conf >= 0.70:
        return "Strong", "🟢"
    if conf >= 0.60:
        return "Moderate", "🟡"
    return "Weak", "🔴"


def _render_header(horizon: int, visible_preds: pd.DataFrame) -> None:
    """Compact header using native Streamlit columns + metrics."""
    symbol = settings.stock.symbol_display
    left, right = st.columns([3, 2])

    left.title(f"💹 {symbol}")
    left.caption(f"{horizon}-day prediction horizon")

    if not visible_preds.empty:
        latest = visible_preds.iloc[-1]
        is_up = int(latest["prediction"]) == 1
        conf = float(latest["confidence"])
        direction = "▲ UP" if is_up else "▼ DOWN"
        dir_color = "green" if is_up else "red"
        strength, icon = _confidence_label(conf)
        date_str = pd.Timestamp(visible_preds.index[-1]).strftime("%b %d, %Y")

        right.markdown(
            f"**Direction:** :{dir_color}[{direction}] &nbsp; · &nbsp; "
            f"**Confidence:** {conf:.0%} {icon} *{strength}*  \n"
            f"<sub>{date_str}</sub>",
            unsafe_allow_html=True,
        )

    st.divider()


def _render_performance_tab(
    ohlcv: pd.DataFrame,
    visible_preds: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    """Performance tab: charts + P&L table + summary metrics."""
    if outcomes.empty:
        st.info("No predictions available yet.")
        return

    # Summary metrics first
    closed = outcomes[outcomes["status"] == "closed"]
    if not closed.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total P&L", f"${closed['pnl'].sum():+,.2f}")
        c2.metric("Accuracy", f"{closed['correct'].mean():.0%}")
        c3.metric("Avg Return", f"{closed['return_pct'].mean():+.2%}")

    # Charts
    if not visible_preds.empty:
        st.plotly_chart(prediction_line_chart(ohlcv, visible_preds), width="stretch")
    st.plotly_chart(prediction_performance_chart(outcomes), width="stretch")

    # P&L table
    st.subheader("Prediction Results")
    st.dataframe(_build_results_table(outcomes), width="stretch")


def _build_results_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Format outcomes into a display-ready DataFrame."""
    tbl = outcomes.copy()
    tbl["Date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d")

    tbl["Direction"] = tbl["direction"].apply(lambda d: f"{'🟢' if d == 'UP' else '🔴'} {d}")
    tbl["Confidence"] = tbl["confidence"].apply(lambda c: f"{c:.0%} {_confidence_label(c)[1]}")
    tbl["Entry $"] = tbl["entry_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    tbl["Exit $"] = tbl["exit_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "⏳")
    tbl["Return %"] = tbl["return_pct"].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    tbl["P&L $"] = tbl["pnl"].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "—")
    tbl["Result"] = tbl.apply(
        lambda r: "⏳" if r["status"] == "open" else ("✅" if r["correct"] else "❌"),
        axis=1,
    )

    cols = ["Date", "Direction", "Confidence", "Entry $", "Exit $", "Return %", "P&L $", "Result"]
    return tbl[cols].sort_values("Date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{settings.stock.symbol_display} Trend Prediction",
        page_icon="💹",
        layout="wide",
    )

    horizon = settings.stock.prediction_horizon_days

    with st.sidebar:
        st.header("⚙️ Settings")
        months = st.slider("Chart history (months)", min_value=1, max_value=24, value=6)

    # --- Data ---
    ohlcv = load_price_data(months=months)
    predictions = load_predictions()

    if ohlcv.empty:
        st.error("Could not load price data.")
        return

    visible_preds = predictions[predictions.index >= ohlcv.index.min()]

    _render_header(horizon, visible_preds)

    outcomes = (
        compute_prediction_outcomes(ohlcv, visible_preds)
        if not visible_preds.empty
        else pd.DataFrame()
    )

    # --- Tabs ---
    tab_chart, tab_perf, tab_conf = st.tabs(["📈 Chart", "📊 Performance", "🎯 Confidence"])

    with tab_chart:
        chart_type = st.radio(
            "Chart type",
            ["Candlestick", "Line"],
            horizontal=True,
            label_visibility="collapsed",
        )
        has_preds = not visible_preds.empty
        preds = visible_preds if has_preds else pd.DataFrame()
        if chart_type == "Candlestick":
            st.plotly_chart(candlestick_chart(ohlcv, preds), width="stretch")
        else:
            st.plotly_chart(price_line_chart(ohlcv, preds), width="stretch")
        if has_preds:
            render_model_card()
        else:
            st.info("No predictions yet. Run `make predict` first.")

    with tab_perf:
        _render_performance_tab(ohlcv, visible_preds, outcomes)

    with tab_conf:
        if not visible_preds.empty:
            st.plotly_chart(confidence_chart(visible_preds), width="stretch")
        else:
            st.info("No predictions available yet.")


if __name__ == "__main__":
    main()
