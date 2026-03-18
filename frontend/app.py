import os

import pandas as pd
import streamlit as st

# Bridge Streamlit Cloud secrets → env vars (before config import)
for key in ("DAGSHUB_REPO_OWNER", "DAGSHUB_REPO_NAME", "DAGSHUB_TOKEN"):
    if key not in os.environ and hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = st.secrets[key]

from backend.core.config import settings
from frontend.charts import (
    candlestick_chart,
    compute_prediction_outcomes,
    confidence_chart,
    prediction_line_chart,
    prediction_performance_chart,
)
from frontend.data_loader import load_predictions, load_price_data
from frontend.model_card import render_model_card

UP_COLOR = "#00E676"
DOWN_COLOR = "#FF5252"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _render_header(
    horizon: int,
    visible_preds: pd.DataFrame,
) -> None:
    """Compact hero header: symbol left, latest prediction right."""
    symbol = settings.stock.symbol_display

    if not visible_preds.empty:
        latest = visible_preds.iloc[-1]
        is_up = int(latest["prediction"]) == 1
        conf = float(latest["confidence"])
        color = UP_COLOR if is_up else DOWN_COLOR
        arrow = "▲" if is_up else "▼"
        direction = "UP" if is_up else "DOWN"
        date_str = pd.Timestamp(visible_preds.index[-1]).strftime("%b %d, %Y")

        right_html = (
            f'<span style="font-size:1.6em;font-weight:700;color:{color}">'
            f"{arrow} {direction}</span>"
            f'<span style="font-size:1.1em;color:#aaa">'
            f"{conf:.0%} conf · {date_str}</span>"
        )
    else:
        right_html = ""

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;
                    justify-content:space-between;
                    padding:.6em 0;margin-bottom:.2em">
          <div style="display:flex;align-items:center;gap:.5em">
            <span style="font-size:2.2em;font-weight:700">{symbol}</span>
            <span style="font-size:1.1em;color:#888;font-weight:400">
              {horizon}d prediction
            </span>
          </div>
          <div style="display:flex;align-items:center;gap:1.2em">
            {right_html}
          </div>
        </div>
        """,
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

    # Charts
    if not visible_preds.empty:
        st.plotly_chart(
            prediction_line_chart(ohlcv, visible_preds),
            width="stretch",
        )
    st.plotly_chart(
        prediction_performance_chart(outcomes),
        width="stretch",
    )

    # P&L table
    st.subheader("Prediction Results")
    tbl = _build_results_table(outcomes)
    st.dataframe(tbl, width="stretch")

    # Summary metrics
    closed = outcomes[outcomes["status"] == "closed"]
    if not closed.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total P&L", f"${closed['pnl'].sum():+,.2f}")
        c2.metric("Accuracy", f"{closed['correct'].mean():.0%}")
        c3.metric("Avg Return", f"{closed['return_pct'].mean():+.2%}")


def _build_results_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Format outcomes into a display-ready DataFrame."""
    tbl = outcomes.copy()
    tbl["Date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d")

    direction_map = {"UP": "🟢 UP", "DOWN": "🔴 DOWN"}
    tbl["Direction"] = tbl["direction"].map(direction_map)
    tbl["Confidence"] = tbl["confidence"].map("{:.1%}".format)
    tbl["Entry $"] = tbl["entry_price"].map("${:,.2f}".format)
    tbl["Exit $"] = tbl["exit_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "⏳ pending")
    tbl["Return %"] = tbl["return_pct"].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    tbl["P&L $"] = tbl["pnl"].apply(lambda x: f"{x:+,.2f}" if pd.notna(x) else "—")
    tbl["Result"] = tbl.apply(
        lambda r: "⏳" if r["status"] == "open" else ("✅" if r["correct"] else "❌"),
        axis=1,
    )

    cols = [
        "Date",
        "Direction",
        "Confidence",
        "Entry $",
        "Exit $",
        "Return %",
        "P&L $",
        "Result",
    ]
    return tbl[cols].sort_values("Date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{settings.stock.symbol_display} Trend Prediction",
        page_icon="📈",
        layout="wide",
    )

    horizon = settings.stock.prediction_horizon_days

    with st.sidebar:
        st.header("Settings")
        months = st.slider(
            "Chart history (months)",
            min_value=1,
            max_value=24,
            value=6,
        )

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
    tab_candle, tab_perf, tab_conf = st.tabs(["🕯️ Candlestick", "📊 Performance", "🎯 Confidence"])

    with tab_candle:
        has_preds = not visible_preds.empty
        st.plotly_chart(
            candlestick_chart(
                ohlcv,
                visible_preds if has_preds else pd.DataFrame(),
            ),
            width="stretch",
        )
        if has_preds:
            render_model_card()
        else:
            st.info("No predictions yet. Run `make predict` first.")

    with tab_perf:
        _render_performance_tab(ohlcv, visible_preds, outcomes)

    with tab_conf:
        if not visible_preds.empty:
            st.plotly_chart(
                confidence_chart(visible_preds),
                width="stretch",
            )
        else:
            st.info("No predictions available yet.")


if __name__ == "__main__":
    main()
