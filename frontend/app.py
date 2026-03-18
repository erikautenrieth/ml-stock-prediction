import pandas as pd
import streamlit as st

from backend.core.config import settings
from frontend.charts import (
    candlestick_chart,
    compute_prediction_outcomes,
    confidence_chart,
    prediction_line_chart,
    prediction_performance_chart,
)
from frontend.data_loader import load_predictions, load_price_data


def main() -> None:
    st.set_page_config(
        page_title=f"{settings.stock.symbol_display} Trend Prediction",
        page_icon="📈",
        layout="wide",
    )

    horizon = settings.stock.prediction_horizon_days
    st.markdown(
        f"""
        <h1 style="display:flex; align-items:center; gap:.4em; margin:0 0 .3em 0;">
            <span style="font-size:1.6em">📈</span>
            <span>{settings.stock.symbol_display}
                <span style="color:#888; font-weight:400;">—</span>
                {horizon}-Day Trend Prediction
            </span>
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        months = st.slider("Chart history (months)", min_value=1, max_value=24, value=6)

    # --- Data ---
    ohlcv = load_price_data(months=months)
    predictions = load_predictions()

    if ohlcv.empty:
        st.error("Could not load price data. Check your internet connection.")
        return

    # Filter predictions to the visible date range
    visible_preds = predictions[predictions.index >= ohlcv.index.min()]

    # --- Latest prediction highlight ---
    if not visible_preds.empty:
        latest = visible_preds.iloc[-1]
        latest_date = visible_preds.index[-1]
        direction = "UP ▲" if int(latest["prediction"]) == 1 else "DOWN ▼"
        conf = float(latest["confidence"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Prediction", direction)
        col2.metric("Confidence", f"{conf:.1%}")
        col3.metric("Date", pd.Timestamp(latest_date).strftime("%Y-%m-%d"))
        st.divider()

    # --- Compute outcomes ---
    outcomes = pd.DataFrame()
    if not visible_preds.empty:
        outcomes = compute_prediction_outcomes(ohlcv, visible_preds)

    # --- Charts ---
    tab_candle, tab_perf, tab_line, tab_confidence = st.tabs(
        ["🕯️ Candlestick", "📊 Performance", "📈 Line + Outcomes", "🎯 Confidence"]
    )

    with tab_candle:
        if not visible_preds.empty:
            st.plotly_chart(
                candlestick_chart(ohlcv, visible_preds),
                width="stretch",
            )
        else:
            st.plotly_chart(
                candlestick_chart(ohlcv, pd.DataFrame()),
                width="stretch",
            )
            st.info("No predictions available yet. Run `make predict` first.")

    with tab_perf:
        if not outcomes.empty:
            st.plotly_chart(
                prediction_performance_chart(outcomes),
                width="stretch",
            )
        else:
            st.info("No predictions available yet.")

    with tab_line:
        if not visible_preds.empty:
            st.plotly_chart(
                prediction_line_chart(ohlcv, visible_preds),
                width="stretch",
            )
        else:
            st.info("No predictions available yet.")

    with tab_confidence:
        if not visible_preds.empty:
            st.plotly_chart(
                confidence_chart(visible_preds),
                width="stretch",
            )
        else:
            st.info("No predictions available yet.")

    # --- Prediction results table ---
    if not outcomes.empty:
        st.subheader("Prediction Results")
        tbl = outcomes.copy()
        tbl["Date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d")
        tbl["Direction"] = tbl["direction"].map({"UP": "🟢 UP", "DOWN": "🔴 DOWN"})
        tbl["Confidence"] = tbl["confidence"].map(lambda x: f"{x:.1%}")
        tbl["Entry $"] = tbl["entry_price"].map(lambda x: f"${x:,.2f}")
        tbl["Exit $"] = tbl["exit_price"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "⏳ pending"
        )
        tbl["Return %"] = tbl["return_pct"].map(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
        )
        tbl["P&L $"] = tbl["pnl"].map(
            lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}" if pd.notna(x) else "—"
        )
        tbl["Result"] = tbl.apply(
            lambda r: "⏳" if r["status"] == "open"
            else ("✅" if r["correct"] else "❌"),
            axis=1,
        )
        display_cols = ["Date", "Direction", "Confidence", "Entry $", "Exit $", "Return %", "P&L $", "Result"]
        st.dataframe(
            tbl[display_cols].sort_values("Date", ascending=False).reset_index(drop=True),
            width="stretch",
        )

        # Summary metrics for closed predictions
        closed = outcomes[outcomes["status"] == "closed"]
        if not closed.empty:
            total_pnl = closed["pnl"].sum()
            accuracy = closed["correct"].mean()
            avg_return = closed["return_pct"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total P&L", f"${total_pnl:+,.2f}")
            c2.metric("Accuracy", f"{accuracy:.0%}")
            c3.metric("Avg Return", f"{avg_return:+.2%}")


if __name__ == "__main__":
    main()
