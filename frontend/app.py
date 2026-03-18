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
from frontend.model_card import render_model_card


def main() -> None:
    st.set_page_config(
        page_title=f"{settings.stock.symbol_display} Trend Prediction",
        page_icon="📈",
        layout="wide",
    )

    horizon = settings.stock.prediction_horizon_days

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

    visible_preds = predictions[predictions.index >= ohlcv.index.min()]

    # --- Hero header ---
    if not visible_preds.empty:
        latest = visible_preds.iloc[-1]
        latest_date = visible_preds.index[-1]
        is_up = int(latest["prediction"]) == 1
        conf = float(latest["confidence"])
        arrow = "▲" if is_up else "▼"
        color = "#00E676" if is_up else "#FF5252"
        direction = "UP" if is_up else "DOWN"
        date_str = pd.Timestamp(latest_date).strftime("%b %d, %Y")

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        padding:0.6em 0; margin-bottom:0.2em;">
                <div style="display:flex; align-items:center; gap:0.5em;">
                    <span style="font-size:2.2em; font-weight:700;">
                        {settings.stock.symbol_display}
                    </span>
                    <span style="font-size:1.1em; color:#888; font-weight:400;">
                        {horizon}d prediction
                    </span>
                </div>
                <div style="display:flex; align-items:center; gap:1.2em;">
                    <span style="font-size:1.6em; font-weight:700; color:{color};">
                        {arrow} {direction}
                    </span>
                    <span style="font-size:1.1em; color:#aaa;">
                        {conf:.0%} conf · {date_str}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="padding:0.6em 0; margin-bottom:0.2em;">
                <span style="font-size:2.2em; font-weight:700;">
                    {settings.stock.symbol_display}
                </span>
                <span style="font-size:1.1em; color:#888; font-weight:400;">
                    — {horizon}d prediction
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()

    # --- Compute outcomes ---
    outcomes = pd.DataFrame()
    if not visible_preds.empty:
        outcomes = compute_prediction_outcomes(ohlcv, visible_preds)

    # --- Tabs ---
    tab_candle, tab_perf, tab_confidence = st.tabs(
        ["🕯️ Candlestick", "📊 Performance", "🎯 Confidence"]
    )

    with tab_candle:
        if not visible_preds.empty:
            st.plotly_chart(
                candlestick_chart(ohlcv, visible_preds),
                use_container_width=True,
            )
            # Model info below chart
            render_model_card()
        else:
            st.plotly_chart(
                candlestick_chart(ohlcv, pd.DataFrame()),
                use_container_width=True,
            )
            st.info("No predictions available yet. Run `make predict` first.")

    with tab_perf:
        if not outcomes.empty:
            # Line + Outcomes chart (main view)
            if not visible_preds.empty:
                st.plotly_chart(
                    prediction_line_chart(ohlcv, visible_preds),
                    use_container_width=True,
                )
            # Return bar chart
            st.plotly_chart(
                prediction_performance_chart(outcomes),
                use_container_width=True,
            )
            # P&L table
            st.subheader("Prediction Results")
            tbl = outcomes.copy()
            tbl["Date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d")
            tbl["Direction"] = tbl["direction"].map({"UP": "🟢 UP", "DOWN": "🔴 DOWN"})
            tbl["Confidence"] = tbl["confidence"].map(lambda x: f"{x:.1%}")
            tbl["Entry $"] = tbl["entry_price"].map(lambda x: f"${x:,.2f}")
            tbl["Exit $"] = tbl["exit_price"].map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "⏳ pending"
            )
            tbl["Return %"] = tbl["return_pct"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
            tbl["P&L $"] = tbl["pnl"].map(
                lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}" if pd.notna(x) else "—"
            )
            tbl["Result"] = tbl.apply(
                lambda r: "⏳" if r["status"] == "open" else ("✅" if r["correct"] else "❌"),
                axis=1,
            )
            display_cols = [
                "Date",
                "Direction",
                "Confidence",
                "Entry $",
                "Exit $",
                "Return %",
                "P&L $",
                "Result",
            ]
            st.dataframe(
                tbl[display_cols].sort_values("Date", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
            # Summary metrics
            closed = outcomes[outcomes["status"] == "closed"]
            if not closed.empty:
                total_pnl = closed["pnl"].sum()
                accuracy = closed["correct"].mean()
                avg_return = closed["return_pct"].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total P&L", f"${total_pnl:+,.2f}")
                c2.metric("Accuracy", f"{accuracy:.0%}")
                c3.metric("Avg Return", f"{avg_return:+.2%}")
        else:
            st.info("No predictions available yet.")

    with tab_confidence:
        if not visible_preds.empty:
            st.plotly_chart(
                confidence_chart(visible_preds),
                use_container_width=True,
            )
        else:
            st.info("No predictions available yet.")


if __name__ == "__main__":
    main()
