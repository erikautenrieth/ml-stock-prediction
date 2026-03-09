import pandas as pd
import streamlit as st

from backend.core.config import settings
from frontend.charts import candlestick_chart, confidence_chart, prediction_line_chart
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

    # --- Charts ---
    tab_candle, tab_line, tab_confidence = st.tabs(
        ["🕯️ Candlestick", "📊 Line + Outcomes", "🎯 Confidence"]
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

    # --- Prediction history table ---
    if not visible_preds.empty:
        st.subheader("Prediction History")
        display_df = visible_preds.copy()
        display_df.index = pd.to_datetime(display_df.index).strftime("%Y-%m-%d")
        display_df.index.name = "Date"
        display_df["Direction"] = display_df["prediction"].map({1: "🟢 UP", 0: "🔴 DOWN"})
        display_df["Confidence"] = display_df["confidence"].map(lambda x: f"{x:.1%}")
        cols = ["Direction", "Confidence"]
        if "symbol" in display_df.columns:
            display_df["Symbol"] = display_df["symbol"]
            cols.append("Symbol")
        if "horizon_days" in display_df.columns:
            display_df["Horizon (days)"] = display_df["horizon_days"]
            cols.append("Horizon (days)")
        st.dataframe(
            display_df[cols].sort_index(ascending=False),
            width="stretch",
        )


if __name__ == "__main__":
    main()
