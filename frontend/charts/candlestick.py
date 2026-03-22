import pandas as pd
import plotly.graph_objs as go
import ta

from backend.core.config import settings

from .theme import CANDLE_DOWN, CANDLE_UP, DOWN_COLOR, LAYOUT, UP_COLOR

# ATR strategy signal colors (distinct from ML prediction colors)
ATR_BUY_COLOR = "#00BCD4"  # cyan  (buy = cool/positive)
ATR_SELL_COLOR = "#FF9800"  # orange (sell = warm/warning)
ATR_BAND_COLOR = "rgba(255,152,0,0.18)"


def _compute_atr_signals(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute ATR-Keltner breakout signals for 10-day holding."""
    close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

    atr = ta.volatility.average_true_range(high, low, close, window=14)
    ema20 = ta.trend.ema_indicator(close, window=20)

    upper = ema20 + 1.5 * atr
    lower = ema20 - 1.5 * atr

    broke_upper = (close > upper) & (close.shift(1) <= upper.shift(1))
    broke_lower = (close < lower) & (close.shift(1) >= lower.shift(1))

    signals = pd.DataFrame(index=ohlcv.index)
    signals["keltner_upper"] = upper
    signals["keltner_lower"] = lower
    signals["signal"] = ""
    signals.loc[broke_upper, "signal"] = "BUY"
    signals.loc[broke_lower, "signal"] = "SELL"

    return signals


def candlestick_chart(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
) -> go.Figure:
    """Candlestick chart with prediction markers overlaid."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["Open"],
                high=ohlcv["High"],
                low=ohlcv["Low"],
                close=ohlcv["Close"],
                name=settings.stock.symbol_display,
                increasing_line_color=CANDLE_UP,
                increasing_fillcolor=CANDLE_UP,
                decreasing_line_color=CANDLE_DOWN,
                decreasing_fillcolor=CANDLE_DOWN,
            )
        ]
    )

    up_dates, up_prices, dn_dates, dn_prices = [], [], [], []
    for date, row in predictions.iterrows():
        if date not in ohlcv.index:
            continue
        price = float(ohlcv.loc[date, "Close"])
        if int(row["prediction"]) == 1:
            up_dates.append(date)
            up_prices.append(price)
        else:
            dn_dates.append(date)
            dn_prices.append(price)

    if up_dates:
        fig.add_trace(
            go.Scatter(
                x=up_dates,
                y=up_prices,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 16,
                    "color": UP_COLOR,
                    "line": {"width": 1, "color": "rgba(0,0,0,0.3)"},
                },
                name="Prediction UP",
                hovertemplate="%{x|%b %d}<br>UP @ $%{y:,.2f}<extra></extra>",
            )
        )
    if dn_dates:
        fig.add_trace(
            go.Scatter(
                x=dn_dates,
                y=dn_prices,
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": 16,
                    "color": DOWN_COLOR,
                    "line": {"width": 1, "color": "rgba(0,0,0,0.3)"},
                },
                name="Prediction DOWN",
                hovertemplate="%{x|%b %d}<br>DOWN @ $%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        **LAYOUT,
        title={
            "text": f"{settings.stock.symbol_display} — Price & Predictions",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_rangeslider_visible=False,
        xaxis_rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"step": "all"},
            ],
            "font": {"color": "#FAFAFA"},
            "bgcolor": "#1E2A3A",
            "activecolor": "#3D5A80",
            "y": 1.15,
        },
        height=620,
    )
    fig.update_layout(margin={"l": 50, "r": 20, "t": 90, "b": 40})
    return fig


def price_line_chart(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
) -> go.Figure:
    """Clean line chart (Close price) with prediction markers overlaid."""
    fig = go.Figure()

    # Invisible baseline trace for the fill area (anchored near data minimum)
    y_min = ohlcv["Close"].min() * 0.97
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=[y_min] * len(ohlcv),
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main price line with fill down to baseline
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=ohlcv["Close"],
            mode="lines",
            line={"color": "#5C9DFF", "width": 2.5},
            name=settings.stock.symbol_display,
            fill="tonexty",
            fillcolor="rgba(92,157,255,0.15)",
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
        )
    )

    # 20-day moving average as subtle context line
    if len(ohlcv) >= 20:
        ma20 = ohlcv["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ma20,
                mode="lines",
                line={"color": "rgba(255,255,255,0.25)", "width": 1.5, "dash": "dot"},
                name="20d MA",
                hoverinfo="skip",
            )
        )

    up_dates, up_prices, dn_dates, dn_prices = [], [], [], []
    for date, row in predictions.iterrows():
        if date not in ohlcv.index:
            continue
        price = float(ohlcv.loc[date, "Close"])
        if int(row["prediction"]) == 1:
            up_dates.append(date)
            up_prices.append(price)
        else:
            dn_dates.append(date)
            dn_prices.append(price)

    if up_dates:
        fig.add_trace(
            go.Scatter(
                x=up_dates,
                y=up_prices,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 12,
                    "color": UP_COLOR,
                    "line": {"width": 1, "color": "rgba(0,0,0,0.3)"},
                },
                name="Prediction UP",
                hovertemplate="%{x|%b %d}<br>UP @ $%{y:,.2f}<extra></extra>",
            )
        )
    if dn_dates:
        fig.add_trace(
            go.Scatter(
                x=dn_dates,
                y=dn_prices,
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": 12,
                    "color": DOWN_COLOR,
                    "line": {"width": 1, "color": "rgba(0,0,0,0.3)"},
                },
                name="Prediction DOWN",
                hovertemplate="%{x|%b %d}<br>DOWN @ $%{y:,.2f}<extra></extra>",
            )
        )

    # --- ATR-Keltner strategy signals ---
    atr_signals = _compute_atr_signals(ohlcv)

    # Keltner bands (subtle)
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=atr_signals["keltner_upper"],
            mode="lines",
            line={"color": ATR_BUY_COLOR, "width": 1, "dash": "dash"},
            name="Keltner Upper",
            hoverinfo="skip",
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=atr_signals["keltner_lower"],
            mode="lines",
            line={"color": ATR_SELL_COLOR, "width": 1, "dash": "dash"},
            name="Keltner Lower",
            hoverinfo="skip",
            opacity=0.4,
        )
    )

    # ATR buy/sell markers (diamonds — distinct from ML triangles)
    atr_buys = atr_signals[atr_signals["signal"] == "BUY"]
    atr_sells = atr_signals[atr_signals["signal"] == "SELL"]

    if not atr_buys.empty:
        fig.add_trace(
            go.Scatter(
                x=atr_buys.index,
                y=ohlcv.loc[atr_buys.index, "Close"],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 11,
                    "color": ATR_BUY_COLOR,
                    "line": {"width": 1.5, "color": "white"},
                },
                name="ATR Buy",
                hovertemplate="%{x|%b %d}<br>ATR BUY @ $%{y:,.2f}<extra></extra>",
            )
        )
    if not atr_sells.empty:
        fig.add_trace(
            go.Scatter(
                x=atr_sells.index,
                y=ohlcv.loc[atr_sells.index, "Close"],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 11,
                    "color": ATR_SELL_COLOR,
                    "line": {"width": 1.5, "color": "white"},
                },
                name="ATR Sell",
                hovertemplate="%{x|%b %d}<br>ATR SELL @ $%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        **LAYOUT,
        title={
            "text": f"{settings.stock.symbol_display} — Price & Predictions",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_rangeslider_visible=False,
        xaxis_rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"step": "all"},
            ],
            "font": {"color": "#FAFAFA"},
            "bgcolor": "#1E2A3A",
            "activecolor": "#3D5A80",
            "y": 1.15,
        },
        height=620,
    )
    fig.update_layout(margin={"l": 50, "r": 20, "t": 90, "b": 40})
    return fig
