import pandas as pd
import plotly.graph_objs as go

from backend.core.config import settings

from .theme import CANDLE_DOWN, CANDLE_UP, DOWN_COLOR, LAYOUT, UP_COLOR


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
