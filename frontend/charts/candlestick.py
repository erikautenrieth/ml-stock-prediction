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
                mode="markers+text",
                marker={"symbol": "triangle-up", "size": 16, "color": UP_COLOR},
                text=["▲"] * len(up_dates),
                textposition="top center",
                textfont={"color": UP_COLOR, "size": 16},
                name="Prediction UP",
                showlegend=True,
            )
        )
    if dn_dates:
        fig.add_trace(
            go.Scatter(
                x=dn_dates,
                y=dn_prices,
                mode="markers+text",
                marker={"symbol": "triangle-down", "size": 16, "color": DOWN_COLOR},
                text=["▼"] * len(dn_dates),
                textposition="bottom center",
                textfont={"color": DOWN_COLOR, "size": 16},
                name="Prediction DOWN",
                showlegend=True,
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
