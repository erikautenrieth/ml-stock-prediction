import pandas as pd
import plotly.graph_objs as go

from backend.core.config import settings

HORIZON = settings.stock.prediction_horizon_days

# --- Shared theme ---
_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#0E1117",
    "font": {"family": "Inter, sans-serif", "size": 13, "color": "#FAFAFA"},
    "margin": {"l": 50, "r": 20, "t": 70, "b": 40},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.5, "xanchor": "center"},
    "xaxis": {"gridcolor": "#1E2A3A", "zeroline": False},
    "yaxis": {"gridcolor": "#1E2A3A", "zeroline": False, "tickprefix": "$", "tickformat": ",.0f"},
}

UP_COLOR = "#00E676"
DOWN_COLOR = "#FF5252"
CANDLE_UP = "#26A69A"
CANDLE_DOWN = "#EF5350"


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

    # Prediction markers as scatter with big symbols
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
        **_LAYOUT,
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


def compute_prediction_outcomes(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute actual outcomes for each prediction after HORIZON days."""
    rows = []
    for date, row in predictions.iterrows():
        if date not in ohlcv.index:
            continue
        entry_price = float(ohlcv.loc[date, "Close"])
        predicted_up = int(row["prediction"]) == 1
        confidence = float(row["confidence"])

        future = date + pd.Timedelta(days=HORIZON)
        future_dates = ohlcv.index[ohlcv.index >= future]
        if len(future_dates) == 0:
            rows.append({
                "date": date,
                "direction": "UP" if predicted_up else "DOWN",
                "confidence": confidence,
                "entry_price": entry_price,
                "exit_price": None,
                "return_pct": None,
                "pnl": None,
                "correct": None,
                "status": "open",
            })
            continue
        exit_date = future_dates[0]
        exit_price = float(ohlcv.loc[exit_date, "Close"])
        pnl = exit_price - entry_price
        return_pct = pnl / entry_price
        actual_up = exit_price > entry_price
        correct = predicted_up == actual_up

        rows.append({
            "date": date,
            "direction": "UP" if predicted_up else "DOWN",
            "confidence": confidence,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": return_pct,
            "pnl": pnl,
            "correct": correct,
            "status": "closed",
        })
    return pd.DataFrame(rows)


def prediction_performance_chart(
    outcomes: pd.DataFrame,
) -> go.Figure:
    """Bar chart showing return % per prediction, colored by correct/wrong."""
    closed = outcomes[outcomes["status"] == "closed"].copy()
    if closed.empty:
        return go.Figure()

    colors = [
        UP_COLOR if c else DOWN_COLOR
        for c in closed["correct"]
    ]
    labels = [
        f"{r:+.1%}" for r in closed["return_pct"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=closed["date"],
                y=closed["return_pct"] * 100,
                marker_color=colors,
                marker_line_width=0,
                text=labels,
                textposition="outside",
                textfont={"size": 11},
                name="Return %",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Return: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(
        y=0,
        line_color="rgba(255,255,255,0.3)",
        line_width=1,
    )
    fig.update_layout(
        **_LAYOUT,
        title=f"{settings.stock.symbol_display} — Prediction Returns ({HORIZON}d)",
        height=420,
    )
    fig.update_yaxes(title="Return %", ticksuffix="%", tickprefix="")
    fig.update_xaxes(title="")
    return fig


def prediction_line_chart(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
) -> go.Figure:
    """Close price line with prediction arrows and outcome lines."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                name="Close",
                line={"color": "#4FC3F7", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(79,195,247,0.08)",
            )
        ]
    )

    for i, (date, row) in enumerate(predictions.iterrows(), 1):
        if date not in ohlcv.index:
            continue
        is_up = int(row["prediction"]) == 1
        price_at = float(ohlcv.loc[date, "Close"])
        color = UP_COLOR if is_up else DOWN_COLOR

        # Marker at prediction point
        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price_at],
                mode="markers+text",
                marker={"size": 12, "color": color, "symbol": "diamond"},
                text=[f"#{i}"],
                textposition="top center",
                textfont={"color": color, "size": 11},
                showlegend=False,
            )
        )

        # Outcome line to actual future price
        future = date + pd.Timedelta(days=HORIZON)
        future_dates = ohlcv.index[ohlcv.index >= future]
        if len(future_dates) == 0:
            continue
        future_date = future_dates[0]
        future_price = float(ohlcv.loc[future_date, "Close"])
        diff = future_price - price_at
        outcome_color = UP_COLOR if diff > 0 else DOWN_COLOR
        sign = "+" if diff > 0 else ""

        fig.add_trace(
            go.Scatter(
                x=[date, future_date],
                y=[price_at, future_price],
                mode="lines",
                name=f"#{i} {sign}{diff:.0f}",
                line={"color": outcome_color, "dash": "dot", "width": 1.5},
                showlegend=True,
            )
        )

    fig.update_layout(
        **_LAYOUT,
        title=(f"{settings.stock.symbol_display} — Predictions & Outcomes ({HORIZON}d)"),
        height=500,
    )
    return fig


def confidence_chart(predictions: pd.DataFrame) -> go.Figure:
    """Bar chart of prediction confidence over time."""
    if predictions.empty or "confidence" not in predictions.columns:
        return go.Figure()

    colors = [UP_COLOR if int(p) == 1 else DOWN_COLOR for p in predictions["prediction"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=predictions.index,
                y=predictions["confidence"],
                marker_color=colors,
                marker_line_width=0,
                name="Confidence",
                opacity=0.85,
            )
        ]
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="50%",
        annotation_font_color="#888",
    )
    fig.update_layout(
        **_LAYOUT,
        title="Prediction Confidence",
        height=350,
    )
    fig.update_yaxes(title="Confidence", range=[0.3, 1], tickformat=".0%", tickprefix="")
    return fig
