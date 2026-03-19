import pandas as pd
import plotly.graph_objs as go

from backend.core.config import settings

from .theme import DOWN_COLOR, HORIZON, LAYOUT, UP_COLOR


def compute_prediction_outcomes(
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute actual outcomes for each prediction after HORIZON days."""
    rows = []
    for date, row in predictions.iterrows():
        predicted_up = int(row["prediction"]) == 1
        confidence = float(row["confidence"])

        # Use stored entry_price if available, otherwise look up in OHLCV
        if "entry_price" in row and pd.notna(row.get("entry_price")):
            entry_price = float(row["entry_price"])
        elif date in ohlcv.index:
            entry_price = float(ohlcv.loc[date, "Close"])
        else:
            # No price data at all — still show the prediction as open
            rows.append(
                {
                    "date": date,
                    "direction": "UP" if predicted_up else "DOWN",
                    "confidence": confidence,
                    "entry_price": None,
                    "exit_price": None,
                    "return_pct": None,
                    "pnl": None,
                    "correct": None,
                    "status": "open",
                }
            )
            continue

        future = date + pd.Timedelta(days=HORIZON)
        future_dates = ohlcv.index[ohlcv.index >= future]
        if len(future_dates) == 0:
            rows.append(
                {
                    "date": date,
                    "direction": "UP" if predicted_up else "DOWN",
                    "confidence": confidence,
                    "entry_price": entry_price,
                    "exit_price": None,
                    "return_pct": None,
                    "pnl": None,
                    "correct": None,
                    "status": "open",
                }
            )
            continue
        exit_date = future_dates[0]
        exit_price = float(ohlcv.loc[exit_date, "Close"])
        pnl = exit_price - entry_price
        return_pct = pnl / entry_price
        actual_up = exit_price > entry_price
        correct = predicted_up == actual_up

        rows.append(
            {
                "date": date,
                "direction": "UP" if predicted_up else "DOWN",
                "confidence": confidence,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": return_pct,
                "pnl": pnl,
                "correct": correct,
                "status": "closed",
            }
        )
    return pd.DataFrame(rows)


def prediction_performance_chart(
    outcomes: pd.DataFrame,
) -> go.Figure:
    """Bar chart showing return % per prediction, colored by correct/wrong."""
    closed = outcomes[outcomes["status"] == "closed"].copy()
    if closed.empty:
        return go.Figure()

    colors = [UP_COLOR if c else DOWN_COLOR for c in closed["correct"]]
    labels = [f"{r:+.1%}" for r in closed["return_pct"]]

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
                hovertemplate=("Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<br><extra></extra>"),
            )
        ]
    )
    fig.add_hline(
        y=0,
        line_color="rgba(255,255,255,0.3)",
        line_width=1,
    )
    fig.update_layout(
        **LAYOUT,
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
    """Close price line with prediction markers and outcome lines."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["Close"],
                mode="lines",
                name="Close",
                line={"color": "#4FC3F7", "width": 2},
            )
        ]
    )

    for i, (date, row) in enumerate(predictions.iterrows(), 1):
        if date not in ohlcv.index:
            continue
        is_up = int(row["prediction"]) == 1
        price_at = float(ohlcv.loc[date, "Close"])
        color = UP_COLOR if is_up else DOWN_COLOR
        direction = "▲" if is_up else "▼"

        fig.add_trace(
            go.Scatter(
                x=[date],
                y=[price_at],
                mode="markers+text",
                marker={
                    "size": 14,
                    "color": color,
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#fff"},
                },
                text=[f"{direction} #{i}"],
                textposition="top center",
                textfont={"color": color, "size": 12},
                showlegend=False,
                hovertemplate=f"Prediction #{i}<br>Price: $%{{y:,.0f}}<br>Direction: {'UP' if is_up else 'DOWN'}<extra></extra>",
            )
        )

        future = date + pd.Timedelta(days=HORIZON)
        future_dates = ohlcv.index[ohlcv.index >= future]
        if len(future_dates) == 0:
            continue
        future_date = future_dates[0]
        future_price = float(ohlcv.loc[future_date, "Close"])
        diff = future_price - price_at
        pct = diff / price_at * 100
        outcome_color = UP_COLOR if diff > 0 else DOWN_COLOR
        sign = "+" if diff > 0 else ""

        fig.add_trace(
            go.Scatter(
                x=[date, future_date],
                y=[price_at, future_price],
                mode="lines+markers",
                name=f"#{i} {sign}{pct:.1f}%",
                line={"color": outcome_color, "dash": "dot", "width": 2.5},
                marker={"size": 6, "color": outcome_color},
                showlegend=True,
                hovertemplate=f"#{i}: {sign}${diff:,.0f} ({sign}{pct:.1f}%)<extra></extra>",
            )
        )

    # Y-axis: zoom to price range with 2% padding
    close_vals = ohlcv["Close"]
    y_min = float(close_vals.min()) * 0.98
    y_max = float(close_vals.max()) * 1.02

    fig.update_layout(
        **LAYOUT,
        title=f"{settings.stock.symbol_display} — Predictions & Outcomes ({HORIZON}d)",
        height=520,
    )
    fig.update_yaxes(range=[y_min, y_max])
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
        **LAYOUT,
        title="Prediction Confidence",
        height=350,
    )
    fig.update_yaxes(title="Confidence", range=[0.3, 1], tickformat=".0%", tickprefix="")
    return fig
