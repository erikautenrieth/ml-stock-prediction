import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ta

from backend.core.config import settings

from .theme import CANDLE_DOWN, CANDLE_UP, DOWN_COLOR, LAYOUT, UP_COLOR

# ATR Day marker colors (directional: green=up day, red=down day)
ATR_DAY_UP = "#00E676"  # green — up day (close > open)
ATR_DAY_UP_LIGHT = "rgba(0,230,118,0.25)"  # light green — 1× ATR up
ATR_DAY_DOWN = "#FF5252"  # red — down day (close < open)
ATR_DAY_DOWN_LIGHT = "rgba(255,82,82,0.25)"  # light red — 1× ATR down

# Dow Theory colors
DOW_UP_COLOR = "rgba(0,230,118,0.6)"  # green zigzag (uptrend)
DOW_DOWN_COLOR = "rgba(255,82,82,0.6)"  # red zigzag (downtrend)
DOW_NEUTRAL_COLOR = "rgba(255,255,255,0.3)"  # neutral/undefined


def _compute_atr_days(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Classify each day by its range relative to ATR (14)."""
    close, high, low, opn = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Open"]

    atr = ta.volatility.average_true_range(high, low, close, window=14)
    daily_range = high - low
    atr_ratio = daily_range / atr
    is_up = close >= opn

    result = pd.DataFrame(index=ohlcv.index)
    result["atr_ratio"] = atr_ratio
    result["is_up"] = is_up
    result["double_atr"] = atr_ratio >= 2.0
    result["single_atr"] = (atr_ratio >= 1.0) & (atr_ratio < 2.0)

    return result


def _detect_dow_swings(ohlcv: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Detect Dow Theory swing highs/lows and classify as HH/HL/LH/LL.

    Args:
        ohlcv: OHLCV DataFrame.
        order: Number of bars on each side to confirm a swing point.

    Returns:
        DataFrame with columns: date, price, swing_type ('high'/'low'),
        label ('HH','HL','LH','LL'), trend ('up','down','neutral').
    """
    high = ohlcv["High"].values
    low = ohlcv["Low"].values
    dates = ohlcv.index

    swings = []

    # Detect swing highs: high[i] is the max in window [i-order, i+order]
    for i in range(order, len(high) - order):
        window_high = high[i - order : i + order + 1]
        if high[i] == np.max(window_high) and np.sum(window_high == high[i]) == 1:
            swings.append({"date": dates[i], "price": high[i], "swing_type": "high"})

        window_low = low[i - order : i + order + 1]
        if low[i] == np.min(window_low) and np.sum(window_low == low[i]) == 1:
            swings.append({"date": dates[i], "price": low[i], "swing_type": "low"})

    if not swings:
        return pd.DataFrame(columns=["date", "price", "swing_type", "label", "trend"])

    df = pd.DataFrame(swings).sort_values("date").reset_index(drop=True)

    # Remove consecutive same-type swings (keep the more extreme one)
    cleaned = [df.iloc[0].to_dict()]
    for i in range(1, len(df)):
        row = df.iloc[i].to_dict()
        if row["swing_type"] == cleaned[-1]["swing_type"]:
            if row["swing_type"] == "high" and row["price"] > cleaned[-1]["price"]:
                cleaned[-1] = row
            elif row["swing_type"] == "low" and row["price"] < cleaned[-1]["price"]:
                cleaned[-1] = row
        else:
            cleaned.append(row)
    df = pd.DataFrame(cleaned).reset_index(drop=True)

    # Classify: HH/HL/LH/LL
    prev_high = None
    prev_low = None
    labels = []
    for _, row in df.iterrows():
        if row["swing_type"] == "high":
            if prev_high is None:
                labels.append("H")
            elif row["price"] > prev_high:
                labels.append("HH")
            else:
                labels.append("LH")
            prev_high = row["price"]
        else:
            if prev_low is None:
                labels.append("L")
            elif row["price"] > prev_low:
                labels.append("HL")
            else:
                labels.append("LL")
            prev_low = row["price"]
    df["label"] = labels

    # Count sequences: consecutive bullish (HH/HL) or bearish (LH/LL) points
    seq_counts = []
    seq_dirs = []  # "long" or "short"
    count = 0
    current_dir = None

    for lbl in labels:
        if lbl in ("HH", "HL"):
            if current_dir == "long":
                count += 1
            else:
                current_dir = "long"
                count = 1
        elif lbl in ("LH", "LL"):
            if current_dir == "short":
                count += 1
            else:
                current_dir = "short"
                count = 1
        else:
            count = 0
            current_dir = None

        seq_counts.append(count)
        seq_dirs.append(current_dir)

    df["seq_count"] = seq_counts
    df["seq_dir"] = seq_dirs

    # Determine trend at each swing point
    trends = ["neutral"]
    for i in range(1, len(df)):
        lbl = df.iloc[i]["label"]
        prev_lbl = df.iloc[i - 1]["label"]
        if lbl in ("HH", "HL") and prev_lbl in ("HH", "HL"):
            trends.append("up")
        elif lbl in ("LH", "LL") and prev_lbl in ("LH", "LL"):
            trends.append("down")
        else:
            trends.append(trends[-1])  # carry previous trend
    df["trend"] = trends

    return df


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

    # --- ATR Day markers (only show 2× ATR days, yellow border) ---
    atr_days = _compute_atr_days(ohlcv)

    double_days = atr_days[atr_days["double_atr"]]
    if not double_days.empty:
        for is_up, dir_label in [(True, "Up"), (False, "Down")]:
            mask = double_days["is_up"] == is_up
            days = double_days[mask]
            if days.empty:
                continue

            fill = ATR_DAY_UP if is_up else ATR_DAY_DOWN
            symbol = "triangle-up" if is_up else "triangle-down"

            ratios = days["atr_ratio"]
            fig.add_trace(
                go.Scatter(
                    x=days.index,
                    y=ohlcv.loc[days.index, "Close"],
                    mode="markers",
                    marker={
                        "symbol": symbol,
                        "size": 9,
                        "color": fill,
                        "line": {"width": 1, "color": "#FFD600"},
                    },
                    name=f"2× ATR {dir_label}",
                    text=[f"{r:.1f}× ATR" for r in ratios],
                    hovertemplate="%{x|%b %d}<br>%{text}<br>$%{y:,.2f}<extra></extra>",
                )
            )

    # --- Dow Theory zigzag ---
    swings = _detect_dow_swings(ohlcv, order=5)

    if len(swings) >= 2:
        # Zigzag line segments colored by trend
        for i in range(1, len(swings)):
            s0 = swings.iloc[i - 1]
            s1 = swings.iloc[i]
            trend = s1["trend"]
            if trend == "up":
                color = DOW_UP_COLOR
            elif trend == "down":
                color = DOW_DOWN_COLOR
            else:
                color = DOW_NEUTRAL_COLOR

            fig.add_trace(
                go.Scatter(
                    x=[s0["date"], s1["date"]],
                    y=[s0["price"], s1["price"]],
                    mode="lines",
                    line={"color": color, "width": 1.5},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Swing point labels with sequence count (HH/HL/LH/LL + number)
        for _, sw in swings.iterrows():
            if sw["label"] in ("H", "L"):
                continue  # skip first unlabeled points
            if sw["label"] in ("HH", "HL"):
                lbl_color = "#00E676"
            else:
                lbl_color = "#FF5252"

            # Show label + sequence number
            seq_num = int(sw["seq_count"])
            label_text = f"<b>{sw['label']}</b>"
            if seq_num > 0:
                label_text = f"<b>{sw['label']} {seq_num}</b>"

            fig.add_annotation(
                x=sw["date"],
                y=sw["price"],
                text=label_text,
                showarrow=False,
                font={"size": 9, "color": lbl_color},
                yshift=14 if sw["swing_type"] == "high" else -14,
            )

            # Horizontal S/R line at each swing point
            fig.add_shape(
                type="line",
                x0=sw["date"],
                x1=ohlcv.index[-1],
                y0=sw["price"],
                y1=sw["price"],
                line={
                    "color": lbl_color,
                    "width": 0.7,
                    "dash": "dot",
                },
                opacity=0.25,
            )

            # Sequence start marker at point 3 → "Trend Long" / "Trend Short"
            if seq_num == 3:
                if sw["seq_dir"] == "long":
                    trend_text = "▲ SQ Long"
                    trend_color = "#00E676"
                else:
                    trend_text = "▼ SQ Short"
                    trend_color = "#FF5252"

                fig.add_annotation(
                    x=sw["date"],
                    y=sw["price"],
                    text=f"<b>{trend_text}</b>",
                    showarrow=True,
                    arrowhead=0,
                    arrowcolor=trend_color,
                    arrowwidth=1.5,
                    ax=40,
                    ay=-30 if sw["swing_type"] == "high" else 30,
                    font={"size": 11, "color": trend_color},
                    bgcolor="rgba(14,17,23,0.8)",
                    bordercolor=trend_color,
                    borderwidth=1,
                    borderpad=3,
                )

                # Stop-Loss at last significant high
                idx = swings.index.get_loc(sw.name)
                prev_swings = swings.iloc[:idx]
                highs = prev_swings[prev_swings["swing_type"] == "high"]
                sl = highs.iloc[-1] if not highs.empty else None

                if sl is not None:
                    sl_color = "#FF5252" if sw["seq_dir"] == "long" else "#00E676"

                    # Thick dashed SL line from signal date to chart end
                    fig.add_shape(
                        type="line",
                        x0=sw["date"],
                        x1=ohlcv.index[-1],
                        y0=sl["price"],
                        y1=sl["price"],
                        line={
                            "color": sl_color,
                            "width": 2,
                            "dash": "dash",
                        },
                        opacity=0.6,
                    )

                    # SL label
                    fig.add_annotation(
                        x=ohlcv.index[-1],
                        y=sl["price"],
                        text=f"<b>SL ${sl['price']:,.0f}</b>",
                        showarrow=False,
                        font={"size": 10, "color": sl_color},
                        xanchor="left",
                        xshift=5,
                        bgcolor="rgba(14,17,23,0.8)",
                        bordercolor=sl_color,
                        borderwidth=1,
                        borderpad=2,
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
