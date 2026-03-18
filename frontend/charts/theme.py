import plotly.graph_objs as go

from backend.core.config import settings

HORIZON = settings.stock.prediction_horizon_days

# --- Colors ---
UP_COLOR = "#00E676"
DOWN_COLOR = "#FF5252"
CANDLE_UP = "#26A69A"
CANDLE_DOWN = "#EF5350"

# --- Shared Plotly layout ---
LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#0E1117",
    "font": {"family": "Inter, sans-serif", "size": 13, "color": "#FAFAFA"},
    "margin": {"l": 50, "r": 20, "t": 70, "b": 40},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.5, "xanchor": "center"},
    "xaxis": {"gridcolor": "#1E2A3A", "zeroline": False},
    "yaxis": {"gridcolor": "#1E2A3A", "zeroline": False, "tickprefix": "$", "tickformat": ",.0f"},
}
