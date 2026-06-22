import streamlit as st

from backend.core.config import settings

HORIZON = settings.stock.prediction_horizon_days

# --- Colors ---
UP_COLOR = "#00E676"
DOWN_COLOR = "#FF5252"
CANDLE_UP = "#26A69A"
CANDLE_DOWN = "#EF5350"


def _is_dark_mode() -> bool:
    """Check if dark mode is active via session state."""
    return st.session_state.get("theme_dark", True)


def get_layout() -> dict:
    """Return Plotly layout dict based on current theme."""
    if _is_dark_mode():
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "#0E1117",
            "font": {"family": "Inter, sans-serif", "size": 13, "color": "#FAFAFA"},
            "margin": {"l": 50, "r": 20, "t": 70, "b": 40},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {"gridcolor": "#1E2A3A", "zeroline": False},
            "yaxis": {
                "gridcolor": "#1E2A3A",
                "zeroline": False,
                "tickprefix": "$",
                "tickformat": ",.0f",
            },
        }
    return {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Inter, sans-serif", "size": 13, "color": "#1a1a1a"},
        "margin": {"l": 50, "r": 20, "t": 70, "b": 40},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {"gridcolor": "#E0E0E0", "zeroline": False},
        "yaxis": {
            "gridcolor": "#E0E0E0",
            "zeroline": False,
            "tickprefix": "$",
            "tickformat": ",.0f",
        },
    }


# Keep LAYOUT as a property-like access for backward compat
# All chart modules use **LAYOUT — we make it a dynamic getter via module-level variable
# that's re-evaluated each render cycle
class _LayoutProxy(dict):
    """Dict that refreshes values from get_layout() on every access."""

    def __getitem__(self, key):
        return get_layout()[key]

    def __iter__(self):
        return iter(get_layout())

    def __len__(self):
        return len(get_layout())

    def keys(self):
        return get_layout().keys()

    def values(self):
        return get_layout().values()

    def items(self):
        return get_layout().items()


LAYOUT = _LayoutProxy()
