# ML Stock Prediction

Machine learning pipeline for S&P 500 directional forecasting (up/down/neutral) using technical indicators, with MLflow experiment tracking and a Streamlit dashboard.

**[▶ Live Demo](https://ml-stock-pred.streamlit.app/)**: Interactive dashboard with candlestick charts, model predictions, and performance metrics for the S&P 500.

## Stack

```mermaid
graph LR
    subgraph DATA [" Data "]
        direction LR
        A[yfinance]:::data --> B[DuckDB]:::data
    end

    subgraph MODELS [" ML "]
        direction LR
        C[scikit-learn]:::ml
        D[LightGBM]:::ml
        E[Optuna]:::ml
        D --> E
    end

    subgraph TRACK [" Tracking "]
        direction LR
        F[MLflow]:::ops --> G[DVC]:::ops
    end

    subgraph FRONT [" Frontend "]
        H[Streamlit]:::ui
    end

    B --> C
    B --> D
    C --> F
    D --> F
    F --> H

    classDef data fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef ml fill:#fff3e0,stroke:#f57c00,color:#e65100
    classDef ops fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef ui fill:#fce4ec,stroke:#c62828,color:#b71c1c
```

<p align="center">
  <img src="https://img.shields.io/badge/yfinance-purple?logo=yahoo&logoColor=white" alt="yfinance" />
  <img src="https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black" alt="DuckDB" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/LightGBM-9558B2?logo=lightgbm&logoColor=white" alt="LightGBM" />
  <img src="https://img.shields.io/badge/Optuna-2196F3?logo=python&logoColor=white" alt="Optuna" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white" alt="DVC" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Python_3.12-3776AB?logo=python&logoColor=white" alt="Python" />
</p>

## Quickstart

```bash
poetry install
make pipeline   # fetch → train → predict
make fe         # launch dashboard
```

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| Fetch | `make fetch` | Downloads OHLCV data from Yahoo Finance, computes technical indicators |
| Train | `make train` | Trains ExtraTrees (default) or LightGBM with Optuna hyperparameter tuning |
| Predict | `make predict` | Generates predictions using the best registered model |

## Configuration

All parameters live in `params.yaml` — symbol, prediction horizon, feature windows, training window, tuning trials.

## Project Layout

- `backend/core/` — feature engineering, indicators, schemas
- `backend/ml/` — model training, tuning, registry
- `backend/workflows/` — pipeline entry points (fetch, train, predict)
- `frontend/` — Streamlit app with candlestick charts
- `tests/` — pytest suite

## CI/CD

Two GitHub Actions workflows run the pipeline automatically. **Daily** (Mon–Fri after market close): fetches the latest data and generates predictions with the current champion model. **Weekly** (Sunday): runs full Optuna retraining and promotes a new model only if it beats the existing one.
