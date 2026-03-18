# Vorgeschlagene Projektstruktur

## Übersicht

```
stock-prediction/
├── README.md
├── LICENSE
├── Makefile                        # make setup / train / tune / predict / test / lint
├── pyproject.toml                  # Dependencies: frontend, backend, dev extras
├── dvc.yaml                        # Pipeline-DAG: fetch → train → predict
├── dvc.lock                        # Reproduzierbarkeit (Hashes)
├── params.yaml                     # HP-Defaults: n_estimators, max_depth, n_trials, etc.
├── .env-init                       # Env-Template (→ .env kopieren)
├── .env                            # Lokale Env-Variablen (git-ignored)
├── .gitignore
├── .dvc/
│   └── config                      # [core] remote = dagshub
│
├── Dockerfile                      # Multi-stage: builder + runtime
├── docker-compose.yml              # MLflow Server (lokal)
│
├── frontend/                       # ── UI Layer ──────────────────
│   ├── __init__.py
│   ├── app.py                      # Streamlit Entry — migriert aus streamlit_app.py
│   ├── charts.py                   # Plotly: Candlestick, Linien, Optional: Accuracy-Chart
│   ├── data_loader.py              # Daten-Zugriff (DataStore/DagsHub, kein yfinance)
│   └── .streamlit/
│       └── config.toml             # Streamlit Config (Theme, Port, CORS)
│
├── backend/                        # ── Server / ML Layer ─────────
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic Settings (DB, MLflow, Features, PipelineParams)
│   │   ├── schemas.py              # Pydantic Models für Datengrenzen:
│   │   │                           #   TrainingResult — Training-Output (accuracy, params, hash)
│   │   │                           #   PredictionRecord — einzelne Prediction (target ∈ {0,1})
│   │   │                           #   PipelineMetrics — metrics.json Schema (DagsHub-sichtbar)
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py    # get_data(), calc_target(), scale_data()
│   │   │   ├── indicators.py       # calc_indicators() — ta-Library Indikatoren
│   │   │   └── manifest.py         # FeatureManifest (Pydantic): Spalten + Hash + JSON-Serialisierung
│   │   └── prediction/
│   │       ├── __init__.py
│   │       ├── base.py             # PredictionService ABC
│   │       └── mlflow_predict.py   # Modell laden (@champion) → predict → speichern
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # BaseTrainer ABC (split, evaluate, train, create_model)
│   │   │   ├── extra_tree.py       # ExtraTreeTrainer + DEFAULT_PARAMS + TUNING_SPACE
│   │   │   └── tuning/
│   │   │       ├── __init__.py
│   │   │       └── optuna_tuner.py # ModelTuner — generisch, ersetzt Ray Tune
│   │   └── registry/
│   │       ├── __init__.py
│   │       └── model_registry.py   # log, promote (@champion), load + Optional: YAML Fallback
│   │
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── __init__.py         # get_data_store() Factory
│   │   │   ├── base.py             # DataStore ABC (save/get features + predictions)
│   │   │   └── duckdb_store.py     # DuckDB in-memory + Parquet I/O
│   │   └── storage/
│   │       ├── __init__.py         # get_storage() Factory
│   │       ├── base.py             # ArtifactStorage ABC
│   │       └── dagshub_storage.py  # DagsHub DVC (20 GB free)
│   │
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── fetch_data.py           # Daten holen → Features → DataStore
│   │   ├── train.py                # Daten laden → train/tune → Registry
│   │   └── predict.py              # Features laden → predict → speichern
│   │
│   ├── data/                       # git-ignored, DVC-tracked
│   │   ├── raw/
│   │   │   ├── sp500_ohlcv.parquet     # DVC-tracked: OHLCV-Rohdaten
│   │   │   └── sp500_ohlcv.parquet.dvc
│   │   ├── features/
│   │   │   ├── sp500_features.parquet  # DVC-tracked: techn. + fund. Indikatoren
│   │   │   └── sp500_features.parquet.dvc
│   │   ├── predictions/
│   │   │   ├── sp500_predictions.parquet
│   │   │   └── sp500_predictions.parquet.dvc
│   │   ├── models/
│   │   │   └── *.joblib                # Optional: Fallback ohne MLflow
│   │   ├── metrics.json                # DVC Metrics (git-tracked) — Schema: PipelineMetrics
│   │   └── mlflow/
│   │       └── mlruns/
│   │
│   └── notebooks/                  # Lokales Testen & Experimentieren
│       ├── 01_feature_engineering.ipynb # Features laden, Indikatoren, Datenqualität prüfen
│       ├── 02_model_training.ipynb     # Train ExtraTree, evaluate, Confusion Matrix
│       ├── 03_hyperparameter_tuning.ipynb # Optuna Tuning, Param-Space, Study-Visualisierung
│       ├── 04_database_store.ipynb     # DuckDB DataStore testen: save/get features+predictions
│       └── 05_full_pipeline.ipynb      # Kompletter Durchlauf: fetch → train → predict → store
│
├── tests/                          # Spiegelt backend/ Struktur
│   ├── __init__.py
│   ├── conftest.py                 # Shared Fixtures: sample DataFrames,
│   │                               #   Mock-DataStore, tmp DuckDB, MLflow Experiment
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_preprocessing.py   # calc_target (Up/Down korrekt?), get_data (Spalten, Index)
│   │   ├── test_indicators.py      # calc_indicators: alle Spalten vorhanden, keine NaNs,
│   │   │                           #   Wertebereiche (RSI 0-100, ADX 0-100)
│   │   └── test_feature_manifest.py # Spalten-Validierung, Hash-Konsistenz
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── test_training.py        # ExtraTreeTrainer: fit + predict, Accuracy > 0.5,
│   │   │                           #   Confusion Matrix Shape, MLflow-Logging
│   │   ├── test_tuning.py          # Optuna: Study erstellt, best_params vorhanden,
│   │   │                           #   n_trials korrekt, bestes Modell geladen
│   │   └── test_model_registry.py  # log_model, promote @champion, load_model roundtrip
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── test_duckdb_store.py    # save_features → get_features roundtrip,
│   │   │                           #   save_prediction → get_predictions, :memory: DB
│   │   └── test_datastore_abc.py   # Interface-Vertrag: alle Methoden implementiert
│   └── workflows/
│       ├── __init__.py
│       └── test_predict.py         # Prediction-Pipeline E2E: Features laden → predict
│                                   #   → Ergebnis in DataStore, Target ∈ {0, 1}
│
├── docs/
│   └── resources/
│       ├── commands.md
│       └── financial_data_sources.md
│
└── .github/
    └── workflows/
        ├── daily_pipeline.yml      # Mo-Fr 22:00 UTC: fetch + predict → dvc push
        └── weekly_train.yml        # So: Re-Training → Optional: Alert bei Accuracy-Drop
```


---

## Prozess-Übersicht (Top View)

→ Siehe [docs/process_overview.md](process_overview.md)

## Modell-Verwaltung

### DagsHub + MLflow Setup

DagsHub hostet einen MLflow-Server pro Repository unter `https://dagshub.com/<user>/<repo>.mlflow`.
Setup via [`dagshub.init()`](https://dagshub.com/docs/integration_guide/mlflow_tracking/) — konfiguriert URI + Auth automatisch:

```python
# backend/ml/registry/model_registry.py
import dagshub
import mlflow
from mlflow import MlflowClient
from backend.core.config import settings

# Einmalig pro Prozess — setzt MLFLOW_TRACKING_URI + Credentials
dagshub.init(
    repo_owner=settings.dagshub.repo_owner,
    repo_name=settings.dagshub.repo_name,
    mlflow=True,
)
mlflow.set_experiment(settings.mlflow.experiment_name)
client = MlflowClient()


def log_and_promote(model, accuracy: float, X_train=None) -> str:
    """Modell loggen, als @champion promoten wenn besser als aktuelles."""
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            model, "model",
            input_example=X_train[:1] if X_train is not None else None,
        )
        mlflow.log_metric("accuracy", accuracy)

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, settings.mlflow.model_name)

        try:
            champ = client.get_model_version_by_alias(settings.mlflow.model_name, "champion")
            champ_acc = client.get_run(champ.run_id).data.metrics.get("accuracy", 0)
            if accuracy > champ_acc:
                client.set_registered_model_alias(settings.mlflow.model_name, "champion", mv.version)
        except mlflow.exceptions.MlflowException:
            client.set_registered_model_alias(settings.mlflow.model_name, "champion", mv.version)

        return model_uri


def load_champion():
    """Aktuelles Champion-Modell laden."""
    return mlflow.pyfunc.load_model(f"models:/{settings.mlflow.model_name}@champion")
```

- `dagshub.init()` setzt URI + Auth automatisch — lokal über `.env`, in CI über Env-Vars
- Artefakte auf DagsHub Storage (20 GB free, kein S3 nötig)
- `@champion` Alias statt deprecated `transition_model_version_stage`

## Daten-Architektur: DagsHub + Parquet (keine DB nötig)

### TL;DR

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DagsHub (20 GB Free)                             │
│                                                                         │
│   DVC Storage         MLflow Server          Git Repo Mirror            │
│   ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐     │
│   │ raw/*.parquet│   │ Experiments      │   │ dvc.yaml / .lock   │     │
│   │ features/*   │   │ Runs + Metrics   │   │ params.yaml        │     │
│   │ predictions/*│   │ Model Registry   │   │ metrics.json       │     │
│   │ models/*     │   │ Artefakte (Model)│   │ *.dvc Pointer      │     │
│   └──────────────┘   └──────────────────┘   └────────────────────┘     │
│         ▲                     ▲                       ▲                  │
│         │ dvc push            │ dagshub.init()        │ git push        │
└─────────┼─────────────────────┼───────────────────────┼─────────────────┘
          │                     │                       │
┌─────────┼─────────────────────┼───────────────────────┼─────────────────┐
│ LOKAL   │                     │                       │                  │
│         ▼                     ▼                       ▼                  │
│   backend/data/          MLflow Client           Git Repo               │
│   ├── raw/*.parquet      (loggt nach DagsHub)    (Code + .dvc Pointer)  │
│   ├── features/*.parquet                                                │
│   ├── predictions/*.parquet                                             │
│   └── models/*.joblib                                                   │
│         ▲                                                               │
│         │ liest direkt                                                  │
│   ┌─────┴───────────┐                                                  │
│   │ DuckDB (in-mem) │  ← Query-Engine, KEINE Datenbank                 │
│   │ SELECT * FROM   │     kein Server, kein Prozess, keine .db-Datei   │
│   │ 'file.parquet'  │     nur import duckdb + SQL auf Parquet-Files     │
│   └─────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Brauche ich eine Datenbank?

**Nein.** Alle Daten liegen als **Parquet-Files** auf der Platte und werden via **DVC** nach **DagsHub** versioniert. Das ersetzt eine klassische DB komplett:

| Frage | Antwort |
|---|---|
| **Wo liegen die Daten?** | Lokal als Parquet in `backend/data/`, remote auf DagsHub (DVC) |
| **Wie lese ich die Daten?** | `duckdb.sql("SELECT * FROM 'file.parquet'")` oder `pd.read_parquet()` |
| **Wie versioniere ich?** | `dvc push` → DagsHub (20 GB free), `git commit` für `.dvc` Pointer |
| **Brauche ich DuckDB als Server?** | Nein — DuckDB läuft **in-process**, kein Docker/Port/Daemon |
| **Brauche ich PostgreSQL/SQLite?** | Nein — Parquet + DuckDB reicht für analytische Workloads |
| **Was ist mit dem Frontend?** | `dagshub.streaming.install_hooks()` → liest Parquet remote, kein `dvc pull` nötig |

### Warum Parquet statt einer echten DB?

1. **DagsHub kann Parquet** — Tabellen-Preview + Diffs direkt im Browser (DuckDB/.db-Files nicht)
2. **DVC-effizient** — spaltenbasiert, stabile Deltas, kleine `.dvc` Pointer in Git
3. **Schema embedded** — Typen (float64, datetime, int) im Parquet-Header, kein Casting nötig
4. **Kompakt** — ~10× kleiner als CSV, ~2× kleiner als DuckDB-Files
5. **Zero Infra** — kein Server, kein Port, kein Docker für die DB

### Was macht DuckDB dann genau?

DuckDB ist **keine Datenbank** in diesem Projekt — es ist eine **Query-Engine** die Parquet-Files per SQL liest:

```python
import duckdb

# Das ist alles — kein Server, kein Setup, kein connect()
df = duckdb.sql("SELECT * FROM 'backend/data/features/sp500_features.parquet'").df()

# Filtern ohne den ganzen DataFrame in RAM zu laden:
recent = duckdb.sql("""
    SELECT * FROM 'backend/data/predictions/sp500_predictions.parquet'
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
""").df()
```

**Alernative ohne DuckDB** (geht auch, ist nur langsamer bei großen Files):
```python
import pandas as pd
df = pd.read_parquet("backend/data/features/sp500_features.parquet")
```

### Kompletter Datenfluss: Wer speichert was wo?

```
1. FETCH (GH Actions / lokal)
   ├─ yfinance → OHLCV DataFrame
   ├─ pandas → df.to_parquet("backend/data/raw/sp500_ohlcv.parquet")
   ├─ calc_indicators() + extract_fundamentals()
   ├─ pandas → df.to_parquet("backend/data/features/sp500_features.parquet")
   └─ dvc push → DagsHub (remote Backup + Versionierung)

2. TRAIN (GH Actions / lokal)
   ├─ duckdb.sql("SELECT * FROM 'backend/data/features/...'") → DataFrame
   ├─ ExtraTreeTrainer.train(df) → model + accuracy
   ├─ mlflow.sklearn.log_model(model) → DagsHub MLflow (Artefakt)
   ├─ joblib.dump(model, "backend/data/models/model.joblib")  ← Fallback
   └─ metrics.json → git commit (DagsHub zeigt Metrics-History)

3. PREDICT (GH Actions / lokal)
   ├─ duckdb.sql("SELECT * FROM 'backend/data/features/...'") → DataFrame
   ├─ mlflow.pyfunc.load_model("models:/...@champion") → predict
   ├─ pandas → df.to_parquet("backend/data/predictions/sp500_predictions.parquet")
   └─ dvc push → DagsHub

4. FRONTEND (Streamlit Cloud)
   ├─ dagshub.streaming.install_hooks()  ← Magic: DVC-Files remote lesen
   ├─ duckdb.sql("SELECT * FROM 'backend/data/predictions/...'") → df
   └─ Plotly Charts → User sieht Predictions
```

### DataStore Interface (Wrapper um Parquet I/O)

Der `DataStore` ist kein DB-Client — er kapselt nur `read_parquet()` / `to_parquet()` + DuckDB-Queries:

```python
# backend/infra/database/base.py
from abc import ABC, abstractmethod
import pandas as pd


class DataStore(ABC):
    """Wrapper um Parquet I/O — kein DB-Server nötig."""

    @abstractmethod
    def save_features(self, df: pd.DataFrame, model_name: str, accuracy: float) -> None: ...

    @abstractmethod
    def save_raw(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def save_prediction(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_features(self) -> pd.DataFrame: ...

    @abstractmethod
    def load_predictions(self, days: int = 30) -> pd.DataFrame: ...

    @abstractmethod
    def load_raw(self) -> pd.DataFrame: ...
```

### Implementierung: DuckDBStore

```python
# backend/infra/database/duckdb_store.py
from pathlib import Path
import duckdb
import pandas as pd
from backend.infra.database.base import DataStore


class DuckDBStore(DataStore):
    """Parquet I/O mit DuckDB als Query-Engine — kein DB-Server, kein Prozess."""

    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "raw" / "sp500_ohlcv.parquet"
        self.features_path = self.data_dir / "features" / "sp500_features.parquet"
        self.predictions_path = self.data_dir / "predictions" / "sp500_predictions.parquet"

    def save_raw(self, df: pd.DataFrame) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.raw_path)

    def save_features(self, df: pd.DataFrame, model_name: str, accuracy: float) -> None:
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.features_path)

    def save_prediction(self, df: pd.DataFrame) -> None:
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.predictions_path)

    def load_raw(self) -> pd.DataFrame:
        return duckdb.sql(f"SELECT * FROM '{self.raw_path}'").df()

    def load_features(self) -> pd.DataFrame:
        return duckdb.sql(f"SELECT * FROM '{self.features_path}'").df()

    def load_predictions(self, days: int = 30) -> pd.DataFrame:
        return duckdb.sql(f"""
            SELECT * FROM '{self.predictions_path}'
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
        """).df()
```

```python
# backend/infra/database/__init__.py
from backend.core.config import settings
from backend.infra.database.duckdb_store import DuckDBStore

def get_data_store() -> DuckDBStore:
    return DuckDBStore(settings.database.data_dir)
```

### Docker Compose (nur MLflow lokal — optional)

```yaml
# docker-compose.yml — nur für lokales MLflow-UI, NICHT für Daten-Storage
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5001:5000"]
    volumes:
      - ./backend/data/mlflow:/mlflow
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
```

> **Hinweis:** Im Normalfall nutzt du den **DagsHub MLflow-Server** (kein Docker nötig). Docker Compose ist nur ein optionaler lokaler Fallback.

---

## Pydantic Settings: Zentrale Config

### `backend/core/config.py`

```python
# backend/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, SecretStr

class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")
    
    tracking_uri: str = "backend/data/mlflow/mlruns"  # Local fallback — dagshub.init() setzt DagsHub-URI
    tracking_username: str = ""                      # von dagshub.init() gesetzt
    tracking_password: SecretStr = SecretStr("")     # von dagshub.init() gesetzt
    experiment_name: str = "sp500_prediction"
    model_name: str = "best_ExtraTreesClassifier_model"

class DagsHubSettings(BaseSettings):
    """DagsHub-Anbindung für MLflow, DVC und Direct Data Access"""
    model_config = SettingsConfigDict(env_prefix="DAGSHUB_")

    repo_owner: str = ""            # DagsHub Username
    repo_name: str = "stock-prediction"
    token: SecretStr = SecretStr("")  # Access Token (https://dagshub.com/user/settings/tokens)

class StockSettings(BaseSettings):
    """Zentrale Konfiguration für Stock-Parameter"""
    symbol: str = "^GSPC"
    symbol_display: str = "S&P500"
    start_date: str = "2000-08-01"
    prediction_horizon_days: int = 15

class TechnicalIndicatorSettings(BaseSettings):
    """Konfiguration der technischen Indikatoren (berechnet aus OHLCV via ta-Library)"""
    # Konkrete Spalten wie sie im DataFrame stehen — müssen mit calc_indicators() übereinstimmen
    enabled: list[str] = [
        # Trend
        "SMA 10", "EMA 10", "EMA 20", "WMA 10", "Momentum 10", "SAR",
        # Momentum
        "RSI", "ROC", "%R",
        # Volume
        "OBV",
        # MACD
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        # Weitere
        "CCI", "ADOSC", "%K", "%D",
        # DMI / ADX
        "+DMI", "-DMI", "ADX",
        # Bollinger Bands
        "up_band", "mid_band", "low_band",
    ]

class FundamentalIndicatorSettings(BaseSettings):
    """Externe Marktdaten (yfinance) — jeder Ticker wird als '{ticker} Close'-Spalte geladen"""
    # Rohstoffe
    commodities: list[str] = ["GC=F", "CL=F", "SI=F", "HG=F", "NG=F"]
    # Devisen
    currencies: list[str] = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CNY=X"]
    # Internationale Indizes
    indices: list[str] = [
        "^IXIC", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^FCHI",
        "^N225", "^HSI", "^BSESN", "^MXX", "^AXJO", "^IBEX",
    ]
    # Anleihen
    bonds: list[str] = ["^TNX", "^IRX", "^FVX", "^TYX"]
    # ETFs / Benchmarks
    etfs: list[str] = ["SPY", "EFA"]

    @property
    def all_tickers(self) -> list[str]:
        """Alle externen Ticker als flache Liste"""
        return self.commodities + self.currencies + self.indices + self.bonds + self.etfs

    @property
    def all_columns(self) -> list[str]:
        """Konkrete DataFrame-Spaltennamen wie sie im Feature-DF stehen"""
        return [f"{t} Close" for t in self.all_tickers]

class FeatureSettings(BaseSettings):
    """Zentrale Feature-Konfiguration — bündelt technische + fundamentale Indikatoren"""
    technical: TechnicalIndicatorSettings = TechnicalIndicatorSettings()
    fundamental: FundamentalIndicatorSettings = FundamentalIndicatorSettings()

    @property
    def expected_feature_count(self) -> int:
        """Erwartete Gesamtzahl an Feature-Spalten (ohne Target)"""
        base_features = 2  # Close, Volume
        return base_features + len(self.technical.enabled) + len(self.fundamental.all_tickers)

class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    data_dir: str = "backend/data"     # Parquet-Root: backend/data/raw/, features/, predictions/

class TrainingParams(BaseModel):
    """Validiert params.yaml Training-Sektion beim Pipeline-Start"""
    n_trials: int = Field(gt=0, le=500, default=50)
    test_size: float = Field(gt=0.0, lt=1.0, default=0.20)
    random_state: int = 42

class PipelineParams(BaseModel):
    """Validiert params.yaml komplett — Tippfehler (n_trals statt n_trials) fliegen sofort auf"""
    stock: StockSettings
    training: TrainingParams

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    database: DatabaseSettings = DatabaseSettings()
    dagshub: DagsHubSettings = DagsHubSettings()
    mlflow: MLflowSettings = MLflowSettings()
    stock: StockSettings = StockSettings()
    features: FeatureSettings = FeatureSettings()

# Singleton
settings = Settings()
```

**Nutzung:**

```python
from backend.core.config import settings

settings.dagshub.repo_owner                  # DagsHub Username
settings.mlflow.model_name                   # "best_ExtraTreesClassifier_model"
settings.stock.symbol                        # "^GSPC"
settings.features.technical.enabled          # Liste der aktiven Indikatoren
```

**params.yaml validieren** (am Anfang jedes Workflows):

```python
import yaml
from backend.core.config import PipelineParams

params = PipelineParams.model_validate(yaml.safe_load(open("params.yaml")))
params.training.n_trials    # → int, validiert: 1–500
params.stock.symbol          # → str, aus params.yaml
```

### `backend/core/schemas.py` — Pydantic Models für Datengrenzen

Gezielter Pydantic-Einsatz an den 3 Stellen wo Daten zwischen Komponenten/Dateien fließen:

```python
# backend/core/schemas.py
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class TrainingResult(BaseModel):
    """Bündelt alles was ein Training produziert.
    Statt 6 lose Argumente an MLflow, metrics.json und DataStore weiterzureichen."""
    model_name: str
    accuracy: float = Field(ge=0.0, le=1.0)
    params: dict[str, Any]
    features_hash: str
    trained_at: datetime = Field(default_factory=datetime.now)
    n_features: int = Field(gt=0)


class PredictionRecord(BaseModel):
    """Einzelne Prediction — validiert dass target nur 0 oder 1 sein kann."""
    date: datetime
    target: Literal[0, 1]           # fängt ab wenn Modell float statt int liefert
    model_name: str
    accuracy: float = Field(ge=0.0, le=1.0)
    close: float = Field(gt=0)


class PipelineMetrics(BaseModel):
    """Schema für backend/data/metrics.json — git-tracked, in DagsHub UI sichtbar.
    Definiertes Format statt loses dict."""
    accuracy: float = Field(ge=0.0, le=1.0)
    model_name: str
    n_features: int
    n_training_samples: int
    features_hash: str
    trained_at: datetime | None = None
    predicted_at: datetime | None = None
```

**Nutzung in train.py:**

```python
from backend.core.schemas import TrainingResult, PipelineMetrics
from pathlib import Path

# Training liefert ein TrainingResult
result = TrainingResult(
    model_name="ExtraTreesClassifier", accuracy=0.85,
    params=best_params, features_hash=manifest.hash,
    n_features=manifest.count,
)

# 1x loggen → 3 Ziele
mlflow.log_metrics({"accuracy": result.accuracy})
mlflow.log_params(result.params)

# metrics.json (DVC/DagsHub) — Schema validiert
metrics = PipelineMetrics(
    accuracy=result.accuracy, model_name=result.model_name,
    n_features=result.n_features, n_training_samples=len(X_train),
    features_hash=result.features_hash, trained_at=result.trained_at,
)
Path("backend/data/metrics.json").write_text(metrics.model_dump_json(indent=2))
```

---

## Abstraktionen & Code-Verbesserungen

### 1. Feature-Manifest: Config ↔ DB synchron halten

Manifest (Spalten + Hash) wird beim Preprocessing erzeugt, beim Laden validiert. Pydantic gibt JSON-Serialisierung + Validierung gratis:

```python
# backend/core/features/manifest.py
import hashlib
from pydantic import BaseModel, field_validator, computed_field

class FeatureManifest(BaseModel):
    """Spalten-Konsistenz zwischen Training und Prediction sicherstellen.
    Pydantic statt dataclass: automatisches JSON + Validierung."""
    columns: list[str]

    @field_validator("columns")
    @classmethod
    def not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Feature-Liste darf nicht leer sein")
        return sorted(v)  # sortiert → stabiler Hash

    @computed_field
    @property
    def hash(self) -> str:
        return hashlib.sha256("|".join(self.columns).encode()).hexdigest()[:12]

    @computed_field
    @property
    def count(self) -> int:
        return len(self.columns)

    def validate_against(self, df_columns: list[str]) -> None:
        """Prüft ob ein DataFrame die erwarteten Spalten hat"""
        expected = set(self.columns)
        actual = set(df_columns)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(f"Feature mismatch! Missing: {missing}, Unexpected: {extra}")

# Workflow:
manifest = FeatureManifest(columns=df.columns.tolist())
manifest.model_dump_json()                 # → in Parquet-Metadata speichern
FeatureManifest.model_validate_json(s)     # → beim Laden validieren + Hash prüfen
```

**Flow:**

```
Preprocessing                  DataStore (Parquet)          Predict
─────────────                  ──────────────────           ───────
get_data()
  ├─ yfinance → raw df
  ├─ db.save_raw(df)       ────────→ backend/data/raw/sp500_ohlcv.parquet
  ├─ calc_indicators()
  ├─ extract_yahoo_data()
  ├─ manifest = FeatureManifest(df.columns)   
  ├─ db.save_features(df, ...) ───→ backend/data/features/sp500_features.parquet
  │                                                         │
  │                                  db.load_features() ←───┘
  │                                    ├─ Parquet → DataFrame
  │                                    ├─ manifest.validate_against(df)
  │                                    └─ predict(df) ✓
  │                                    └─ db.save_prediction(df)
  │                                       → backend/data/predictions/sp500_predictions.parquet
```

Ändert jemand die Feature-Liste in der Config, schlägt `validate_against()` sofort an — statt stilles Dimension-Mismatch im Model.

### 2. Ticker-Liste & Indikator-Liste über `FeatureSettings` steuerbar

→ Siehe vollständige Implementierung im Abschnitt [Feature Engineering: `ta`-Library](#feature-engineering-ta-library).

### 3. BaseTrainer ABC + ExtraTreeTrainer

```python
# backend/ml/training/base.py
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BaseTrainer(ABC):
    """Basis-Klasse für Model-Training — Split, Train, Evaluate."""

    TUNING_SPACE: dict = {}  # Überschrieben in Subklassen

    @abstractmethod
    def create_model(self, **params):
        ...

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
        X = df.drop(columns=["Target"])
        y = df["Target"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42, **params):
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size, random_state)
        model = self.create_model(**params)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy, X_train
```

```python
# backend/ml/training/extra_tree.py
from sklearn.ensemble import ExtraTreesClassifier
from backend.ml.training.base import BaseTrainer


class ExtraTreeTrainer(BaseTrainer):
    DEFAULT_PARAMS = {"n_estimators": 2000, "random_state": 42}
    TUNING_SPACE = {
        "n_estimators": (100, 3000),
        "max_depth": (10, 200),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }

    def create_model(self, **params):
        return ExtraTreesClassifier(**{**self.DEFAULT_PARAMS, **params})
```

Nutzung im Workflow:

```python
# backend/workflows/train.py
params = PipelineParams.model_validate(yaml.safe_load(open("params.yaml")))
trainer = ExtraTreeTrainer()
model, accuracy, X_train = trainer.train(
    df,
    test_size=params.training.test_size,
    random_state=params.training.random_state,
)
log_and_promote(model, accuracy, X_train)
```

### 4. Predict-Abstraktion

```python
# backend/core/prediction/base.py
from abc import ABC, abstractmethod
import pandas as pd
from backend.core.config import settings
from backend.core.schemas import PredictionRecord
from backend.infra.database.base import DataStore


class PredictionService(ABC):
    @abstractmethod
    def load_model(self):
        ...

    def predict(self, features: pd.DataFrame) -> list:
        return self.load_model().predict(features)

    def run(self, db: DataStore) -> list[PredictionRecord]:
        """Daten laden → predicten → validieren → speichern."""
        features_df = db.load_features()
        preds = self.predict(features_df)
        records = [
            PredictionRecord(
                date=idx, target=int(p),
                model_name=settings.mlflow.model_name,
                accuracy=0.0, close=float(features_df.loc[idx, "Close"]),
            )
            for idx, p in zip(features_df.index, preds)
        ]
        db.save_prediction(pd.DataFrame([r.model_dump() for r in records]))
        return records


class MLflowPredictionService(PredictionService):
    def load_model(self):
        from backend.ml.registry.model_registry import load_champion
        return load_champion()
```

---

## Feature Engineering: `ta`-Library

[`ta`](https://github.com/bukosabino/ta) — reine Python-Library (Pandas + NumPy), keine C-Library nötig:

```bash
pip install ta
```

### `backend/core/features/indicators.py`

```python
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator, MACD, CCIIndicator, ADXIndicator, PSARIndicator
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    time_period = 10
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend
    df[f"SMA {time_period}"] = SMAIndicator(close, window=time_period).sma_indicator()
    df[f"EMA {time_period}"] = EMAIndicator(close, window=time_period).ema_indicator()
    df[f"EMA {20}"] = EMAIndicator(close, window=20).ema_indicator()
    df[f"WMA {time_period}"] = WMAIndicator(close, window=time_period).wma()
    df[f"Momentum {time_period}"] = close - close.shift(time_period)
    df["SAR"] = PSARIndicator(high, low, close).psar()

    # Momentum
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["ROC"] = ROCIndicator(close, window=10).roc()
    df["%R"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # Volume
    df["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    # Weitere
    df["CCI"] = CCIIndicator(high, low, close, window=14).cci()
    df["ADOSC"] = AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
    df["%K"] = (df["Close"] - df["Low"]) * 100 / (df["High"] - df["Low"])
    df["%D"] = df["%K"].rolling(3).mean()

    # DMI / ADX
    adx = ADXIndicator(high, low, close, window=14)
    df["+DMI"] = adx.adx_pos()
    df["-DMI"] = adx.adx_neg()
    df["ADX"] = adx.adx()

    # Bollinger Bands
    bb = BollingerBands(close, window=20)
    df["up_band"] = bb.bollinger_hband()
    df["mid_band"] = bb.bollinger_mavg()
    df["low_band"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    df.drop(["High", "Low", "Adj Close", "Open"], axis=1, inplace=True, errors="ignore")
    return df
```

---

## Tech Stack, Prozesse & Cloud (Stand März 2026)

Gesamtkosten: **$0/Monat** — alles im Free Tier.

### DVC Pipeline: Reproduzierbare ML-Workflows

Statt losen Makefile-Targets definiert `dvc.yaml` eine DAG mit Abhängigkeiten. DVC führt nur Stages aus, deren Inputs sich geändert haben:

```yaml
# dvc.yaml
stages:
  fetch:
    cmd: python -m backend.workflows.fetch_data
    deps:
      - backend/workflows/fetch_data.py
      - backend/core/features/
    params:
      - stock                       # aus params.yaml: symbol, start_date
      - features                    # aus params.yaml: technische + fundamentale Indikatoren
    outs:
      - backend/data/raw/sp500_ohlcv.parquet:
          persist: true
      - backend/data/features/sp500_features.parquet:
          persist: true

  train:
    cmd: python -m backend.workflows.train
    deps:
      - backend/data/features/sp500_features.parquet
      - backend/ml/training/
    params:
      - training                    # n_trials, test_size, random_state
    outs:
      - backend/data/models/:
          persist: true
    metrics:
      - backend/data/metrics.json:
          cache: false              # git-tracked, in DagsHub UI sichtbar

  predict:
    cmd: python -m backend.workflows.predict
    deps:
      - backend/data/features/sp500_features.parquet
      - backend/data/models/
      - backend/workflows/predict.py
    outs:
      - backend/data/predictions/sp500_predictions.parquet:
          persist: true
    metrics:
      - backend/data/metrics.json:
          cache: false
```

```yaml
# params.yaml — zentrale Pipeline-Parameter (änderbar ohne Code-Änderung)
stock:
  symbol: "^GSPC"
  start_date: "2000-08-01"
  prediction_horizon_days: 15

features:
  technical:
    - RSI
    - MACD
    - ADX
    # ... (vollständige Liste in config.py)
  fundamental_tickers:
    - GC=F
    - CL=F
    # ... (vollständige Liste in config.py)

training:
  n_trials: 50
  test_size: 0.20
  random_state: 42
```

**Pipeline-DAG (DagsHub zeigt das als Graph):**
```
fetch ──→ train ──→ predict
  │                   ▲
  └───────────────────┘
       (Parquet-Files)
```

**Nutzung:**
```bash
# Lokal: komplette Pipeline
dvc repro                      # führt nur geänderte Stages aus

# Nur Fetch + Predict (ohne Training)
dvc repro predict              # skipped train wenn nichts geändert

# In GH Actions:
dvc repro && dvc push          # Pipeline + Ergebnisse nach DagsHub syncen
```

**DVC-Vorteile:** Caching (nur geänderte Stages laufen), Reproduzierbarkeit (`dvc.lock`), Parameter-Tracking (`params.yaml`), DAG-Visualisierung in DagsHub.

### Data Versioning mit DVC

```bash
# Einmalig: DagsHub als DVC Remote einrichten
dvc remote add dagshub https://dagshub.com/<user>/stock-prediction.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <user>
dvc remote modify dagshub --local password <token>

# Daten tracken
dvc add backend/data/raw/sp500_ohlcv.parquet
dvc add backend/data/features/sp500_features.parquet
dvc add backend/data/predictions/sp500_predictions.parquet
git add backend/data/**/*.dvc .gitignore
git commit -m "data: training data 2026-03-08"
dvc push
```

Reproduzierbarer Zustand:
```bash
git checkout v1.0 && dvc checkout   # Code + Daten von v1.0
```

### DagsHub Direct Data Access (Frontend)

`dagshub.streaming` ermöglicht transparenten Remote-Zugriff auf DVC-tracked Files — kein `dvc pull` nötig:

```python
# frontend/data_loader.py
import duckdb
from dagshub.streaming import install_hooks

install_hooks()  # DVC-Dateien werden on-demand von DagsHub geladen


def load_predictions():
    return duckdb.sql(
        "SELECT * FROM 'backend/data/predictions/sp500_predictions.parquet'"
    ).df()


def load_features():
    return duckdb.sql(
        "SELECT * FROM 'backend/data/features/sp500_features.parquet'"
    ).df()
```

### Prozess → Tool → Alternativen

| Prozess | Frequenz | Tool (empfohlen) | Alternative |
|---|---|---|---|
| Data Fetching | Täglich | **yfinance 1.2.0** + GH Actions Cron | – |
| Feature Engineering | Täglich | **pandas 3.0.1** + **`ta` 0.11.0** | polars · pandas-ta · TA-Lib |
| Data Storage | Täglich | **DuckDB 1.4.4** (Query-Engine) + **Parquet** (Storage) | PostgreSQL/TimescaleDB · InfluxDB |
| Data Versioning | Bei Änderung | **DVC 3.x** + **DagsHub** (20 GB) | Git LFS · Cloudflare R2 |
| Pipeline/DAG | Täglich/Wöch. | **DVC Pipelines** (`dvc.yaml`) | Makefile · Airflow (Overkill) |
| Model Training | Wöchentlich | **scikit-learn 1.8.0** (ExtraTrees) | **LightGBM 4.6.0** · XGBoost 3.2.0 |
| HP Tuning | Wöchentlich | **Optuna 4.7.0** (~2 MB) | Ray Tune (~200 MB, Overkill) |
| Experiment Tracking | Jedes Training | **MLflow 3.10.1** via **DagsHub** | W&B |
| Model Registry | Nach Training | **MLflow Model Registry** (DagsHub) | joblib + YAML Fallback |
| Model Serving | Täglich | **mlflow.pyfunc** im GH Action | Ray Serve (Overkill) |
| Frontend | Dauerhaft | **Streamlit 1.55.0** + **Plotly 6.6.0** | HuggingFace Spaces |
| Scheduling | Täglich/Wöch. | **GH Actions** `cron` | – |
| CI/CD | Bei Push | **GH Actions** (Tests, Linting) | – |
| Artifact Storage | Nach Run | **DagsHub** (20 GB, MLflow-Artefakte) | Cloudflare R2 (10 GB, S3-kompatibel) |
| Monitoring & Alerts | Täglich | **Zapier Free** (100 Tasks/Mo) | GH Actions Email |
| Config | Statisch | **pydantic-settings 2.13.1** + **pydantic 2.x** | – |
| Daten-Validierung | An Grenzen | **Pydantic Models** (schemas.py) | dataclasses (kein JSON) |
| Linting/Format | Bei Push | **ruff 0.15.5** | – |

### Cloud-Infrastruktur: Free-Tier Optionen

| Dienst | Free Tier | Wofür |
|---|---|---|
| **GitHub Actions** | 2000 Min/Mo, 4 vCPU, 16 GB | Compute (Fetch, Train, Predict) |
| **DagsHub** | 20 GB Storage, 100 Experiments, gehostetes MLflow | MLflow-UI + Experiment Tracking + Artefakte + DVC Storage + DAG-Visualisierung + Data Versioning |
| **Streamlit Community Cloud** | Unbegrenzt (public repos) | Frontend-Hosting |
| **HuggingFace Spaces** | CPU Basic (2 vCPU, 16 GB) | Alternative Frontend-Hosting |
| **Cloudflare R2** | 10 GB, 0 Egress, S3-kompatibel | Alternative Storage (falls DagsHub nicht reicht) |
| **Zapier Free** | 100 Tasks/Mo, Two-Step Zaps | Slack/Telegram Alerts (~33 Tasks/Mo) |

### GH Actions Workflow

```yaml
# .github/workflows/daily_pipeline.yml
name: Daily Fetch & Predict
on:
  schedule:
    - cron: '30 21 * * 1-5'  # Mo-Fr 21:30 UTC (nach US-Börsenschluss)
  workflow_dispatch:

jobs:
  fetch-and-predict:
    runs-on: ubuntu-latest
    env:
      DAGSHUB_REPO_OWNER: ${{ secrets.DAGSHUB_USERNAME }}
      DAGSHUB_REPO_NAME: stock-prediction
      DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
          cache: pip
      - run: pip install -e ".[dev]"

      - name: Configure DVC
        run: |
          dvc remote modify dagshub --local auth basic
          dvc remote modify dagshub --local user $DAGSHUB_REPO_OWNER
          dvc remote modify dagshub --local password $DAGSHUB_TOKEN

      - name: Pull data
        run: dvc pull || true

      - name: Run pipeline
        run: dvc repro predict        # fetch + predict (skipped train)

      - name: Push results
        run: dvc push

      - name: Zapier Notification
        if: always()
        run: |
          curl -s -X POST "${{ secrets.ZAPIER_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '{"status":"${{ job.status }}","run":"${{ github.run_id }}"}'
```
### pyproject.toml

```bash
python3.13 -m venv .venv-stock-pred-py313 && source .venv-stock-pred-py313/bin/activate
pip install -e ".[frontend,dev]"
```

```toml
[project]
name = "stock-prediction"
requires-python = ">=3.13"

dependencies = [
    "pandas>=3.0.1",
    "numpy>=2.4.2",
    "yfinance>=1.2.0",
    "ta>=0.11.0",
    "scikit-learn>=1.8.0",
    "lightgbm>=4.6.0",
    "mlflow>=3.10.1",
    "dagshub>=0.5.0",
    "optuna>=4.7.0",
    "duckdb>=1.4.4",
    "pyarrow>=19.0.1",
    "pydantic-settings>=2.13.1",
    "python-dotenv>=1.2.2",
]

[project.optional-dependencies]
frontend = ["streamlit>=1.55.0", "plotly>=6.6.0"]
dev = ["pytest>=9.0.2", "ruff>=0.15.5", "optuna-dashboard>=0.20.0", "dvc>=3.50"]

[tool.ruff]
target-version = "py313"
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

