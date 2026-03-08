from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class StockSettings(BaseSettings):
    symbol: str = "^GSPC"
    symbol_display: str = "S&P500"
    start_date: str = "2000-08-01"
    prediction_horizon_days: int = 10
    indicator_window: int = 10
    extra_tickers: list[str] = [
        "GC=F",
        "CL=F",
        "EURUSD=X",
        "GBPUSD=X",
        "JPY=X",
        "CNY=X",
        "^IXIC",
        "^DJI",
        "^RUT",
        "^FTSE",
        "^GDAXI",
        "^FCHI",
        "^N225",
        "^HSI",
        "^BSESN",
        "^MXX",
        "^AXJO",
        "^IBEX",
        "SI=F",
        "HG=F",
        "NG=F",
        "^TNX",
        "^IRX",
        "^FVX",
        "^TYX",
        "SPY",
        "EFA",
    ]


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = "backend/data/mlflow/mlruns"
    tracking_username: str = ""
    tracking_password: SecretStr = SecretStr("")
    experiment_name: str = "sp500_prediction"
    model_name: str = "best_ExtraTreesClassifier_model"


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    dagshub_repo: str = ""
    dagshub_token: SecretStr = SecretStr("")


class DatabaseSettings(BaseSettings):
    data_dir: str = "backend/data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    stock: StockSettings = StockSettings()
    mlflow: MLflowSettings = MLflowSettings()
    storage: StorageSettings = StorageSettings()
    database: DatabaseSettings = DatabaseSettings()


settings = Settings()
