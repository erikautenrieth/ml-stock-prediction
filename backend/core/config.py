from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class DagsHubSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DAGSHUB_", env_file=".env", extra="ignore")

    repo_owner: str = ""
    repo_name: str = ""
    token: SecretStr = SecretStr("")

    @property
    def repo_full(self) -> str:
        return f"{self.repo_owner}/{self.repo_name}"

    @property
    def tracking_uri(self) -> str:
        return f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow"

    @property
    def bucket_url(self) -> str:
        return f"https://dagshub.com/api/v1/repo-buckets/s3/{self.repo_owner}"


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", extra="ignore")

    tracking_uri: str = ""  # set dynamically from DagsHub or override via env
    tracking_username: str = ""
    tracking_password: SecretStr = SecretStr("")
    experiment_name: str = "sp500_prediction"
    model_name: str = "best_ExtraTreesClassifier_model"


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE_", env_file=".env", extra="ignore")

    dagshub_repo: str = ""
    dagshub_token: SecretStr = SecretStr("")


class DatabaseSettings(BaseSettings):
    data_dir: str = "backend/data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    stock: StockSettings = StockSettings()
    dagshub: DagsHubSettings = DagsHubSettings()
    mlflow: MLflowSettings = MLflowSettings()
    storage: StorageSettings = StorageSettings()
    database: DatabaseSettings = DatabaseSettings()


settings = Settings()
