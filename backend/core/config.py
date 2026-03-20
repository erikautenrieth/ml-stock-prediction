from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StockSettings(BaseSettings):
    """Identity & data-fetching settings for the target stock."""

    symbol: str = "^GSPC"
    symbol_display: str = "S&P500"
    start_date: str = "2000-08-01"
    prediction_horizon_days: int = 10
    extra_tickers: list[str] = [
        # Commodities
        "GC=F",  # Gold
        "CL=F",  # Crude Oil
        # SI=F (Silver) dropped — ρ ≈ 0.8 with Gold
        "HG=F",  # Copper
        "NG=F",  # Natural Gas
        # Currencies
        "EURUSD=X",
        "GBPUSD=X",
        "JPY=X",
        "CNY=X",
        # US indices
        "^IXIC",  # Nasdaq (tech tilt)
        # ^DJI dropped — ρ > 0.95 with ^GSPC, ^IXIC
        "^RUT",  # Russell 2000 (small-cap divergence)
        # SPY dropped — ρ ≈ 0.99 with ^GSPC (target)
        "EFA",  # Intl Developed ETF
        # Global indices
        "^FTSE",  # FTSE 100
        "^GDAXI",  # DAX
        # ^FCHI (CAC 40) dropped — ρ > 0.9 with ^GDAXI
        "^N225",  # Nikkei 225
        "^HSI",  # Hang Seng
        "^BSESN",  # BSE Sensex
        "^MXX",  # IPC Mexico
        "^AXJO",  # ASX 200
        # ^IBEX dropped — ρ > 0.9 with ^GDAXI, ^FTSE
        # Volatility & rates
        "^VIX",  # CBOE VIX — strongest S&P500 direction predictor
        "^TNX",  # 10Y Treasury
        "^IRX",  # 3M Treasury
        # ^FVX (5Y) & ^TYX (30Y) dropped — ρ > 0.9 with ^TNX
    ]


class FeatureSettings(BaseSettings):
    """Feature engineering parameters."""

    indicator_window: int = 10
    target_threshold: float = 0.03  # dead zone: |return| <= threshold → drop row


class TrainingSettings(BaseSettings):
    """ML training & tuning hyperparameters."""

    training_years: int | None = 12  # rolling window: keep only last N years; None = all data
    sample_weight_halflife: float = 0.33  # exponential decay fraction


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


class DatabaseSettings(BaseSettings):
    data_dir: str = "backend/data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    stock: StockSettings = StockSettings()
    features: FeatureSettings = FeatureSettings()
    training: TrainingSettings = TrainingSettings()
    dagshub: DagsHubSettings = DagsHubSettings()
    mlflow: MLflowSettings = MLflowSettings()
    database: DatabaseSettings = DatabaseSettings()


settings = Settings()
