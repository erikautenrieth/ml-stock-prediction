from abc import ABC, abstractmethod

import pandas as pd


class DataStore(ABC):
    @abstractmethod
    def save_raw(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def save_features(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def save_predictions(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_raw(self) -> pd.DataFrame: ...

    @abstractmethod
    def load_features(self) -> pd.DataFrame: ...

    @abstractmethod
    def load_predictions(self, days: int = 30) -> pd.DataFrame: ...
