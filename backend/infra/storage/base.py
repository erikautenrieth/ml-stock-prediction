from abc import ABC, abstractmethod
from pathlib import Path


class RemoteStorage(ABC):
    """Abstract base for uploading/downloading files to remote storage."""

    @abstractmethod
    def upload(self, local_path: Path, remote_key: str) -> None: ...

    @abstractmethod
    def download(self, remote_key: str, local_path: Path) -> None: ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool: ...
