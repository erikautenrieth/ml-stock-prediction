from pathlib import Path

import boto3
import structlog
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from backend.core.config import settings
from backend.infra.storage.base import RemoteStorage

logger = structlog.get_logger(__name__)


class DagsHubStorage(RemoteStorage):
    """S3-compatible storage backed by a DagsHub repo bucket."""

    def __init__(self) -> None:
        cfg = settings.dagshub
        token = cfg.token.get_secret_value()
        self.bucket = cfg.repo_name

        self._client = boto3.client(
            "s3",
            endpoint_url=cfg.bucket_url,
            aws_access_key_id=token,
            aws_secret_access_key=token,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        logger.info("dagshub_storage_initialized", bucket=self.bucket)

    def upload(self, local_path: Path, remote_key: str) -> None:
        self._client.upload_file(str(local_path), self.bucket, remote_key)
        logger.info("uploaded", local=str(local_path), remote=remote_key)

    def download(self, remote_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, remote_key, str(local_path))
        logger.info("downloaded", remote=remote_key, local=str(local_path))

    def exists(self, remote_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except ClientError:
            return False
