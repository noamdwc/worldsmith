"""Local R2 access. Colab gets a short-lived PUT URL for one job object."""

import os
import re
from contextlib import contextmanager
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from worldsmith.research_lifecycle import ROOT

PREFIX = "worldsmith/evaluations"


class StorageError(RuntimeError):
    pass


@contextmanager
def transfer(operation):
    try:
        yield
    except (BotoCoreError, ClientError, OSError, ValueError):
        # SDK errors and HTTP URLs may contain credential material.
        raise StorageError(f"R2 {operation} failed") from None


class R2:
    def __init__(self):
        bucket = os.environ.get("R2_BUCKET")
        credential_path = os.environ.get("R2_CREDENTIALS_FILE")
        if not bucket or not credential_path:
            raise StorageError("Set R2_BUCKET and R2_CREDENTIALS_FILE before starting Colab")
        path = Path(credential_path).expanduser().resolve()
        if path.is_relative_to(ROOT.resolve()):
            raise StorageError("R2_CREDENTIALS_FILE must be outside the Worldsmith repository")
        with transfer("credential read"):
            values = dict(
                line.split("=", 1) for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        account = values.get("R2_ACCOUNT_ID", "")
        access_key = values.get("R2_ACCESS_KEY_ID")
        secret_key = values.get("R2_SECRET_ACCESS_KEY")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", account) or not access_key or not secret_key:
            raise StorageError("Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in the credential file")
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4", connect_timeout=10, read_timeout=30,
                retries={"total_max_attempts": 1}, request_checksum_calculation="when_required",
            ),
        )

    def check(self):
        with transfer("bucket check"):
            self.client.head_bucket(Bucket=self.bucket)

    def put(self, key: str, source: Path):
        with transfer("upload"), source.open("rb") as body:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body, IfNoneMatch="*")

    def output_url(self, key: str, *, expires: int) -> str:
        with transfer("output signing"):
            return self.client.generate_presigned_url(
                "put_object", Params={"Bucket": self.bucket, "Key": key, "IfNoneMatch": "*"},
                ExpiresIn=expires,
            )

    def download(self, key: str, destination: Path):
        with transfer("download"):
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            with response["Body"] as body, destination.open("wb") as file:
                for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                    file.write(chunk)
