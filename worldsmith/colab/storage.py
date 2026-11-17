"""Local R2 access and short-lived, object-specific URLs for Colab."""

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


class ObjectMissing(StorageError):
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

    def exists(self, key: str) -> bool:
        with transfer("cache check"):
            try:
                self.client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"404", "NotFound", "NoSuchKey"}:
                    raise
                return False
        return True

    def put(self, key: str, source: Path, *, cache: bool = False):
        with transfer("upload"), source.open("rb") as body:
            try:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=body, IfNoneMatch="*")
            except ClientError as error:
                if not cache or error.response["Error"]["Code"] != "PreconditionFailed":
                    raise

    def signed_url(self, key: str, *, expires: int, upload: bool = False) -> str:
        params = {"Bucket": self.bucket, "Key": key}
        if upload:
            params["IfNoneMatch"] = "*"
        with transfer("URL signing"):
            return self.client.generate_presigned_url(
                "put_object" if upload else "get_object", Params=params, ExpiresIn=expires,
            )

    def download(self, key: str, destination: Path):
        with transfer("download"):
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=key)
            except ClientError as error:
                if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                    raise ObjectMissing("R2 has no artifact bundle for this job") from None
                raise
            with response["Body"] as body, destination.open("wb") as file:
                for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                    file.write(chunk)
