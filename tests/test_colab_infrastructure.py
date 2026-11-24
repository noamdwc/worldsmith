"""R2 storage contracts for the Colab infrastructure."""

import subprocess
import tomllib
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from botocore.response import StreamingBody

from worldsmith.colab import session, storage


@pytest.mark.parametrize("stderr, message", [
    ("[colab] Session 'test-runtime' appears to be lost (404/401).", "Colab session disappeared"),
    ("RuntimeError: Connection was lost.", "Colab exec lost the connection"),
    ("TimeoutError: Timeout waiting for reply", "Colab exec timed out waiting for a reply"),
    ("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", "Set SSL_CERT_FILE"),
    ("Unrecognized CLI error", "Colab exec failed for test-runtime"),
])
def test_colab_cli_errors_report_safe_diagnostics(monkeypatch, stderr, message):
    monkeypatch.setattr(session.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, "private stdout", f"{stderr}\nprivate stderr",
    ))
    with pytest.raises(session.ColabError, match=message) as error:
        session.ColabSession("test-runtime").execute("print(1)", timeout=10)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("output, message", [
    ("ColabRequestError: Failed to issue request POST https://colab.research.google.com/tun/m/assign: Service Unavailable",
     "Service Unavailable \\(HTTP 503\\). Retry later"),
    ("Unrecognized CLI error", "See ~/.config/colab-cli/colab.log for the underlying error"),
])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_colab_allocation_errors_report_safe_diagnostics(monkeypatch, output, message, stream):
    streams = {"stdout": "private stdout", "stderr": "private stderr"}
    streams[stream] += f"\n{output}\nhttps://example.com/?token=private"
    monkeypatch.setattr(session.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, **streams,
    ))

    with pytest.raises(session.ColabError, match=message) as error:
        session.ColabSession("test-runtime").allocate()

    assert "private" not in str(error.value)
    assert "https://" not in str(error.value)
    assert "quota" not in str(error.value)


def test_research_mcp_forwards_ca_configuration():
    config = storage.ROOT / "autoresearch/.codex/config.toml"
    with config.open("rb") as source:
        environment = tomllib.load(source)["mcp_servers"]["worldsmith"]["env_vars"]
    assert {"SSL_CERT_FILE", "SSL_CERT_DIR", "WEBSOCKET_CLIENT_CA_BUNDLE"} <= set(environment)


def test_r2_download_streams_the_sdk_body(tmp_path):
    contents = b"evaluation artifacts"
    raw = BytesIO(contents)
    body = StreamingBody(raw, len(contents))
    bucket = storage.R2.__new__(storage.R2)
    bucket.bucket = "test-bucket"
    bucket.client = SimpleNamespace(get_object=lambda **kwargs: {"Body": body})
    destination = tmp_path / "artifact.tar.gz"

    bucket.download("output/artifacts.tar.gz", destination)

    assert destination.read_bytes() == contents
    assert raw.closed


def test_storage_signs_a_single_conditional_put_with_external_credentials(tmp_path, monkeypatch):
    credentials = tmp_path / "r2.env"
    credentials.write_text(
        f"R2_ACCOUNT_ID={'a' * 32}\nR2_ACCESS_KEY_ID=example-key\nR2_SECRET_ACCESS_KEY=example-secret\n",
    )
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.setenv("R2_CREDENTIALS_FILE", str(credentials))
    url = urlparse(storage.R2().signed_url(
        "worldsmith/evaluations/job/output/artifacts.tar.gz", expires=500, upload=True,
    ))
    query = parse_qs(url.query)
    assert url.hostname == f"{'a' * 32}.r2.cloudflarestorage.com"
    assert url.path == "/test-bucket/worldsmith/evaluations/job/output/artifacts.tar.gz"
    assert query["X-Amz-SignedHeaders"] == ["host;if-none-match"]
    assert query["X-Amz-Expires"] == ["500"]
    assert "example-secret" not in url.geturl()

    verified = urlparse(storage.R2().signed_url(
        "worldsmith/cache/wheels/pymunk/wheelhouse.tar",
        expires=500, upload=True, metadata={"worldsmith-verified": "1"},
    ))
    assert parse_qs(verified.query)["X-Amz-SignedHeaders"] == ["host;if-none-match;x-amz-meta-worldsmith-verified"]

    monkeypatch.setenv("R2_CREDENTIALS_FILE", str(storage.ROOT / "pyproject.toml"))
    with pytest.raises(storage.StorageError, match="must be outside"):
        storage.R2()
