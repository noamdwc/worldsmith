"""Own one loopback-only vLLM process and its GPU workers inside Colab."""

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from .config import BASE_URL, server_command


class ModelServer:
    def __init__(self, *, python: str = sys.executable, log: Path = Path("vllm.log"), startup_timeout: float = 900):
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        self.python = python
        self.log = log.resolve()
        self.startup_timeout = startup_timeout
        self.process = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("This model server has already been started")
        # Never mistake an unrelated server for our child during readiness checks.
        with socket.socket() as port:
            port.bind(("127.0.0.1", 8000))
        with self.log.open("w") as log:
            self.process = subprocess.Popen(
                server_command(self.python), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        try:
            deadline = monotonic() + self.startup_timeout
            while monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with status {self.process.returncode}; see {self.log}")
                try:
                    with urlopen(BASE_URL.removesuffix("/v1") + "/health", timeout=2):
                        return self
                except (URLError, TimeoutError):
                    sleep(1)
            raise TimeoutError(f"vLLM did not become ready within {self.startup_timeout}s; see {self.log}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is None:
            return
        # vLLM creates engine workers. Signal the entire owned process group,
        # including when the API process has already exited.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # Reaping the API process does not imply every engine worker exited.
        # Remove any survivors after giving the server time for graceful cleanup.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=10)
        self.process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
