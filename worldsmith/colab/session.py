"""Manage the named Colab runtime for this checkout."""

import fcntl
import json
import re
import signal
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from worldsmith.research_lifecycle import ROOT

from .storage import R2

STATE_DIR = ROOT / "runs" / "colab"
REMOTE_PROJECT = "/content/worldsmith"
SETUP_TIMEOUT = 1800


class ColabError(RuntimeError):
    pass


@contextmanager
def session_lock(name: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / f"{name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ColabError(f"Another Colab {name} is in progress") from None
        yield


def tracked_session() -> dict | None:
    try:
        return json.loads((STATE_DIR / "session.json").read_text())
    except FileNotFoundError:
        return None


def save_session(name: str, *, ready: bool):
    temporary = STATE_DIR / "session.tmp"
    temporary.write_text(json.dumps({"name": name, "ready": ready}) + "\n")
    temporary.replace(STATE_DIR / "session.json")


class ColabSession:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def active(cls, expected: str | None = None):
        state = tracked_session()
        if state is None or not state["ready"] or (expected is not None and state["name"] != expected):
            raise ColabError("Colab session is unavailable. Start it from the owner workflow")
        return cls(state["name"])

    def command(self, action: str, *args: str, timeout: float = 60, code: str | None = None) -> str:
        try:
            result = subprocess.run(
                ["colab", action, "-s", self.name, *args],
                input=code if code is not None else "", text=True, capture_output=True, timeout=timeout,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise ColabError("Google Colab CLI is unavailable. Install google-colab-cli") from None
        except subprocess.TimeoutExpired:
            raise ColabError(f"Colab {action} timed out") from None
        # Avoid forwarding raw CLI logs, which may include OAuth or proxy tokens.
        if result.returncode:
            if f"Session '{self.name}' not found" in result.stdout or "appears to be lost" in result.stdout:
                raise ColabError("Colab session disappeared or is unavailable")
            if action == "new":
                raise ColabError("Colab L4 allocation failed. Check CLI authentication and L4 quota")
            raise ColabError(f"Colab {action} failed for {self.name}")
        return result.stdout

    def execute(self, code: str, *, timeout: int) -> str:
        return self.command("exec", "--timeout", str(timeout), timeout=timeout + 15, code=code)

    def allocate(self):
        # Let the CLI record the assignment before cleanup handles a signal.
        interrupted = None

        def remember(signum, frame):
            nonlocal interrupted
            interrupted = signum

        previous = {sig: signal.signal(sig, remember) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.command("new", "--gpu", "L4", timeout=180)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if interrupted is not None:
                raise SystemExit(128 + interrupted)

    def is_assigned(self) -> bool:
        return f"Session '{self.name}' not found" not in self.command("status", timeout=15)

    def stop(self):
        try:
            self.command("stop")
        except ColabError:
            # An expired runtime can still have a stale entry in the CLI store.
            if self.is_assigned():
                raise
        (STATE_DIR / "session.json").unlink(missing_ok=True)


def package_project(archive_path: Path):
    # Include new runtime source files from the working tree, without uploading
    # candidate code, caches, simulator assets, or local configuration.
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
         "worldsmith/*.py", "scripts/setup-colab.py", "pyproject.toml", "uv.lock",
         "simulation_environments/*/pyproject.toml", "simulation_environments/libero/setup.sh",
         "simulation_environments/libero/assets.sha256"],
        cwd=ROOT,
    ).decode().split("\0")
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        for name in paths:
            path = ROOT / name
            if name and path.is_file() and not path.is_symlink():
                archive.add(path, arcname=name, recursive=False)


def start_session() -> ColabSession:
    with session_lock("lifecycle"):
        if tracked_session() is not None:
            raise ColabError("A Colab session is already tracked. Inspect it with make colab-status")
        R2().check()
        session = ColabSession(f"worldsmith-{uuid4().hex[:12]}")
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "worldsmith.tar.gz"
            package_project(archive)
            save_session(session.name, ready=False)
            try:
                print(f"Starting Colab L4 session {session.name}", flush=True)
                session.allocate()
                session.command("upload", str(archive), "/content/worldsmith.tar.gz")
                output = session.execute(
                    "import runpy, tarfile\n"
                    "from pathlib import Path\n"
                    f"Path({REMOTE_PROJECT!r}).mkdir()\n"
                    "with tarfile.open('/content/worldsmith.tar.gz') as archive:\n"
                    f"    archive.extractall({REMOTE_PROJECT!r}, filter='data')\n"
                    f"runpy.run_path({(REMOTE_PROJECT + '/scripts/setup-colab.py')!r}, run_name='__main__')\n",
                    timeout=SETUP_TIMEOUT,
                )
                lines = output.splitlines()
                if "WORLDSMITH_COLAB_READY" not in lines:
                    stages = [line for line in lines if re.fullmatch(r"WORLDSMITH_SETUP [a-zA-Z0-9 ._()/-]+", line)]
                    stage = stages[-1].removeprefix("WORLDSMITH_SETUP ") if stages else "project extraction"
                    raise ColabError(f"Colab initialization failed at {stage}. See the local Colab CLI history")
                save_session(session.name, ready=True)
            except BaseException:
                stop_after_run(session)
                raise
        print(f"Colab session {session.name} is ready on NVIDIA L4", flush=True)
        return session


def stop_after_run(session: ColabSession) -> bool:
    try:
        session.stop()
    except (ColabError, OSError) as error:
        print(f"{error}. Retry cleanup with make colab-stop.", file=sys.stderr)
        return False
    return True
