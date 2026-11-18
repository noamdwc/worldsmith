"""Manage the named Colab runtime for this checkout."""

import fcntl
import json
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from worldsmith.research_lifecycle import ROOT


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

    def _command(self, action: str, *args: str, timeout: float = 60, code: str | None = None) -> str:
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
        except OSError:
            raise ColabError(f"Could not invoke Colab {action}") from None
        # Avoid forwarding raw CLI logs, which may include OAuth or proxy tokens.
        if result.returncode:
            if f"Session '{self.name}' not found" in result.stdout or "appears to be lost" in result.stdout:
                raise ColabError("Colab session disappeared or is unavailable")
            if action == "new":
                raise ColabError("Colab L4 allocation failed. Check CLI authentication and L4 quota")
            raise ColabError(f"Colab {action} failed for {self.name}")
        return result.stdout

    def execute(self, code: str, *, timeout: int) -> str:
        return self._command("exec", "--timeout", str(timeout), timeout=timeout + 15, code=code)

    def allocate(self):
        # Let the CLI record the assignment before cleanup handles a signal.
        interrupted = None

        def remember(signum, frame):
            """Defer interruption until the CLI has recorded the new runtime."""
            nonlocal interrupted
            interrupted = signum

        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Installing a handler returns the old one, which we restore below.
            previous[sig] = signal.signal(sig, remember)
        try:
            self._command("new", "--gpu", "L4", timeout=180)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if interrupted is not None:
                raise SystemExit(128 + interrupted)

    def is_assigned(self) -> bool:
        return f"Session '{self.name}' not found" not in self._command("status", timeout=15)

    def release(self):
        try:
            self._command("stop")
        except ColabError:
            # An expired runtime can still have a stale entry in the CLI store.
            if self.is_assigned():
                raise

    def stop(self):
        self.release()
        (STATE_DIR / "session.json").unlink(missing_ok=True)


def stop_after_run(session: ColabSession) -> bool:
    try:
        session.stop()
    except (ColabError, OSError) as error:
        print(f"{error}. Retry cleanup with make colab-stop.", file=sys.stderr)
        return False
    return True
