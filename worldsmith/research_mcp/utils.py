"""File, snapshot, and subprocess helpers for AutoResearch."""

import json
import os
import signal
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from worldsmith.research_lifecycle import copy_entry, research_entries


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_code(workspace: Path, destination: Path):
    destination.mkdir(parents=True)
    # A regular package prevents imports from falling back to the editable workspace.
    (destination / "__init__.py").touch()
    for entry in research_entries(workspace):
        if entry.name not in {"JOURNAL.md", "runs", "__pycache__"}:
            copy_entry(entry, destination / entry.name, skip_cache=True)
    if not (destination / "controller.py").is_file():
        raise ValueError("Workspace has no controller.py to evaluate.")


def snapshot_hash(code: Path) -> str:
    # Include helper files as well as controller.py in the snapshot identity.
    digest = sha256()
    for path in sorted(code.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(code).as_posix().encode() + b"\0")
            digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def run_group_process(command: list[str], *, timeout: float, **options) -> subprocess.CompletedProcess:
    with subprocess.Popen(command, start_new_session=True, **options) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            # The suite deadline also covers spawned episode workers and native code.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            raise
    return subprocess.CompletedProcess(command, returncode)
