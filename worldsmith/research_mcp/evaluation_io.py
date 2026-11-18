"""File, snapshot, and subprocess helpers for AutoResearch."""

import json
import os
import signal
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic

from worldsmith.benchmark.task_catalog import DEFAULT_WORKERS
from worldsmith.research_lifecycle import ROOT, copy_entry, research_entries

from .constant import WORKER_OVERRIDE


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


def group_command(python: str, output: Path, *, tasks: list[str], seeds: list[int], workers: int):
    return [
        python, "-B", "-m", "worldsmith.evaluation.research_evaluation", str(output),
        "--workers", str(workers), "--tasks", *tasks, "--seeds", *map(str, seeds),
    ]


def run_group(
    group: str, python: str, code: Path, output: Path, deadline: float,
    *, tasks: list[str], seeds: list[int],
):
    workers = DEFAULT_WORKERS[group] if WORKER_OVERRIDE is None else WORKER_OVERRIDE
    log = output.parent.parent / f"{group}-stderr.txt"
    with log.open("w", encoding="utf-8") as stderr:
        try:
            result = run_group_process(
                group_command(python, output, tasks=tasks, seeds=seeds, workers=workers),
                cwd=code,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                stdout=stderr,
                stderr=subprocess.STDOUT,
                timeout=max(0, deadline - monotonic()),
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationTimeout(task_context(output)) from exc
    if result.returncode:
        diagnostic = output / "controller_feedback.txt"
        if diagnostic.is_file():
            raise EvaluationFailure(diagnostic.read_text(encoding="utf-8").strip())
        message = "Evaluator could not complete the run" + task_context(output)
        raise EvaluationFailure(message) from subprocess.CalledProcessError(result.returncode, result.args)


def task_context(output: Path) -> str:
    marker = output / "task.txt"
    return f" while evaluating {marker.read_text(encoding='utf-8')}" if marker.is_file() else ""


class EvaluationFailure(RuntimeError):
    """Evaluation failure whose message can be included in public feedback."""


class EvaluationTimeout(EvaluationFailure):
    """The suite reached its evaluation deadline."""
