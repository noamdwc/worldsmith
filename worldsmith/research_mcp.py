"""Local MCP tools for the AutoResearch development benchmark."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from traceback import format_exc
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_evaluation import SUITES
from worldsmith.research_feedback import ResearchFeedback, publish_feedback
from worldsmith.research_lifecycle import ROOT, WORKSPACE, copy_entry, research_entries

# The full starter suite takes about 17–19 seconds on an Apple M2 Pro.
EVALUATION_TIMEOUT_SECONDS = 90
SUITE_TIMEOUT_SECONDS = {"libero": 300}
OWNER_RUNS = ROOT / "runs" / "owner"
SUITE = os.environ.get("WORLDSMITH_RESEARCH_SUITE", "all")
if SUITE not in SUITES:
    raise ValueError(f"WORLDSMITH_RESEARCH_SUITE must be one of {', '.join(SUITES)}")

mcp = FastMCP("worldsmith")
latest_feedback: ResearchFeedback | None = None
retained_summary: dict | None = None


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_code(destination: Path):
    destination.mkdir(parents=True)
    # A regular package prevents imports from falling back to the editable workspace.
    (destination / "__init__.py").touch()
    for entry in research_entries(WORKSPACE):
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


def run_group(group: str, python: str, code: Path, output: Path, deadline: float):
    log = output.parent.parent / f"{group}-stderr.txt"
    with log.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            [python, "-B", "-m", "worldsmith.research_evaluation", group, str(output)],
            cwd=code,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=stderr,
            stderr=subprocess.STDOUT,
            timeout=max(0, deadline - monotonic()),
        )
    if result.returncode:
        diagnostic = output / "controller_feedback.txt"
        if diagnostic.is_file():
            raise EvaluationFailure(diagnostic.read_text(encoding="utf-8").strip())
        marker = output / "task.txt"
        message = "Evaluator could not complete the run"
        if marker.is_file():
            message += f" while evaluating {marker.read_text(encoding='utf-8')}"
        raise EvaluationFailure(message) from subprocess.CalledProcessError(result.returncode, result.args)


class EvaluationFailure(RuntimeError):
    """Evaluation failure whose message can be included in public feedback."""


@mcp.tool(structured_output=True)
def evaluate_controller() -> dict[str, object]:
    """Evaluate the shared controller on this session's development suite and save the run."""
    global latest_feedback, retained_summary
    started = monotonic()
    timeout = SUITE_TIMEOUT_SECONDS.get(SUITE, EVALUATION_TIMEOUT_SECONDS)
    deadline = started + timeout
    run_id = f"mcp-{uuid4().hex}"
    output = WORKSPACE / "runs" / run_id
    owner_run = OWNER_RUNS / run_id
    try:
        owner_run.mkdir(parents=True)
        output.mkdir(parents=True)
    except OSError:
        raise RuntimeError(f"Could not create evaluation directories. Run: {run_id}") from None

    record = {"run_id": run_id, "started_at": now()}

    def save_status(status: str, **details):
        for directory in (owner_run, output):
            write_json(directory / "evaluation.json", {**record, "status": status, **details})

    code = owner_run / "code"
    owner_results = owner_run / "results"
    try:
        save_status("preparing")
        snapshot_code(code / "autoresearch")
        copy_entry(code, output / "code", skip_cache=True)
        digest = snapshot_hash(code)
        save_status("running")
        _, groups = SUITES[SUITE]
        for group in groups:
            python = Path(sys.executable)
            if group != "pymunk":
                variable = f"WORLDSMITH_{group.upper()}_PYTHON"
                python = Path(os.environ.get(variable, ROOT / group / ".venv/bin/python")).expanduser()
                if not python.is_absolute() or not os.access(python, os.X_OK):
                    raise EvaluationFailure("Evaluation environment is unavailable") from ValueError(
                        f"Set {variable} to the absolute path of the pinned interpreter"
                    )
            group_results = owner_results / group
            try:
                run_group(group, str(python), code, group_results, deadline)
            except subprocess.TimeoutExpired:
                marker = group_results / "task.txt"
                (owner_run / "error.txt").write_text(format_exc(), encoding="utf-8")
                message = f"Evaluation timed out after {timeout} seconds"
                if marker.is_file():
                    message += f" while evaluating {marker.read_text(encoding='utf-8')}"
                save_status("timed_out", error=message, ended_at=now())
                raise RuntimeError(f"{message}. Run: {run_id}") from None
            if snapshot_hash(code) != digest:
                raise EvaluationFailure("Controller snapshot changed during evaluation")
        feedback = publish_feedback(
            owner_results, output / "results", suite=SUITE, controller_sha256=digest,
            started=started, retained=retained_summary,
        )
    except EvaluationFailure as exc:
        (owner_run / "error.txt").write_text(format_exc(), encoding="utf-8")
        message = f"Evaluation failed: {exc}"
        save_status("failed", error=message, ended_at=now())
        raise RuntimeError(f"{message}. Run: {run_id}") from None
    except (OSError, KeyError, ValueError):
        (owner_run / "error.txt").write_text(format_exc(), encoding="utf-8")
        message = "Evaluation could not prepare the snapshot or publish feedback."
        save_status("failed", error=message, ended_at=now())
        raise RuntimeError(f"{message} Run: {run_id}") from None
    save_status("succeeded", ended_at=now())
    latest_feedback = feedback
    if retained_summary is None or feedback.summary["aggregate_score"] > retained_summary["aggregate_score"]:
        retained_summary = feedback.summary
    return {
        "run_id": run_id,
        "run_dir": (output / "results").relative_to(WORKSPACE).as_posix(),
        **feedback.summary,
    }


@mcp.tool(structured_output=True)
def get_episode_trace(seed: int, task: str | None = None) -> dict[str, object]:
    """Return a task/seed episode from the latest successful development run in this session."""
    if latest_feedback is None:
        raise ValueError("Run evaluate_controller() first.")
    if task is None:
        task = next(iter(TASK_GROUPS[SUITES[SUITE][1][0]]))
    return latest_feedback.episode(seed, task)


if __name__ == "__main__":
    mcp.run(transport="stdio")
