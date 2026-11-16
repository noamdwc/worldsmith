"""Local MCP tools for the AutoResearch development benchmark."""

import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from traceback import format_exc
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.benchmark.evaluator import worker_count
from worldsmith.benchmark.task_catalog import DEFAULT_WORKERS, DEV_SEEDS, DEV_TASK_GROUPS, TASK_CATALOG
from worldsmith.research_feedback import ResearchFeedback, publish_feedback
from worldsmith.research_lifecycle import ROOT, WORKSPACE as DEFAULT_WORKSPACE
from worldsmith.research_lifecycle import copy_entry, research_entries

EVALUATION_TIMEOUT_SECONDS = 90
SUITE_TIMEOUT_SECONDS = {"all": 360, "libero": 300, "robosuite": 120}
SUITES = {
    "all": ("worldsmith-all-v5", tuple(DEV_TASK_GROUPS)),
    "maniskill": ("worldsmith-maniskill-v3", ("maniskill",)),
    "libero": ("worldsmith-arm-position-v2", ("libero",)),
    "robosuite": ("worldsmith-arm-skills-v3", ("robosuite",)),
}
SUITES.update({
    task: (f"worldsmith-task-{task}", (group,))
    for group, tasks in DEV_TASK_GROUPS.items()
    for task in tasks
})
OWNER_RUNS = ROOT / "runs" / "owner"
WORKSPACE = Path(os.environ.get("WORLDSMITH_RESEARCH_WORKSPACE", DEFAULT_WORKSPACE)).absolute()
SUITE = os.environ.get("WORLDSMITH_RESEARCH_SUITE", "all")
if SUITE not in SUITES:
    raise ValueError(f"WORLDSMITH_RESEARCH_SUITE must be one of {', '.join(SUITES)}")
_worker_override = os.environ.get("WORLDSMITH_EVALUATION_WORKERS")
WORKER_OVERRIDE = None if _worker_override is None else worker_count(_worker_override)

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


def run_group(
    group: str, python: str, code: Path, output: Path, deadline: float,
    *, tasks: list[str], seeds: list[int],
):
    workers = DEFAULT_WORKERS[group] if WORKER_OVERRIDE is None else WORKER_OVERRIDE
    log = output.parent.parent / f"{group}-stderr.txt"
    with log.open("w", encoding="utf-8") as stderr:
        result = run_group_process(
            [python, "-B", "-m", "worldsmith.research_evaluation", str(output),
             "--workers", str(workers), "--tasks", *tasks, "--seeds", *map(str, seeds)],
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
    benchmark, groups = SUITES[SUITE]
    tasks_by_group = {
        group: [SUITE] if SUITE in TASK_CATALOG else list(DEV_TASK_GROUPS[group])
        for group in groups
    }
    seeds = list(DEV_SEEDS)
    timeout = SUITE_TIMEOUT_SECONDS.get(
        SUITE, SUITE_TIMEOUT_SECONDS.get(groups[0], EVALUATION_TIMEOUT_SECONDS),
    )
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
        for group, tasks in tasks_by_group.items():
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
                run_group(group, str(python), code, group_results, deadline, tasks=tasks, seeds=seeds)
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
            owner_results, output / "results", benchmark=benchmark, tasks_by_group=tasks_by_group,
            seeds=seeds, controller_sha256=digest,
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
        task = next(iter(latest_feedback.summary["tasks"]))
    return latest_feedback.episode(seed, task)


if __name__ == "__main__":
    mcp.run(transport="stdio")
