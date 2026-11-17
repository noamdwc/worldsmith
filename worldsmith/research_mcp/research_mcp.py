"""Local MCP tools for the AutoResearch development benchmark."""

import os
import subprocess
import sys
from pathlib import Path
from time import monotonic
from traceback import format_exc
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.benchmark.task_catalog import DEV_SEEDS, DEV_TASK_GROUPS, TASK_CATALOG
from worldsmith.research_feedback import ResearchFeedback, publish_feedback
from worldsmith.research_lifecycle import ROOT, copy_entry

from .constant import (
    EVALUATION_TIMEOUT_SECONDS, OWNER_RUNS, SUITE, SUITE_TIMEOUT_SECONDS, SUITES, WORKSPACE,
)
from .evaluation_io import EvaluationFailure, now, run_group, snapshot_code, snapshot_hash, write_json

mcp = FastMCP("worldsmith")
latest_feedback: ResearchFeedback | None = None
retained_summary: dict | None = None


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
        snapshot_code(WORKSPACE, code / "autoresearch")
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
