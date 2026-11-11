"""Local MCP tools for the AutoResearch development benchmark."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from traceback import format_exc
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.research_feedback import ResearchFeedback, publish_feedback
from worldsmith.research_lifecycle import ROOT, WORKSPACE, copy_entry, research_entries

EVALUATION_TIMEOUT_SECONDS = 30
OWNER_RUNS = ROOT / "runs" / "owner"
mcp = FastMCP("worldsmith")
latest_feedback: ResearchFeedback | None = None


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


@mcp.tool(structured_output=True)
def evaluate_controller() -> dict[str, object]:
    """Evaluate the shared controller on all three development tasks and save the run."""
    global latest_feedback
    run_id = f"mcp-{uuid4().hex}"
    output = WORKSPACE / "runs" / run_id
    owner_run = OWNER_RUNS / run_id
    try:
        owner_run.mkdir(parents=True)
        output.mkdir(parents=True)
    except OSError:
        raise RuntimeError(f"Could not create evaluation directories. Run: {run_id}") from None

    record = {
        "run_id": run_id,
        "started_at": now(),
    }

    def save_status(status: str, **details):
        for directory in (owner_run, output):
            write_json(directory / "evaluation.json", {**record, "status": status, **details})

    code = owner_run / "code"
    owner_results = owner_run / "results"
    try:
        save_status("preparing")
        snapshot_code(code / "autoresearch")
        copy_entry(code, output / "code", skip_cache=True)
        save_status("running")
        # Run the exact snapshot in a fresh process and keep its stdout off MCP's wire.
        result = subprocess.run(
            [
                sys.executable, "-B", "-m", "worldsmith", "evaluate",
                "--controller", "autoresearch.controller:make_controller",
                "--task", "all", "--split", "dev", "--output", str(owner_results),
            ],
            cwd=code,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )
        if result.stderr:
            (owner_run / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            diagnostic = owner_results / "controller_error.txt"
            detail = diagnostic.read_text(encoding="utf-8").strip() if diagnostic.is_file() else (
                f"evaluator exited with status {result.returncode}"
            )
            message = f"Evaluation failed: {detail}"
            save_status("failed", error=message, ended_at=now())
            raise RuntimeError(f"{message}. Run: {run_id}")
        feedback = publish_feedback(owner_results, output / "results")
    except subprocess.TimeoutExpired:
        message = f"Evaluation timed out after {EVALUATION_TIMEOUT_SECONDS} seconds."
        save_status("timed_out", error=message, ended_at=now())
        raise RuntimeError(f"{message} Run: {run_id}") from None
    except (OSError, KeyError, ValueError):
        (owner_run / "error.txt").write_text(format_exc(), encoding="utf-8")
        message = "Evaluation could not prepare the snapshot or publish feedback."
        save_status("failed", error=message, ended_at=now())
        raise RuntimeError(f"{message} Run: {run_id}") from None
    save_status("succeeded", ended_at=now())
    latest_feedback = feedback
    summary = feedback.summary
    return {
        "run_id": run_id,
        "run_dir": (output / "results").relative_to(WORKSPACE).as_posix(),
        "benchmark": summary["benchmark"],
        "episodes": summary["episodes"],
        "aggregate_score": summary["aggregate_score"],
        "tasks": summary["tasks"],
    }


@mcp.tool(structured_output=True)
def get_episode_trace(seed: int, task: str = "interception") -> dict[str, object]:
    """Return a task/seed episode from the latest successful development run in this session."""
    if latest_feedback is None:
        raise ValueError("Run evaluate_controller() first.")
    return latest_feedback.episode(seed, task)


if __name__ == "__main__":
    mcp.run(transport="stdio")
