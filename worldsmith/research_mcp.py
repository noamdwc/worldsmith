"""Local MCP tools for the AutoResearch development benchmark."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.research_lifecycle import ROOT, WORKSPACE, copy_entry, research_entries

EVALUATION_TIMEOUT_SECONDS = 30
mcp = FastMCP("worldsmith")
latest_run: Path | None = None


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
    global latest_run
    run_id = f"mcp-{uuid4().hex}"
    output = WORKSPACE / "runs" / run_id
    output.mkdir(parents=True)
    record = {
        "run_id": run_id,
        "started_at": now(),
        "status": "preparing",
    }
    metadata = output / "evaluation.json"
    write_json(metadata, record)
    code = output / "code"
    try:
        snapshot_code(code / "autoresearch")
        record["status"] = "running"
        write_json(metadata, record)
        # Run the exact snapshot in a fresh process and keep its stdout off MCP's wire.
        result = subprocess.run(
            [
                sys.executable, "-B", "-m", "worldsmith", "evaluate",
                "--controller", "autoresearch.controller:make_controller",
                "--task", "all", "--split", "dev", "--output", str(output / "results"),
            ],
            cwd=code,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )
        if result.returncode:
            lines = result.stderr.strip().splitlines()
            detail = lines[-1] if lines else f"exit status {result.returncode}"
            raise RuntimeError(f"Evaluation failed: {detail}")
        summary = json.loads((output / "results" / "summary.json").read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        message = f"Evaluation timed out after {EVALUATION_TIMEOUT_SECONDS} seconds."
        write_json(metadata, {**record, "status": "timed_out", "error": message, "ended_at": now()})
        raise RuntimeError(f"{message} Run: {run_id}") from None
    except (OSError, RuntimeError, ValueError) as exc:
        write_json(metadata, {**record, "status": "failed", "error": str(exc), "ended_at": now()})
        raise RuntimeError(f"{exc}. Run: {run_id}") from None
    write_json(metadata, {**record, "status": "succeeded", "ended_at": now()})
    latest_run = output
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
    if latest_run is None:
        raise ValueError("Run evaluate_controller() first.")
    with (latest_run / "results" / "episodes.jsonl").open(encoding="utf-8") as trace:
        for line in trace:
            episode = json.loads(line)
            if episode["seed"] == seed and episode["initial_observation"]["goal"] == task:
                return episode
    raise ValueError(f"Seed {seed} was not in the latest development run for task {task!r}.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
