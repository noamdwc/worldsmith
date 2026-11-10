"""Local MCP tools for the AutoResearch development benchmark."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from worldsmith.research_lifecycle import (
    HISTORY, ROOT, WORKSPACE, now, snapshot_code, state, workspace_lock, write_json,
)
EVALUATION_TIMEOUT_SECONDS = 30
mcp = FastMCP("worldsmith")
latest_run: Path | None = None


@mcp.tool(structured_output=True)
def evaluate_controller() -> dict[str, object]:
    """Evaluate the current controller on development episodes and save the run."""
    global latest_run
    run_id = f"mcp-{uuid4().hex}"
    output = WORKSPACE / "runs" / run_id
    with workspace_lock(HISTORY) as lock_fd:
        output.mkdir(parents=True)
        record = {
            "run_id": run_id,
            "session_id": state(WORKSPACE).get("session_id"),
            "started_at": now(),
            "status": "preparing",
        }
        metadata = output / "evaluation.json"
        write_json(metadata, record)
        code = output / "code"
        try:
            snapshot_code(WORKSPACE, code / "autoresearch")
            record["status"] = "running"
            write_json(metadata, record)
            # Run the exact snapshot in a fresh process and keep its stdout off MCP's wire.
            result = subprocess.run(
                [
                    sys.executable, "-B", "-m", "worldsmith", "evaluate",
                    "--controller", "autoresearch.controller:make_controller",
                    "--split", "dev", "--output", str(output / "results"),
                ],
                cwd=code,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                # Keep lifecycle commands blocked even if the MCP parent is interrupted.
                pass_fds=(lock_fd,),
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
        "captures": summary["captures"],
        "capture_rate": summary["capture_rate"],
        "mean_closest_distance": summary["mean_closest_distance"],
    }


@mcp.tool(structured_output=True)
def get_episode_trace(seed: int) -> dict[str, object]:
    """Return an episode from the latest successful development run in this session."""
    if latest_run is None:
        raise ValueError("Run evaluate_controller() first.")
    with (latest_run / "results" / "episodes.jsonl").open(encoding="utf-8") as trace:
        for line in trace:
            episode = json.loads(line)
            if episode["seed"] == seed:
                return episode
    raise ValueError(f"Seed {seed} was not in the latest development run.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
