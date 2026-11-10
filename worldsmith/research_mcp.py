"""Local MCP tools for the AutoResearch development benchmark."""

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "autoresearch"
mcp = FastMCP("worldsmith")
latest_run: Path | None = None


@mcp.tool(structured_output=True)
def evaluate_controller() -> dict[str, object]:
    """Evaluate the current controller on development episodes and save the run."""
    global latest_run
    output = WORKSPACE / "runs" / f"mcp-{uuid4().hex}"
    # A fresh process loads the current controller and keeps its stdout off MCP's wire.
    result = subprocess.run(
        [
            sys.executable, "-m", "worldsmith", "evaluate",
            "--controller", "autoresearch.controller:make_controller",
            "--split", "dev", "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1] if lines else f"exit status {result.returncode}"
        raise RuntimeError(f"Evaluation failed: {detail}")

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    latest_run = output
    return {
        "run_dir": output.relative_to(WORKSPACE).as_posix(),
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
    with (latest_run / "episodes.jsonl").open(encoding="utf-8") as trace:
        for line in trace:
            episode = json.loads(line)
            if episode["seed"] == seed:
                return episode
    raise ValueError(f"Seed {seed} was not in the latest development run.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
