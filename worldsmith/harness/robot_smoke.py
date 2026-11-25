"""One real closed-loop episode on the existing navigation development task."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from worldsmith.benchmark.task_catalog import TASK_CATALOG
from worldsmith.inference.client import create_client
from worldsmith.inference.server import ModelServer

from .controller import load_tokenizer
from .evaluate import evaluate


def main():
    tokenizer = load_tokenizer()
    with ModelServer(), create_client() as client, TemporaryDirectory() as temporary:
        output = Path(temporary) / "evaluation"
        summary = evaluate(TASK_CATALOG["navigation"], client, tokenizer, output, task="navigation")
        episode = json.loads((output / "episodes.jsonl").read_text())
        row = summary["episodes"][0]
        print(json.dumps({
            "task": "navigation", "seed": 0, "action_repeat": 5,
            "environment_steps": row["length"], "slm_decisions": (row["length"] + 4) // 5,
            "terminated": episode["steps"][-1]["terminated"],
            "truncated": episode["steps"][-1]["truncated"],
            **{name: value for name, value in row.items() if name not in ("seed", "length")},
        }, indent=2))


if __name__ == "__main__":
    main()
