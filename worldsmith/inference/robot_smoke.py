"""One real closed-loop episode on the existing navigation development task."""

import json

from worldsmith.benchmark.adapters.pymunk_adapter import open_pymunk

from .client import create_client
from .controller import load_tokenizer, run_slm_episode
from .server import ModelServer


def main():
    tokenizer = load_tokenizer()
    with ModelServer(), create_client() as client, open_pymunk("navigation") as adapter:
        episode, decisions = run_slm_episode(adapter, client, tokenizer, 0, task="navigation")
        print(json.dumps({
            "task": "navigation", "seed": 0, "action_repeat": 5,
            "environment_steps": episode["length"], "slm_decisions": decisions,
            "terminated": episode["steps"][-1]["terminated"],
            "truncated": episode["steps"][-1]["truncated"],
            **adapter.score(episode),
        }, indent=2))


if __name__ == "__main__":
    main()
