"""Compose the SLM controller with the shared benchmark evaluator."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_to_directory
from worldsmith.benchmark.task_catalog import TASK_CATALOG
from worldsmith.inference.client import create_client
from worldsmith.harness.controller import load_tokenizer, make_controller


def evaluate(open_adapter, client, tokenizer, output, *, task, episodes=1, seed=0, action_repeat=5):
    adapter = None
    controller = None

    @contextmanager
    def open_simulator():
        nonlocal adapter
        with open_adapter() as adapter:
            yield adapter

    def factory():
        nonlocal controller
        # The shared runner calls this after reset, when native bounds and clocks
        # are available. Keep its existing zero-argument factory contract.
        controller = make_controller(adapter, client, tokenizer, task=task, action_repeat=action_repeat)
        return controller

    return evaluate_to_directory(
        open_simulator, factory, episodes, seed, output, task=task, workers=1,
        action_repeat=action_repeat, action_result=lambda result: controller.record_result(result),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_CATALOG, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    args = parser.parse_args()
    tokenizer = load_tokenizer()
    with create_client() as client:
        summary = evaluate(
            TASK_CATALOG[args.task], client, tokenizer, args.output,
            task=args.task, episodes=args.episodes, seed=args.seed, action_repeat=args.action_repeat,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
