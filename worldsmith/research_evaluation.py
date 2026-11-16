"""Evaluate the tasks and seeds supplied by the caller in a pinned Python runtime."""

import argparse
from functools import partial
from inspect import signature
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_seeds_to_directory, worker_count
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import TASK_CATALOG


def evaluate_tasks(task_seeds: dict[str, list[int]], output: Path, *, workers: int) -> None:
    output = output.resolve()
    output.mkdir(parents=True)
    try:
        try:
            from autoresearch.controller import make_controller

            parameters = signature(make_controller).parameters
        except Exception as exc:
            raise ControllerError(
                f"Controller import: {type(exc).__name__}: {exc}",
                public_message="Could not load controller.py or its make_controller factory",
            ) from exc

        for task, seeds in task_seeds.items():
            (output / "task.txt").write_text(task, encoding="utf-8")
            open_adapter, _ = TASK_CATALOG[task]()
            factory = partial(make_controller, task=task) if "task" in parameters else make_controller
            evaluate_seeds_to_directory(
                open_adapter, factory, seeds, output / task, task=task,
                workers=workers,
            )
    except ControllerError as exc:
        (output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        (output / "controller_feedback.txt").write_text(exc.public_message + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True, choices=TASK_CATALOG)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--workers", required=True, type=worker_count)
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Task names must be distinct")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32)")
    evaluate_tasks(dict.fromkeys(args.tasks, args.seeds), args.output, workers=args.workers)


if __name__ == "__main__":
    main()
