"""Command-line entry point for local and automated evaluations."""

import argparse
import json
import os
from importlib import import_module
from pathlib import Path

from worldsmith.benchmark.evaluate import ControllerError, evaluate, evaluate_all
from worldsmith.benchmark.tasks.constants import (
    BENCHMARKS, COMBINED_BENCHMARK, COMBINED_SPLITS, TASKS, TASK_SPLITS,
)
from worldsmith.benchmark.tasks.interception import Chase, Lead

BASELINES = {"chase": Chase, "lead": Lead}


def main():
    parser = argparse.ArgumentParser(prog="python -m worldsmith")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Run one task or the combined benchmark")
    evaluation.add_argument("--task", choices=(*TASKS, "all"), default="interception")
    evaluation.add_argument(
        "--controller",
        help="chase, lead, or module:factory (default: chase for interception, shared starter otherwise)",
    )
    selection = evaluation.add_mutually_exclusive_group()
    selection.add_argument("--split", choices=COMBINED_SPLITS, help="Seed split (default: dev)")
    selection.add_argument("--seed", type=int, help="Replay a single scenario seed")
    evaluation.add_argument("--output", required=True, type=Path, help="New directory for run files")
    evaluation.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    args = parser.parse_args()

    if args.controller is None:
        args.controller = "chase" if args.task == "interception" else "autoresearch.controller:make_controller"
    if args.controller in BASELINES and args.task != "interception":
        evaluation.error("chase and lead are interception references; use a shared controller for other tasks")
    factory = BASELINES.get(args.controller)
    if factory is None:
        module, _, name = args.controller.partition(":")
        if not module or not name:
            evaluation.error("--controller must be chase, lead, or module:factory")

    if args.seed is None:
        split = args.split or "dev"
        splits = COMBINED_SPLITS if args.task == "all" else TASK_SPLITS[args.task]
        seeds = list(splits[split])
    else:
        split = None
        seeds = [args.seed]
    task = None if args.task == "all" else TASKS[args.task]()
    if task is None:
        benchmark = COMBINED_BENCHMARK
        goal = {name: task_factory().goal for name, task_factory in TASKS.items()}
    else:
        benchmark = BENCHMARKS[args.task]
        goal = task.goal
    config = {
        "benchmark": benchmark,
        "goal": goal,
        "controller": args.controller,
        "split": split,
        "seeds": seeds,
    }

    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        evaluation.error(f"Output path already exists: {args.output}")
    try:
        if factory is None:
            try:
                factory = getattr(import_module(module), name, None)
            except Exception as exc:
                raise ControllerError(f"{type(exc).__name__}: {exc}") from exc
            if not callable(factory):
                raise ControllerError(f"{args.controller} is not a callable controller factory")
        with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as trace:
            if task is None:
                metrics = evaluate_all(factory, seeds, trace)
            else:
                metrics = evaluate(task, factory, seeds, trace)
    except ControllerError as exc:
        # Keep the controller diagnostic separate from the private traceback.
        (args.output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise

    summary = json.dumps({**config, **metrics})
    (args.output / "summary.json").write_text(summary + "\n", encoding="utf-8")

    if args.wandb != "disabled":
        if args.wandb == "offline":
            # SDK error reporting is separate from syncing run metrics.
            os.environ["WANDB_ERROR_REPORTING"] = "false"
        import wandb

        with wandb.init(
            project="worldsmith",
            mode=args.wandb,
            dir=args.output.resolve(),
            config=config,
        ) as run:
            run.log(metrics)

    print(summary)


if __name__ == "__main__":
    main()
