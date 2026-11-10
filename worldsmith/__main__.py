"""Command-line entry point for local and automated evaluations."""

import argparse
import json
import os
from importlib import import_module
from pathlib import Path

from worldsmith.benchmark.baselines import Chase, Lead
from worldsmith.benchmark.evaluate import BENCHMARK, SPLITS, evaluate

BASELINES = {"chase": Chase, "lead": Lead}


def main():
    parser = argparse.ArgumentParser(prog="python -m worldsmith")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Run the interception benchmark")
    evaluation.add_argument("--controller", default="chase", help="chase, lead, or module:factory")
    selection = evaluation.add_mutually_exclusive_group()
    selection.add_argument("--split", choices=SPLITS, help="Seed split (default: dev)")
    selection.add_argument("--seed", type=int, help="Replay a single scenario seed")
    evaluation.add_argument("--output", required=True, type=Path, help="New directory for run files")
    evaluation.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    args = parser.parse_args()

    factory = BASELINES.get(args.controller)
    if factory is None:
        module, _, name = args.controller.partition(":")
        if not module or not name:
            evaluation.error("--controller must be chase, lead, or module:factory")
        factory = getattr(import_module(module), name, None)
        if not callable(factory):
            evaluation.error(f"{args.controller} is not a callable controller factory")

    if args.seed is None:
        split = args.split or "dev"
        seeds = list(SPLITS[split])
    else:
        split = None
        seeds = [args.seed]
    config = {"benchmark": BENCHMARK, "controller": args.controller, "split": split, "seeds": seeds}

    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        evaluation.error(f"Output path already exists: {args.output}")
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as trace:
        metrics = evaluate(factory, seeds, trace)

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
