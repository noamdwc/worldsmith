"""Run one AutoResearch task group with its pinned Python and native adapters."""

import argparse
import json
from functools import partial
from inspect import signature
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_to_directory
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import TASK_CATALOG, TASK_GROUPS

SUITES = {
    "all": ("worldsmith-all-v1", tuple(TASK_GROUPS)),
    "maniskill": ("worldsmith-maniskill-v1", ("maniskill",)),
}
DEV_SEEDS = tuple(range(8))
CONTROLLER = "autoresearch.controller:make_controller"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=TASK_GROUPS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--heldout-seed-file", type=Path, help="Owner file mapping each task to its first held-out seed")
    args = parser.parse_args()
    output = args.output.resolve()
    group = TASK_GROUPS[args.group]
    starts = dict.fromkeys(group, DEV_SEEDS[0])
    if args.heldout_seed_file is not None:
        starts = json.loads(args.heldout_seed_file.read_text(encoding="utf-8"))
        if not isinstance(starts, dict) or starts.keys() != TASK_CATALOG.keys() or any(
            type(seed) is not int or not len(DEV_SEEDS) <= seed <= 2**32 - len(DEV_SEEDS)
            for seed in starts.values()
        ):
            parser.error("Provide one non-development seed start per catalog task")
        seeds = {seed + offset for seed in starts.values() for offset in DEV_SEEDS}
        if len(seeds) != len(TASK_CATALOG) * len(DEV_SEEDS):
            parser.error("Held-out seed ranges must be distinct for every task")

    output.mkdir(parents=True)
    try:
        try:
            from autoresearch.controller import make_controller

            parameters = signature(make_controller).parameters
        except Exception as exc:
            raise ControllerError(f"Controller import: {type(exc).__name__}: {exc}") from exc

        for task, load in group.items():
            (output / "task.txt").write_text(task, encoding="utf-8")
            open_adapter, _ = load()
            factory = partial(make_controller, task=task) if "task" in parameters else make_controller
            evaluate_to_directory(
                open_adapter, factory, len(DEV_SEEDS), starts[task], output / task, task=task,
            )
    except ControllerError as exc:
        (output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
