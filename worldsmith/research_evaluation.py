"""Run one AutoResearch task group with its pinned Python and native adapters."""

import argparse
import json
from functools import partial
from inspect import signature
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_to_directory, worker_count
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import (
    DEFAULT_WORKERS, DEV_SEEDS, DEV_TASK_GROUPS, TASK_GROUPS, VALIDATION_SEEDS,
)

LEGACY_HELDOUT_GROUPS = ("pymunk", "maniskill")
EXPANSION_TASKS = {"roll_ball", "quadruped_reach", "wipe_table"}
SUITES = {
    "all": ("worldsmith-all-v5", tuple(DEV_TASK_GROUPS)),
    "maniskill": ("worldsmith-maniskill-v3", ("maniskill",)),
    "libero": ("worldsmith-arm-position-v2", ("libero",)),
    "robosuite": ("worldsmith-arm-skills-v3", ("robosuite",)),
}
SUITES.update({
    task: (f"worldsmith-task-{task}", (group,))
    for group, tasks in DEV_TASK_GROUPS.items()
    for task in tasks
})
CONTROLLER = "autoresearch.controller:make_controller"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=TASK_GROUPS)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--validation", action="store_true", help="Run all group tasks on final validation seeds",
    )
    selection.add_argument("--heldout-seed-file", type=Path, help="Owner file mapping each task to its first held-out seed")
    selection.add_argument("--task", help="Evaluate one task from the group on development seeds")
    parser.add_argument("--workers", type=worker_count, help="Episode workers per task (default depends on simulator)")
    args = parser.parse_args()
    output = args.output.resolve()
    seeds = VALIDATION_SEEDS if args.validation else DEV_SEEDS
    groups = TASK_GROUPS if args.validation or args.heldout_seed_file is not None else DEV_TASK_GROUPS
    group = groups[args.group]
    if args.task is not None:
        if args.task not in group:
            parser.error(f"{args.task} is not in the {args.group} development task group")
        group = {args.task: group[args.task]}
    workers = DEFAULT_WORKERS[args.group] if args.workers is None else args.workers
    starts = dict.fromkeys(group, seeds.start)
    if args.heldout_seed_file is not None:
        # Preserve the original combined-suite seed file format.
        file_groups = LEGACY_HELDOUT_GROUPS if args.group in LEGACY_HELDOUT_GROUPS else (args.group,)
        tasks = {task for name in file_groups for task in TASK_GROUPS[name]}
        starts = json.loads(args.heldout_seed_file.read_text(encoding="utf-8"))
        if not isinstance(starts, dict) or starts.keys() not in (tasks, tasks - EXPANSION_TASKS) or any(
            type(seed) is not int or not DEV_SEEDS.stop <= seed <= 2**32 - len(seeds)
            for seed in starts.values()
        ):
            parser.error("Provide one non-development seed start per task in the owner seed file")
        heldout_seeds = {seed + offset for seed in starts.values() for offset in range(len(seeds))}
        if len(heldout_seeds) != len(starts) * len(seeds):
            parser.error("Held-out seed ranges must be distinct for every task")
        # A pre-expansion seed file can still replay its original task set.
        group = {task: load for task, load in group.items() if task in starts}

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

        for task, load in group.items():
            (output / "task.txt").write_text(task, encoding="utf-8")
            open_adapter, _ = load()
            factory = partial(make_controller, task=task) if "task" in parameters else make_controller
            evaluate_to_directory(
                open_adapter, factory, len(seeds), starts[task], output / task, task=task,
                workers=workers,
            )
    except ControllerError as exc:
        (output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        (output / "controller_feedback.txt").write_text(exc.public_message + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
