"""Run one AutoResearch task group with its pinned Python and native adapters."""

import argparse
import json
from functools import partial
from inspect import signature
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_to_directory, worker_count
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import DEFAULT_WORKERS, TASK_GROUPS

LEGACY_HELDOUT_GROUPS = ("pymunk", "maniskill")
SUITES = {
    "all": ("worldsmith-all-v3", tuple(TASK_GROUPS)),
    "maniskill": ("worldsmith-manipulation-v1", ("maniskill",)),
    "libero": ("worldsmith-arm-position-v1", ("libero",)),
    "robosuite": ("worldsmith-arm-skills-v1", ("robosuite",)),
}
SUITES.update({
    task: (f"worldsmith-task-{task}", (group,))
    for group, tasks in TASK_GROUPS.items()
    for task in tasks
})
DEV_SEEDS = tuple(range(8))
CONTROLLER = "autoresearch.controller:make_controller"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=TASK_GROUPS)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--heldout-seed-file", type=Path, help="Owner file mapping each task to its first held-out seed")
    selection.add_argument("--task", help="Evaluate one task from the group on development seeds")
    parser.add_argument("--workers", type=worker_count, help="Episode workers per task (default depends on simulator)")
    args = parser.parse_args()
    output = args.output.resolve()
    group = TASK_GROUPS[args.group]
    if args.task is not None:
        if args.task not in group:
            parser.error(f"{args.task} is not in the {args.group} task group")
        group = {args.task: group[args.task]}
    workers = DEFAULT_WORKERS[args.group] if args.workers is None else args.workers
    starts = dict.fromkeys(group, DEV_SEEDS[0])
    if args.heldout_seed_file is not None:
        # Preserve the original combined-suite seed file format.
        groups = LEGACY_HELDOUT_GROUPS if args.group in LEGACY_HELDOUT_GROUPS else (args.group,)
        tasks = {task for name in groups for task in TASK_GROUPS[name]}
        starts = json.loads(args.heldout_seed_file.read_text(encoding="utf-8"))
        if not isinstance(starts, dict) or starts.keys() != tasks or any(
            type(seed) is not int or not len(DEV_SEEDS) <= seed <= 2**32 - len(DEV_SEEDS)
            for seed in starts.values()
        ):
            parser.error("Provide one non-development seed start per task in the owner seed file")
        seeds = {seed + offset for seed in starts.values() for offset in DEV_SEEDS}
        if len(seeds) != len(tasks) * len(DEV_SEEDS):
            parser.error("Held-out seed ranges must be distinct for every task")

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
                open_adapter, factory, len(DEV_SEEDS), starts[task], output / task, task=task,
                workers=workers,
            )
    except ControllerError as exc:
        (output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        (output / "controller_feedback.txt").write_text(exc.public_message + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
