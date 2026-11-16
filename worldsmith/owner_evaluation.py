"""Select development, final validation, or historical replays for owner runs."""

import argparse
import json
from pathlib import Path

from worldsmith.benchmark.evaluator import worker_count
from worldsmith.benchmark.task_catalog import (
    DEFAULT_WORKERS, DEV_SEEDS, DEV_TASK_GROUPS, TASK_GROUPS, VALIDATION_SEEDS,
)
from worldsmith.research_evaluation import evaluate_tasks

LEGACY_HELDOUT_GROUPS = ("pymunk", "maniskill")
EXPANSION_TASKS = {"roll_ball", "quadruped_reach", "wipe_table"}


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
    seeds = VALIDATION_SEEDS if args.validation else DEV_SEEDS
    groups = TASK_GROUPS if args.validation or args.heldout_seed_file is not None else DEV_TASK_GROUPS
    tasks = list(groups[args.group])
    if args.task is not None:
        if args.task not in tasks:
            parser.error(f"{args.task} is not in the {args.group} development task group")
        tasks = [args.task]
    workers = DEFAULT_WORKERS[args.group] if args.workers is None else args.workers
    task_seeds = dict.fromkeys(tasks, list(seeds))
    if args.heldout_seed_file is not None:
        # Preserve the original combined-suite seed file format.
        file_groups = LEGACY_HELDOUT_GROUPS if args.group in LEGACY_HELDOUT_GROUPS else (args.group,)
        file_tasks = {task for name in file_groups for task in TASK_GROUPS[name]}
        starts = json.loads(args.heldout_seed_file.read_text(encoding="utf-8"))
        if not isinstance(starts, dict) or starts.keys() not in (file_tasks, file_tasks - EXPANSION_TASKS) or any(
            type(seed) is not int or not DEV_SEEDS.stop <= seed <= 2**32 - len(seeds)
            for seed in starts.values()
        ):
            parser.error("Provide one non-development seed start per task in the owner seed file")
        heldout_seeds = {seed + offset for seed in starts.values() for offset in range(len(seeds))}
        if len(heldout_seeds) != len(starts) * len(seeds):
            parser.error("Held-out seed ranges must be distinct for every task")
        # A pre-expansion seed file can still replay its original task set.
        task_seeds = {
            task: list(range(starts[task], starts[task] + len(seeds)))
            for task in tasks if task in starts
        }

    evaluate_tasks(task_seeds, args.output, workers=workers)


if __name__ == "__main__":
    main()
