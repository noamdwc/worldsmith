"""Evaluate the tasks and seeds supplied by the caller in a pinned Python runtime."""

import argparse
from pathlib import Path

from worldsmith.benchmark.evaluator import worker_count
from worldsmith.benchmark.task_catalog import TASK_CATALOG
from worldsmith.evaluation.evaluation_utils import evaluate_tasks


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
