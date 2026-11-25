"""Evaluate controllers with task-provided scoring and save their traces."""

import argparse
import json
import platform
from collections.abc import Callable
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import ContextManager

from worldsmith.benchmark.parallel import evaluate_parallel
from worldsmith.benchmark.results import (
    TaskEvaluation, record_episodes, write_scored_episode, write_summary,
)
from worldsmith.benchmark.runner import ControllerFactory, EvaluationAdapter, run_scored_episode
from worldsmith.benchmark.task_catalog import DEFAULT_CONTROLLERS, DEFAULT_WORKERS, TASK_CATALOG, TASK_GROUPS


def worker_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("worker count must be a positive integer")
    return count


def evaluate_episodes(
    adapter: EvaluationAdapter,
    controller_factory: ControllerFactory,
    seeds: list[int],
    record: Callable[[dict, dict], None],
    *,
    task: str,
    action_repeat: int = 1,
    action_result: Callable[[dict], None] | None = None,
) -> TaskEvaluation:
    episodes = (
        run_scored_episode(adapter, controller_factory, seed, task, action_repeat=action_repeat, action_result=action_result)
        for seed in seeds
    )
    rows, scores = record_episodes(episodes, record)
    return TaskEvaluation(rows, adapter.summarize(scores))


def evaluate_task(
    open_adapter: Callable[[], ContextManager[EvaluationAdapter]],
    controller_factory: ControllerFactory,
    seeds: list[int],
    record: Callable[[dict, dict], None],
    *,
    task: str,
    workers: int,
    action_repeat: int = 1,
    action_result: Callable[[dict], None] | None = None,
) -> tuple[TaskEvaluation, dict]:
    if workers > 1:
        return evaluate_parallel(
            open_adapter, controller_factory, seeds, record, task=task, workers=workers,
            action_repeat=action_repeat, action_result=action_result,
        )
    with open_adapter() as adapter:
        result = evaluate_episodes(
            adapter, controller_factory, seeds, record, task=task,
            action_repeat=action_repeat, action_result=action_result,
        )
        return result, adapter.metadata()


def evaluate_to_directory(
    open_adapter: Callable[[], ContextManager[EvaluationAdapter]],
    controller_factory: ControllerFactory,
    episodes: int,
    seed: int,
    output: Path,
    *,
    task: str = "task",
    workers: int = 1,
    action_repeat: int = 1,
    action_result: Callable[[dict], None] | None = None,
) -> dict:
    if episodes < 1 or seed < 0 or seed + episodes > 2**32:
        raise ValueError("Use a positive episode count and seeds in [0, 2**32).")
    return evaluate_seeds_to_directory(
        open_adapter, controller_factory, list(range(seed, seed + episodes)), output,
        task=task, workers=workers, action_repeat=action_repeat, action_result=action_result,
    )


def evaluate_seeds_to_directory(
    open_adapter: Callable[[], ContextManager[EvaluationAdapter]],
    controller_factory: ControllerFactory,
    seeds: list[int],
    output: Path,
    *,
    task: str = "task",
    workers: int = 1,
    action_repeat: int = 1,
    action_result: Callable[[dict], None] | None = None,
) -> dict:
    if type(workers) is not int or workers < 1:
        raise ValueError("Use a positive integer worker count.")

    # Native simulator initialization can change the working directory.
    output = output.resolve()
    output.mkdir(parents=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as traces:
        result, summary = evaluate_task(
            open_adapter, controller_factory, seeds, partial(write_scored_episode, traces),
            task=task, workers=workers, action_repeat=action_repeat, action_result=action_result,
        )

    factory_type = type(controller_factory)
    module = getattr(controller_factory, "__module__", factory_type.__module__)
    name = getattr(controller_factory, "__qualname__", factory_type.__qualname__)
    summary.update(
        controller=f"{module}.{name}",
        python=platform.python_version(),
        platform=platform.platform(),
        seeds=seeds,
        episodes=result.episodes,
        **result.metrics,
    )
    write_summary(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASK_CATALOG)
    parser.add_argument("--controller", help="Controller factory as module:factory")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="First seed in a consecutive sequence")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--workers", type=worker_count, help="Episode workers (default depends on simulator)")
    args = parser.parse_args()

    workers = args.workers
    if workers is None:
        workers = next(
            (DEFAULT_WORKERS[group] for group, tasks in TASK_GROUPS.items() if args.task in tasks), 1,
        )
    open_adapter = TASK_CATALOG[args.task]
    if args.controller is not None:
        module, _, name = args.controller.partition(":")
        if not module or not name:
            parser.error("--controller must be module:factory")
        factory = getattr(import_module(module), name, None)
        if not callable(factory):
            parser.error("--controller must name a callable factory")
    else:
        factory = DEFAULT_CONTROLLERS[args.task]()

    summary = evaluate_to_directory(
        open_adapter, factory, args.episodes, args.seed, args.output, task=args.task, workers=workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
