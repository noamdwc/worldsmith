"""Evaluate controllers with task-provided scoring and save their traces."""

import argparse
import json
import platform
from collections.abc import Callable, Mapping
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import ContextManager, Protocol, TypeVar

from worldsmith.benchmark.results import TaskEvaluation, write_scored_episode, write_summary
from worldsmith.benchmark.runner import Controller, ControllerFactory, Simulator, run_episode
from worldsmith.benchmark.task_catalog import TASK_CATALOG


class EvaluationAdapter(Simulator, Protocol):
    """Provide task metadata, episode metrics, and their native aggregation."""

    def metadata(self) -> dict: ...
    def score(self, episode: dict) -> dict: ...
    def summarize(self, episodes: list[dict]) -> dict: ...


Adapter = TypeVar("Adapter", bound=EvaluationAdapter)


def evaluate_episodes(
    adapter: EvaluationAdapter,
    controller_factory: ControllerFactory,
    seeds: list[int],
    record: Callable[[dict, dict], None],
) -> TaskEvaluation:
    rows = []
    scores = []
    for seed in seeds:
        episode = run_episode(adapter, controller_factory, seed)
        metrics = adapter.score(episode)
        record(episode, metrics)
        rows.append({"seed": seed, "length": episode["length"], **metrics})
        scores.append(metrics)
    return TaskEvaluation(rows, adapter.summarize(scores))


def evaluate(
    tasks: Mapping[str, Adapter],
    controller_factory: ControllerFactory,
    seeds: list[int],
    record: Callable[[dict, dict], None],
    *,
    prepare_controller: Callable[[Adapter, ControllerFactory], Controller] | None = None,
    summarize: Callable[[dict[str, TaskEvaluation]], dict] | None = None,
) -> dict:
    """Evaluate open adapters in task order, with a shared factory and seed set.

    Controller preparation runs after each reset. Return the task results
    directly or pass them to the supplied benchmark summary policy.
    """
    results = {}
    for name, adapter in tasks.items():
        factory = controller_factory
        if prepare_controller is not None:
            factory = partial(prepare_controller, adapter, controller_factory)
        results[name] = evaluate_episodes(adapter, factory, seeds, record)
    return results if summarize is None else summarize(results)


def evaluate_to_directory(
    open_adapter: Callable[[], ContextManager[EvaluationAdapter]],
    controller_factory: ControllerFactory,
    episodes: int,
    seed: int,
    output: Path,
) -> dict:
    if episodes < 1 or seed < 0 or seed + episodes > 2**32:
        raise ValueError("Use a positive episode count and seeds in [0, 2**32).")
    seeds = list(range(seed, seed + episodes))

    # Native simulator initialization can change the working directory.
    output = output.resolve()
    output.mkdir(parents=True)
    with open_adapter() as adapter:
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as traces:
            result = evaluate(
                {"task": adapter}, controller_factory, seeds, partial(write_scored_episode, traces)
            )["task"]
        summary = adapter.metadata()

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
    args = parser.parse_args()

    open_adapter, factory = TASK_CATALOG[args.task]()
    if args.controller is not None:
        module, _, name = args.controller.partition(":")
        if not module or not name:
            parser.error("--controller must be module:factory")
        factory = getattr(import_module(module), name, None)
        if not callable(factory):
            parser.error("--controller must name a callable factory")

    summary = evaluate_to_directory(open_adapter, factory, args.episodes, args.seed, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
