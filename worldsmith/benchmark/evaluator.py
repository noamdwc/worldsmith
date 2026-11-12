"""Evaluate controllers with task-provided scoring and save their traces."""

import argparse
import json
import platform
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import ContextManager, Protocol

from worldsmith.benchmark.results import write_episode, write_summary
from worldsmith.benchmark.runner import ControllerFactory, Simulator, run_episode
from worldsmith.benchmark.task_catalog import TASK_CATALOG


class EvaluationAdapter(Simulator, Protocol):
    """Provide task metadata, episode metrics, and their native aggregation."""

    def metadata(self) -> dict: ...
    def score(self, episode: dict) -> dict: ...
    def summarize(self, episodes: list[dict]) -> dict: ...


def evaluate_episodes(
    adapter: EvaluationAdapter,
    controller_factory: ControllerFactory,
    seeds: list[int],
    record: Callable[[dict, dict], None],
) -> tuple[list[dict], dict]:
    rows = []
    scores = []
    for seed in seeds:
        episode = run_episode(adapter, controller_factory, seed)
        metrics = adapter.score(episode)
        record(episode, metrics)
        rows.append({"seed": seed, "length": episode["length"], **metrics})
        scores.append(metrics)
    return rows, adapter.summarize(scores)


def evaluate(
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
            def record(episode: dict, metrics: dict) -> None:
                write_episode(traces, {**episode, **metrics})

            rows, aggregate = evaluate_episodes(adapter, controller_factory, seeds, record)
        summary = adapter.metadata()

    factory_type = type(controller_factory)
    module = getattr(controller_factory, "__module__", factory_type.__module__)
    name = getattr(controller_factory, "__qualname__", factory_type.__qualname__)
    summary.update(
        controller=f"{module}.{name}",
        python=platform.python_version(),
        platform=platform.platform(),
        seeds=seeds,
        episodes=rows,
        **aggregate,
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

    print(json.dumps(evaluate(open_adapter, factory, args.episodes, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
