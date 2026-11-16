import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest


@pytest.fixture
def research_metrics():
    from worldsmith.benchmark.evaluator import evaluate
    from worldsmith.benchmark.pymunk_adapter import (
        PymunkAdapter, summarize_research, write_research_episode,
    )

    def run(tasks, factory, seeds, trace, *, combined=False):
        return evaluate(
            {name: PymunkAdapter(task) for name, task in tasks.items()},
            factory, seeds, partial(write_research_episode, trace),
            summarize=partial(summarize_research, combined=combined),
        )

    return run


@pytest.fixture
def check_native_parallel_artifacts(tmp_path):
    def check(task):
        for workers in (1, 2):
            subprocess.run(
                [sys.executable, "-B", "-m", "worldsmith.benchmark.evaluator", "--task", task,
                 "--episodes", "2", "--workers", str(workers), "--output", str(tmp_path / str(workers))],
                cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL, check=True, timeout=120,
            )
        for name in ("summary.json", "episodes.jsonl"):
            assert (tmp_path / "1" / name).read_bytes() == (tmp_path / "2" / name).read_bytes()

    return check


@pytest.fixture
def check_episode_order():
    from worldsmith.benchmark.runner import run_scored_episode
    from worldsmith.benchmark.task_catalog import TASK_CATALOG
    from worldsmith.research_baseline.controller import make_controller

    def check(task):
        open_adapter, _ = TASK_CATALOG[task]()
        factory = partial(make_controller, task=task)
        with open_adapter() as adapter:
            forward = {seed: run_scored_episode(adapter, factory, seed, task) for seed in (0, 1)}
            reverse = {seed: run_scored_episode(adapter, factory, seed, task) for seed in (1, 0)}
            assert forward == reverse
            assert adapter.summarize([score for _, score in forward.values()]) == adapter.summarize(
                [score for _, score in reverse.values()]
            )

    return check
