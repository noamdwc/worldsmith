import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest


@pytest.fixture
def evaluate_pymunk(tmp_path_factory):
    from worldsmith.benchmark.evaluator import evaluate_seeds_to_directory
    from worldsmith.benchmark.adapters.pymunk_adapter import open_pymunk

    def run(task, factory, seeds):
        output = tmp_path_factory.mktemp(task) / "results"
        summary = evaluate_seeds_to_directory(
            partial(open_pymunk, task), factory, seeds, output, task=task,
        )
        episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
        return summary, episodes

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
        open_adapter = TASK_CATALOG[task]
        factory = partial(make_controller, task=task)
        with open_adapter() as adapter:
            forward = {seed: run_scored_episode(adapter, factory, seed, task) for seed in (0, 1)}
            reverse = {seed: run_scored_episode(adapter, factory, seed, task) for seed in (1, 0)}
            assert forward == reverse
            assert adapter.summarize([score for _, score in forward.values()]) == adapter.summarize(
                [score for _, score in reverse.values()]
            )

    return check
