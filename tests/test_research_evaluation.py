import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from worldsmith import research_evaluation
from worldsmith.benchmark.task_catalog import (
    DEV_SEEDS, TASK_CATALOG, TASK_GROUPS, VALIDATION_ONLY_TASKS, VALIDATION_SEEDS,
)


def test_task_and_seed_partitions():
    assert VALIDATION_ONLY_TASKS <= TASK_CATALOG.keys()
    assert VALIDATION_ONLY_TASKS.isdisjoint(research_evaluation.SUITES)
    assert set(DEV_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert list(DEV_SEEDS) == list(range(8))
    assert list(VALIDATION_SEEDS) == list(range(8, 16))


@pytest.mark.parametrize("group", TASK_GROUPS)
def test_owner_validation_selects_all_group_tasks(monkeypatch, tmp_path, group):
    tasks = TASK_GROUPS[group]
    monkeypatch.setitem(TASK_GROUPS, group, dict.fromkeys(tasks, lambda: (None, None)))
    output = tmp_path / "validation"
    monkeypatch.setattr(sys, "argv", ["research_evaluation", group, str(output), "--validation"])
    calls = []

    def record(adapter, factory, episodes, seed, directory, *, task, workers):
        assert directory == output / task
        calls.append((task, range(seed, seed + episodes)))

    monkeypatch.setattr(research_evaluation, "evaluate_to_directory", record)
    research_evaluation.main()

    assert tasks.keys() & VALIDATION_ONLY_TASKS
    assert tasks.keys() - VALIDATION_ONLY_TASKS
    assert calls == [(task, VALIDATION_SEEDS) for task in tasks]


def test_owner_validation_runs_selected_snapshot(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "code/autoresearch"
    snapshot.mkdir(parents=True)
    (snapshot / "__init__.py").touch()
    (snapshot / "controller.py").write_text(
        "from worldsmith.api import Action\n"
        "def make_controller(*, task):\n"
        "    return lambda observation: Action(0, 0.125)\n",
    )
    output = tmp_path / "validation"
    subprocess.run(
        [sys.executable, "-B", "-m", "worldsmith.research_evaluation", "pymunk", str(output), "--validation"],
        cwd=snapshot.parent, env={**os.environ, "PYTHONPATH": str(repo)}, check=True, timeout=30,
    )
    for task in TASK_GROUPS["pymunk"]:
        summary = json.loads((output / task / "summary.json").read_text())
        assert summary["seeds"] == list(VALIDATION_SEEDS)
        with (output / task / "episodes.jsonl").open() as trace:
            first_step = json.loads(trace.readline())["steps"][0]
        assert first_step["action"]["throttle"] == 0.125


def test_validation_only_task_is_still_available_for_direct_execution(tmp_path):
    task = "rendezvous"
    output = tmp_path / "debug"
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.research_evaluation", "pymunk", str(output), "--task", task],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "development task group" in result.stderr
    assert not output.exists()

    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.benchmark.evaluator", "--task", task,
         "--episodes", "1", "--seed", "8", "--output", str(output)],
        capture_output=True, text=True, check=True, timeout=10,
    )
    assert json.loads(result.stdout)["seeds"] == [8]
