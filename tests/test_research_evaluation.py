import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

import pytest

from worldsmith import owner_evaluation
from worldsmith.benchmark.task_catalog import (
    DEV_SEEDS, TASK_CATALOG, TASK_GROUPS, VALIDATION_ONLY_TASKS, VALIDATION_SEEDS,
)
from worldsmith.research_feedback import publish_feedback
from worldsmith.research_mcp.constant import SUITES


def test_task_and_seed_partitions():
    assert VALIDATION_ONLY_TASKS <= TASK_CATALOG.keys()
    assert VALIDATION_ONLY_TASKS.isdisjoint(SUITES)
    assert set(DEV_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert list(DEV_SEEDS) == list(range(8))
    assert list(VALIDATION_SEEDS) == list(range(8, 16))


@pytest.mark.parametrize("group", TASK_GROUPS)
def test_owner_validation_selects_all_group_tasks(monkeypatch, tmp_path, group):
    tasks = TASK_GROUPS[group]
    output = tmp_path / "validation"
    monkeypatch.setattr(sys, "argv", ["owner_evaluation", group, str(output), "--validation"])
    calls = []

    def record(task_seeds, directory, *, workers):
        assert directory == output
        calls.extend(task_seeds.items())

    monkeypatch.setattr(owner_evaluation, "evaluate_tasks", record)
    owner_evaluation.main()

    assert tasks.keys() & VALIDATION_ONLY_TASKS
    assert tasks.keys() - VALIDATION_ONLY_TASKS
    assert calls == [(task, list(VALIDATION_SEEDS)) for task in tasks]


@pytest.mark.parametrize("validation", [True, False])
def test_runner_uses_selected_snapshot_and_seeds(tmp_path, validation):
    repo = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "code/autoresearch"
    snapshot.mkdir(parents=True)
    (snapshot / "__init__.py").touch()
    (snapshot / "controller.py").write_text(
        "from worldsmith.api import Action\n"
        "def make_controller(*, task):\n"
        "    return lambda observation: Action(0, 0.125)\n",
    )
    output = tmp_path / "raw/pymunk"
    if validation:
        command = ["worldsmith.owner_evaluation", "pymunk", str(output), "--validation"]
        tasks = list(TASK_GROUPS["pymunk"])
        seeds = list(VALIDATION_SEEDS)
    else:
        tasks = ["rendezvous", "interception"]
        seeds = [12, 2]
        command = [
            "worldsmith.research_evaluation", str(output), "--workers", "2",
            "--tasks", *tasks, "--seeds", *map(str, seeds),
        ]
    subprocess.run(
        [sys.executable, "-B", "-m", *command],
        cwd=snapshot.parent, env={**os.environ, "PYTHONPATH": str(repo)}, check=True, timeout=30,
    )
    assert {path.name for path in output.iterdir() if path.is_dir()} == set(tasks)
    for task in tasks:
        summary = json.loads((output / task / "summary.json").read_text())
        assert summary["seeds"] == seeds
        with (output / task / "episodes.jsonl").open() as trace:
            first_step = json.loads(trace.readline())["steps"][0]
        assert first_step["action"]["throttle"] == 0.125

    if not validation:
        feedback = publish_feedback(
            output.parent, tmp_path / "public", benchmark="explicit-inputs",
            tasks_by_group={"pymunk": tasks}, seeds=seeds,
            controller_sha256="", started=monotonic(), retained=None,
        )
        assert list(feedback.summary["tasks"]) == tasks
        assert [(row["task"], row["seed"]) for row in feedback.episodes] == [
            (task, seed) for task in tasks for seed in seeds
        ]


def test_validation_only_task_is_still_available_for_direct_execution(tmp_path):
    task = "rendezvous"
    output = tmp_path / "debug"
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.owner_evaluation", "pymunk", str(output), "--task", task],
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
