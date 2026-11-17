import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from functools import partial

import pytest

from autoresearch import controller as research_controller
from worldsmith.evaluation import owner_evaluation
from worldsmith.benchmark import evaluator
from worldsmith.benchmark.evaluator import evaluate_to_directory, main
from worldsmith.benchmark.task_catalog import DEV_TASK_GROUPS, TASK_CATALOG, TASK_GROUPS
from worldsmith.evaluation.evaluation_utils import evaluate_tasks


class CounterAdapter:
    def __init__(self):
        self.observation = {"count": [0]}

    def reset(self, seed):
        self.seed = seed
        self.observation["count"][0] = 0
        return self.observation, self.observation

    def encode_action(self, action):
        return action.copy()

    def step(self, action):
        count = self.observation["count"]
        count[0] += action.pop()
        done = count[0] >= 3
        return self.observation, 0, done and self.seed == 7, done and self.seed == 8, {
            "native_score": count[0] * self.seed,
        }

    def metadata(self):
        return {"task_id": "counter"}

    def score(self, episode):
        return {"points": episode["steps"][-1]["info"]["native_score"]}

    def summarize(self, episodes):
        return {"total_points": sum(episode["points"] for episode in episodes)}


def test_seeded_evaluation_uses_native_scores_and_snapshots(tmp_path):
    events = []

    @contextmanager
    def open_adapter():
        events.append("opened")
        try:
            yield CounterAdapter()
        finally:
            events.append("closed")

    def factory():
        events.append("controller")
        calls = 0

        def control(observation):
            nonlocal calls
            assert observation["count"][0] == calls
            calls += 1
            return [calls]

        return control

    output = tmp_path / "evaluation"
    summary = evaluate_to_directory(open_adapter, factory, 2, 7, output)

    assert events == ["opened", "controller", "controller", "closed"]
    assert json.loads((output / "summary.json").read_text()) == summary
    assert summary["task_id"] == "counter"
    assert summary["seeds"] == [7, 8]
    assert summary["episodes"] == [
        {"seed": 7, "length": 2, "points": 21},
        {"seed": 8, "length": 2, "points": 24},
    ]
    assert summary["total_points"] == 45

    first, second = [
        json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()
    ]
    assert first["reset_info"] == {"count": [0]}
    assert first["final_observation"] == {"count": [3]}
    assert [step["observation"]["count"] for step in first["steps"]] == [[0], [1]]
    assert [step["action"] for step in first["steps"]] == [[1], [2]]
    assert first["points"] == 21
    assert first["steps"][-1]["terminated"] and not first["steps"][-1]["truncated"]
    assert second["steps"][-1]["truncated"] and not second["steps"][-1]["terminated"]


def test_task_group_uses_shared_factory_and_native_scores(tmp_path, monkeypatch):
    class DoubleScoreAdapter(CounterAdapter):
        def score(self, episode):
            return {"points": 2 * super().score(episode)["points"]}

    factory_calls = []

    def factory(*, task):
        factory_calls.append(task)
        return lambda observation: [3]

    def counter_setup(adapter_type):
        return partial(nullcontext, adapter_type()), None

    monkeypatch.setattr(research_controller, "make_controller", factory)
    monkeypatch.setitem(TASK_CATALOG, "double", partial(counter_setup, DoubleScoreAdapter))
    monkeypatch.setitem(TASK_CATALOG, "counter", partial(counter_setup, CounterAdapter))
    output = tmp_path / "group"
    evaluate_tasks({"double": [7, 8], "counter": [7, 8]}, output, workers=1)

    assert factory_calls == ["double", "double", "counter", "counter"]
    for task, points in (("double", [42, 48]), ("counter", [21, 24])):
        task_output = output / task
        summary = json.loads((task_output / "summary.json").read_text())
        assert summary["total_points"] == sum(points)
        rows = [json.loads(line) for line in (task_output / "episodes.jsonl").read_text().splitlines()]
        assert [(row["seed"], row["points"]) for row in rows] == list(zip([7, 8], points, strict=True))


@pytest.mark.parametrize(("episodes", "seed"), [(0, 42), (1, -1), (2, 2**32 - 1)])
def test_invalid_seed_sequence(episodes, seed, tmp_path):
    def open_adapter():
        pytest.fail("Invalid requests must be rejected before opening the simulator")

    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="episode count and seeds"):
        evaluate_to_directory(open_adapter, None, episodes, seed, output)
    assert not output.exists()


def test_cli_selects_adapter_and_default_controller_from_catalog(monkeypatch, tmp_path, capsys):
    def controller_factory():
        return lambda observation: [3]

    def load_counter():
        return lambda: nullcontext(CounterAdapter()), controller_factory

    output = tmp_path / "evaluation"
    monkeypatch.setitem(TASK_CATALOG, "counter", load_counter)
    monkeypatch.setattr(sys, "argv", [
        "evaluator", "--task", "counter", "--episodes", "1", "--seed", "7",
        "--output", str(output),
    ])
    main()

    summary = json.loads(capsys.readouterr().out)
    assert summary["task_id"] == "counter"
    assert summary["controller"].endswith(".controller_factory")
    assert summary["episodes"] == [{"seed": 7, "length": 1, "points": 21}]


@pytest.mark.parametrize(("group", "default"), [
    ("pymunk", 1), ("maniskill", 1), ("libero", 2), ("robosuite", 2),
])
@pytest.mark.parametrize("override", [None, 3])
def test_cli_worker_defaults_and_override(monkeypatch, tmp_path, group, default, override):
    task = next(iter(TASK_GROUPS[group]))
    def load():
        return None, None

    monkeypatch.setitem(TASK_CATALOG, task, load)
    monkeypatch.setitem(DEV_TASK_GROUPS, group, {task: load})
    counts = []

    def record_workers(*args, workers, **kwargs):
        counts.append(workers)
        return {}

    flags = [] if override is None else ["--workers", str(override)]
    for module, function, args in (
        (evaluator, "evaluate_to_directory", ["--task", task, "--output", str(tmp_path / "task")]),
        (owner_evaluation, "evaluate_tasks", [group, str(tmp_path / "group")]),
    ):
        monkeypatch.setattr(module, function, record_workers)
        monkeypatch.setattr(sys, "argv", ["evaluator", *args, *flags])
        module.main()

    expected = default if override is None else override
    assert counts == [expected, expected]


def test_cli_loads_lead_factory_and_writes_standard_artifacts(tmp_path):
    output = tmp_path / "evaluation"
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.benchmark.evaluator", "--task", "interception",
         "--controller", "worldsmith.benchmark.tasks.interception:Lead",
         "--episodes", "1", "--seed", "12", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    assert summary == json.loads((output / "summary.json").read_text())
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert summary["task_id"] == "interception-v2"
    assert summary["controller"] == "worldsmith.benchmark.tasks.interception.Lead"
    assert summary["seeds"] == [12]
    assert episode["seed"] == 12
    assert episode["steps"][0]["observation"]["goal"] == "interception"
    assert episode["length"] == len(episode["steps"])
    assert summary["captures"] == int(episode["captured"])
    assert summary["capture_rate"] == float(episode["captured"])
