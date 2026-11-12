import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext

import pytest

from worldsmith.benchmark.evaluator import evaluate, evaluate_to_directory, main
from worldsmith.benchmark.task_catalog import TASK_CATALOG


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


def test_task_collection_uses_shared_factory_and_benchmark_summary():
    class DoubleScoreAdapter(CounterAdapter):
        def score(self, episode):
            return {"points": 2 * super().score(episode)["points"]}

    factory_calls = []
    recorded = []

    def factory():
        factory_calls.append("created")
        return lambda observation: [3]

    def record(episode, metrics):
        recorded.append((episode["seed"], metrics["points"]))

    def summarize(results):
        assert list(results) == ["double", "counter"]
        double_points = results["double"].metrics["total_points"]
        counter_points = results["counter"].metrics["total_points"]
        return {"points_difference": double_points - counter_points}

    summary = evaluate(
        {"double": DoubleScoreAdapter(), "counter": CounterAdapter()},
        factory, [7, 8], record, summarize=summarize,
    )

    assert len(factory_calls) == 4
    assert recorded == [(7, 42), (8, 48), (7, 21), (8, 24)]
    assert summary == {"points_difference": 45}


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


def test_v1_task_uses_standard_artifacts_from_unified_cli(tmp_path):
    output = tmp_path / "evaluation"
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.benchmark.evaluator", "--task", "interception",
         "--controller", "worldsmith.benchmark.tasks.interception:Lead",
         "--episodes", "1", "--seed", "0", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert summary["task_id"] == "interception-v2"
    assert summary["controller"] == "worldsmith.benchmark.tasks.interception.Lead"
    assert summary["seeds"] == [0]
    assert episode["steps"][0]["observation"]["goal"] == "interception"
    assert episode["length"] == len(episode["steps"])
    assert summary["capture_rate"] == float(episode["captured"])
