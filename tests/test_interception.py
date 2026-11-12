import json
import subprocess
import sys
from dataclasses import replace
from io import StringIO
from random import Random

import pytest

from autoresearch.controller import make_controller
from worldsmith.api import Action, Observation
from worldsmith.benchmark.simulation import TIME_STEP, Entity, Simulation
from worldsmith.benchmark.tasks.interception import SPLITS, TARGET_NOISE, Chase, Interception, Lead


def test_seed_replays_noisy_observations():
    world = Interception().start(12)
    replay = Interception().start(12)
    observation = world.initial_observation

    assert observation == replay.initial_observation
    assert observation.entities[observation.robot_id].position != (0.0, 0.0)
    assert observation.entities["target"].covariance == ((TARGET_NOISE**2, 0.0), (0.0, TARGET_NOISE**2))
    assert world.step(Action(0.0)) == replay.step(Action(0.0))
    assert world.step(Action(0.0)).time == 2 * TIME_STEP


@pytest.mark.parametrize("action", [Action(float("nan")), Action(0.0, 1.1)])
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError, match="finite heading and throttle"):
        Interception().start(12).step(action)


def test_reference_development_benchmarks_are_unchanged(research_metrics):
    seeds = list(SPLITS["dev"])
    chase = research_metrics({"interception": Interception()}, Chase, seeds, StringIO())
    lead = research_metrics({"interception": Interception()}, Lead, seeds, StringIO())
    assert chase == {
        "episodes": 32, "captures": 24, "capture_rate": 0.75,
        "mean_closest_distance": pytest.approx(0.440274733137007),
    }
    assert lead == {
        "episodes": 32, "captures": 27, "capture_rate": 0.84375,
        "mean_closest_distance": pytest.approx(0.3012441981496445),
    }


def test_evaluator_passes_goal_to_controller_and_trace(research_metrics):
    observations = []

    def controller(observation):
        observations.append(observation)
        return Action(0.0, throttle=0.0)

    trace = StringIO()
    task = Interception()
    research_metrics({"interception": task}, lambda: controller, [12], trace)
    assert [observation.goal for observation in observations] == ["interception"] * task.step_limit

    episode = json.loads(trace.getvalue())
    assert episode["initial_observation"]["goal"] == "interception"
    assert all(frame["observation"]["goal"] == "interception" for frame in episode["frames"])


def test_simulation_preserves_the_supplied_goal():
    world = Simulation(
        {"robot": Entity((0.0, 0.0), (0.0, 0.0), 0.0)},
        "robot", 1.0, Random(0), goal="example",
    )
    assert world.initial_observation.goal == "example"
    assert world.step(Action(0.0)).goal == "example"


def test_legacy_observation_constructor_and_chase_behavior(research_metrics):
    initial = Interception().start(12).initial_observation
    legacy = Observation(initial.time, initial.robot_id, initial.entities, initial.robot_max_speed)
    assert legacy == initial

    seeds = list(SPLITS["dev"])
    chase_trace = StringIO()
    shared_trace = StringIO()
    chase = research_metrics({"interception": Interception()}, Chase, seeds, chase_trace)
    shared = research_metrics({"interception": Interception()}, make_controller, seeds, shared_trace)
    assert shared == chase
    assert shared_trace.getvalue() == chase_trace.getvalue()


def test_shared_controller_rejects_unsupported_goal():
    observation = replace(Interception().start(12).initial_observation, goal="unknown")
    with pytest.raises(ValueError, match="Unsupported goal: 'unknown'"):
        make_controller()(observation)


def test_cli_loads_factory_and_writes_run_files(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "worldsmith", "evaluate",
            "--controller", "worldsmith.benchmark.tasks.interception:Lead",
            "--seed", "12", "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )

    assert result.stdout == (output / "summary.json").read_text()
    summary = json.loads(result.stdout)
    assert summary["seeds"] == [12]
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert episode["seed"] == 12
    assert episode["steps"] == len(episode["frames"])
    assert summary["captures"] == int(episode["captured"])
