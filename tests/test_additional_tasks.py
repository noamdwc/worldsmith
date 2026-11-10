import json
import math
import subprocess
import sys
from io import StringIO
from random import Random

import pytest

from autoresearch.controller import make_controller
from worldsmith.api import Action
from worldsmith.benchmark.evaluate import evaluate, evaluate_all
from worldsmith.benchmark.simulation import Entity, Simulation
from worldsmith.benchmark.tasks import COMBINED_SPLITS, TASKS, TASK_SPLITS
from worldsmith.benchmark.tasks.navigation import COLLISION_DISTANCE, Navigation
from worldsmith.benchmark.tasks.rendezvous import HOLD_STEPS, TRACKING_RADIUS, Rendezvous


@pytest.mark.parametrize("goal", TASKS)
def test_deterministic_episodes_and_goal_propagation(goal):
    task = TASKS[goal]()
    first_trace, replay_trace = StringIO(), StringIO()
    seen_goals = []

    def factory():
        controller = make_controller()

        def control(observation):
            seen_goals.append(observation.goal)
            action = controller(observation)
            assert math.isfinite(action.heading)
            assert 0 <= action.throttle <= 1
            return action

        return control

    first = evaluate(task, factory, [0, 1], first_trace)
    replay = evaluate(task, make_controller, [0, 1], replay_trace)
    assert first == replay
    assert first_trace.getvalue() == replay_trace.getvalue()
    assert set(seen_goals) == {goal}
    for line in first_trace.getvalue().splitlines():
        episode = json.loads(line)
        assert episode["initial_observation"]["goal"] == goal
        assert all(frame["observation"]["goal"] == goal for frame in episode["frames"])
        assert episode["steps"] <= task.step_limit


def test_navigation_measures_distance_and_moving_obstacle_contact():
    task = Navigation()
    world = Simulation(
        {
            "robot": Entity((0, 0), (0, 0), 0),
            "destination": Entity((0, 0), (0, 0), 0),
            "obstacle_0": Entity((COLLISION_DISTANCE + 0.05, 0), (-1, 0), 0),
            "obstacle_1": Entity((10, 0), (0, 0), 0),
        },
        "robot", 2.4, Random(0), goal="navigation",
    )
    before = task.measure(world)
    assert before["destination_distance"] == 0
    assert before["obstacle_clearance"] == pytest.approx(0.05)
    assert task.succeeded([before])
    world.step(Action(0, 0))
    contact = task.measure(world)
    assert contact["obstacle_clearance"] == pytest.approx(-0.05)
    assert not task.succeeded([before, contact, before])  # A collision cannot be undone.
    metrics = task.episode_metrics([before, contact, before], False)
    assert metrics["collided"]
    assert metrics["minimum_obstacle_clearance"] == pytest.approx(-0.05)
    assert not task.succeeded([{**before, "destination_distance": 0.36}])


def test_rendezvous_requires_consecutive_tracking_steps():
    task = Rendezvous()
    inside = {"distance": TRACKING_RADIUS}
    outside = {"distance": TRACKING_RADIUS + 0.01}
    assert not task.succeeded([inside] * HOLD_STEPS)  # Initial frame isn't a step.
    assert task.succeeded([outside] + [inside] * HOLD_STEPS)
    interrupted = [outside] + [inside] * (HOLD_STEPS - 1) + [outside, inside]
    assert not task.succeeded(interrupted)
    assert task.episode_metrics(interrupted, False)["longest_tracking_seconds"] == pytest.approx(1.9)


@pytest.mark.parametrize("goal", ["navigation", "rendezvous"])
def test_failed_tasks_run_to_their_episode_limit(goal):
    trace = StringIO()
    task = TASKS[goal]()
    metrics = evaluate(task, lambda: lambda observation: Action(0, 0), [0], trace)
    episode = json.loads(trace.getvalue())
    assert metrics["success_rate"] == 0
    assert episode["steps"] == task.step_limit
    assert not episode["success"]


def test_combined_evaluation_matches_independent_results_and_resets_controller():
    seeds = list(COMBINED_SPLITS["dev"])
    trace = StringIO()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return make_controller()

    combined = evaluate_all(factory, seeds, trace)
    assert factory_calls == combined["episodes"] == 24
    rates = []
    for goal, task_factory in TASKS.items():
        independent = evaluate(task_factory(), make_controller, seeds, StringIO())
        if goal == "interception":
            independent["success_rate"] = independent["capture_rate"]
        assert combined["tasks"][goal] == independent
        rates.append(independent["success_rate"])
    assert combined["aggregate_score"] == sum(rates) / 3
    episodes = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert {(item["initial_observation"]["goal"], item["seed"]) for item in episodes} == {
        (goal, seed) for goal in TASKS for seed in seeds
    }
    # The starter must make progress on each task, while leaving room for research.
    assert all(rate > 0 for rate in rates)


def test_development_and_heldout_seeds_are_disjoint():
    for splits in [COMBINED_SPLITS, *TASK_SPLITS.values()]:
        assert set(splits["dev"]).isdisjoint(splits["heldout"])
    assert list(TASK_SPLITS["interception"]["dev"]) == list(range(32))
    assert list(TASK_SPLITS["interception"]["heldout"]) == list(range(1000, 1032))


@pytest.mark.parametrize("task", [*TASKS, "all"])
def test_cli_can_evaluate_each_task_and_combined(tmp_path, task):
    output = tmp_path / task
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith", "evaluate", "--task", task,
         "--seed", "0", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    assert result.stdout == (output / "summary.json").read_text()
    assert summary["seeds"] == [0]
    assert summary["episodes"] == (3 if task == "all" else 1)
    if task == "all":
        assert set(summary["tasks"]) == set(TASKS)
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    expected_goals = set(TASKS) if task == "all" else {task}
    assert {episode["initial_observation"]["goal"] for episode in episodes} == expected_goals
