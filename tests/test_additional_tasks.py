import json
import math
import subprocess
import sys
from functools import partial
from random import Random
from time import monotonic

import pymunk
import pytest

from autoresearch import controller as research_controller
from autoresearch.controller import make_controller
from worldsmith.api import (
    BOUNCING_INTERCEPTION_GOAL, INTERCEPTION_GOAL, OBJECT_PUSHING_GOAL,
    Action, Observation, PositionMeasurement,
)
from worldsmith.benchmark.simulation import Entity, Simulation
from worldsmith.benchmark.task_catalog import DEV_SEEDS, DEV_TASK_GROUPS, VALIDATION_SEEDS
from worldsmith.benchmark.tasks.constants import TASKS
from worldsmith.benchmark.tasks.navigation import COLLISION_DISTANCE, Navigation
from worldsmith.benchmark.tasks.rendezvous import HOLD_STEPS, TRACKING_RADIUS, Rendezvous
from worldsmith.evaluation.evaluation_utils import evaluate_tasks
from worldsmith.research_feedback import publish_feedback


@pytest.mark.parametrize("goal", TASKS)
def test_deterministic_episodes_and_goal_propagation(goal, evaluate_pymunk):
    task = TASKS[goal]()
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

    first, first_trace = evaluate_pymunk(goal, factory, [0, 1])
    replay, replay_trace = evaluate_pymunk(goal, make_controller, [0, 1])
    assert first == {**replay, "controller": first["controller"]}
    assert first_trace == replay_trace
    assert set(seen_goals) == {goal}
    for episode in first_trace:
        assert all(step["observation"]["goal"] == goal for step in episode["steps"])
        assert episode["final_observation"]["goal"] == goal
        assert episode["length"] <= task.step_limit


@pytest.mark.parametrize("goal", TASKS)
def test_starter_uses_only_the_current_observation(goal):
    world = TASKS[goal]().start(0)
    controller = make_controller()
    observation = world.initial_observation
    for _ in range(25):
        assert controller(observation) == make_controller()(observation)
        observation = world.step(Action(0.0))


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


@pytest.mark.parametrize("goal", ["navigation", "rendezvous", OBJECT_PUSHING_GOAL])
def test_failed_tasks_run_to_their_episode_limit(goal, evaluate_pymunk):
    summary, (episode,) = evaluate_pymunk(goal, lambda: lambda observation: Action(0, 0), [0])
    assert summary["success_rate"] == 0
    assert episode["length"] == TASKS[goal].step_limit
    assert not episode["success"]


def test_group_evaluation_matches_independent_results_and_resets_controller(tmp_path, monkeypatch, evaluate_pymunk):
    seeds = list(DEV_SEEDS)
    factory_calls = []

    def factory(*, task):
        factory_calls.append(task)
        return make_controller(task=task)

    monkeypatch.setattr(research_controller, "make_controller", factory)
    owner_results = tmp_path / "owner"
    evaluate_tasks(dict.fromkeys(TASKS, seeds), owner_results / "pymunk", workers=1)
    assert factory_calls == [goal for goal in TASKS for _ in seeds]

    rates = {}
    for goal in TASKS:
        independent, independent_trace = evaluate_pymunk(goal, partial(make_controller, task=goal), seeds)
        task_results = owner_results / "pymunk" / goal
        assert json.loads((task_results / "summary.json").read_text()) == independent
        episodes = [json.loads(line) for line in (task_results / "episodes.jsonl").read_text().splitlines()]
        assert episodes == independent_trace
        assert [episode["seed"] for episode in episodes] == seeds
        assert all(episode["steps"][0]["observation"]["goal"] == goal for episode in episodes)
        rate = "capture_rate" if goal in (INTERCEPTION_GOAL, BOUNCING_INTERCEPTION_GOAL) else "success_rate"
        rates[goal] = independent[rate]

    for goal in (INTERCEPTION_GOAL, "navigation", "rendezvous"):
        assert 0 < rates[goal] < 1

    dev_tasks = list(DEV_TASK_GROUPS["pymunk"])
    feedback = publish_feedback(
        owner_results, tmp_path / "feedback", benchmark="test-pymunk",
        tasks_by_group={"pymunk": dev_tasks}, seeds=seeds, controller_sha256="",
        started=monotonic(), retained=None,
    )
    assert {goal: task["success_rate"] for goal, task in feedback.summary["tasks"].items()} == {
        goal: rates[goal] for goal in dev_tasks
    }
    assert feedback.summary["aggregate_score"] == sum(rates[goal] for goal in dev_tasks) / len(dev_tasks)
    assert [(episode["task"], episode["seed"]) for episode in feedback.episodes] == [
        (goal, seed) for goal in dev_tasks for seed in seeds
    ]


@pytest.mark.parametrize("seeds", [list(DEV_SEEDS), list(VALIDATION_SEEDS)], ids=["development", "validation"])
def test_metrics_and_traces_match_kinematic_reference(seeds, monkeypatch, evaluate_pymunk):
    # Keep the pre-physics integration and sensing as a reference so exact trace
    # equality can be checked on the current platform without large fixtures.
    class KinematicReference:
        def __init__(self, entities, robot_id, robot_max_speed, rng, *, goal):
            self.space = pymunk.Space()
            self.bodies = {}
            for name, entity in entities.items():
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = entity.position
                body.velocity = entity.velocity
                self.space.add(body)
                self.bodies[name] = body
            self.entities = entities
            self.robot_id = robot_id
            self.robot_max_speed = robot_max_speed
            self.goal = goal
            self.rng = rng
            self.steps = 0
            self.initial_observation = self.observe()

        def position(self, name):
            return tuple(self.bodies[name].position)

        def ground_truth(self):
            return {
                name: {"position": tuple(body.position), "velocity": tuple(body.velocity)}
                for name, body in self.bodies.items()
            }

        def step(self, action):
            self.bodies[self.robot_id].velocity = pymunk.Vec2d(
                self.robot_max_speed * action.throttle, 0
            ).rotated(action.heading)
            self.space.step(0.1)
            self.steps += 1
            return self.observe()

        def observe(self):
            measurements = {}
            for name, entity in self.entities.items():
                x, y = self.position(name)
                sigma = entity.position_noise
                measurements[name] = PositionMeasurement(
                    (self.rng.gauss(x, sigma), self.rng.gauss(y, sigma)),
                    ((sigma**2, 0.0), (0.0, sigma**2)),
                )
            return Observation(
                time=self.steps * 0.1, robot_id=self.robot_id, entities=measurements,
                robot_max_speed=self.robot_max_speed, goal=self.goal,
            )

    for goal in (INTERCEPTION_GOAL, "navigation", "rendezvous"):
        actual = evaluate_pymunk(goal, make_controller, seeds)
        task_module = sys.modules[TASKS[goal].__module__]
        with monkeypatch.context() as patch:
            patch.setattr(task_module, "Simulation", KinematicReference)
            reference = evaluate_pymunk(goal, make_controller, seeds)
        assert actual == reference


@pytest.mark.parametrize("task", TASKS)
def test_single_task_cli_evaluates_each_pymunk_task(tmp_path, task):
    output = tmp_path / task
    result = subprocess.run(
        [sys.executable, "-m", "worldsmith.benchmark.evaluator", "--task", task,
         "--seed", "0", "--episodes", "1", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    assert summary == json.loads((output / "summary.json").read_text())
    assert summary["seeds"] == [0]
    assert len(summary["episodes"]) == 1
    assert summary["objective"] == TASKS[task].objective
    (line,) = (output / "episodes.jsonl").read_text().splitlines()
    episode = json.loads(line)
    assert episode["seed"] == 0
    assert episode["steps"][0]["observation"]["goal"] == task
