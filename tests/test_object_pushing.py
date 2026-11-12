import json
import math
from dataclasses import asdict
from io import StringIO

import pytest

from autoresearch.controller import make_controller
from worldsmith.api import Action
from worldsmith.benchmark.evaluator import evaluate_v1
from worldsmith.benchmark.simulation import TIME_STEP
from worldsmith.benchmark.tasks.constants import COMBINED_SPLITS
from worldsmith.benchmark.tasks.object_pushing import (
    DESTINATION_RADIUS, SETTLED_SPEED, SETTLE_STEPS, ObjectPushing,
)


def test_contact_accelerates_the_object_and_free_motion_decays():
    task = ObjectPushing()
    world = task.start(12)
    start = world.ground_truth()
    rx, ry = start["robot"]["position"]
    ox, oy = start["object"]["position"]
    heading = math.atan2(oy - ry, ox - rx)

    for _ in range(20):
        world.step(Action(heading))
        contact = task.measure(world)
        if contact["contact_impulse"] > 0:
            break
        assert world.position("object") == (ox, oy)
        assert contact["object_speed"] == 0
    assert contact["contact_impulse"] > 0
    assert contact["object_speed"] == pytest.approx(contact["contact_impulse"] / start["object"]["mass"])

    world.step(Action(heading + math.pi))
    coast = task.measure(world)
    assert coast["contact_impulse"] == 0
    retention = start["object"]["velocity_retention_per_second"] ** TIME_STEP
    assert 0 < retention < 1
    assert coast["object_speed"] == pytest.approx(contact["object_speed"] * retention)
    assert math.dist((ox, oy), world.position("object")) > 0


def test_success_requires_consecutive_slow_steps_near_destination():
    task = ObjectPushing()
    settled = {"object_distance": DESTINATION_RADIUS, "object_speed": SETTLED_SPEED}
    fast = {**settled, "object_speed": SETTLED_SPEED + 0.01}
    outside = {**settled, "object_distance": DESTINATION_RADIUS + 0.01}
    assert not task.succeeded([settled] * SETTLE_STEPS)
    assert not task.succeeded([outside] + [fast] * SETTLE_STEPS)
    assert not task.succeeded([outside, fast] + [settled] * (SETTLE_STEPS - 1))
    assert not task.succeeded([fast, outside] + [settled] * (SETTLE_STEPS - 1))
    assert task.succeeded([fast] + [settled] * SETTLE_STEPS)


def test_driving_to_destination_leaves_the_object_at_rest():
    task = ObjectPushing()
    world = task.start(12)
    object_start = world.position("object")
    dx, dy = world.position("destination")
    max_step = world.initial_observation.robot_max_speed * TIME_STEP
    measurements = [task.measure(world)]
    for _ in range(task.step_limit):
        rx, ry = world.position("robot")
        distance = math.hypot(dx - rx, dy - ry)
        world.step(Action(math.atan2(dy - ry, dx - rx), min(1, distance / max_step)))
        measurements.append(task.measure(world))
    assert math.dist(world.position("robot"), (dx, dy)) < DESTINATION_RADIUS
    assert world.position("object") == object_start
    assert not any(frame["contact_impulse"] for frame in measurements)
    assert not task.succeeded(measurements)


def test_scene_varies_across_seeds_and_exposes_only_position_measurements():
    task = ObjectPushing()
    world = task.start(12)
    state = world.ground_truth()
    assert state == task.start(12).ground_truth()
    another = task.start(31).ground_truth()
    for name in ("robot", "object", "destination"):
        assert state[name]["position"] != another[name]["position"]
    movable = state["object"]
    assert movable["body_type"] == "dynamic"
    assert movable["velocity"] == (0, 0)
    assert movable["shapes"][0]["type"] == "Circle"
    for key in ("mass", "velocity_retention_per_second", "shapes"):
        assert movable[key] != another["object"][key]

    observation = asdict(world.initial_observation)
    assert set(observation) == {"time", "goal", "robot_id", "robot_max_speed", "entities"}
    assert set(observation["entities"]) == set(state)
    for name in ("robot", "object"):
        measurement = observation["entities"][name]
        assert set(measurement) == {"position", "covariance"}
        assert measurement["position"] != state[name]["position"]
        assert measurement["covariance"][0][0] > 0
    assert observation["entities"]["destination"] == {
        "position": state["destination"]["position"], "covariance": ((0, 0), (0, 0)),
    }


def test_successful_starter_episodes_move_and_settle_via_contact():
    trace = StringIO()
    task = ObjectPushing()
    metrics = evaluate_v1(task, make_controller, list(COMBINED_SPLITS["dev"]), trace)
    assert 0 < metrics["success_rate"] < 1
    for line in trace.getvalue().splitlines():
        episode = json.loads(line)
        if episode["success"]:
            assert episode["contact_seconds"] > 0
            for frame in episode["frames"][-SETTLE_STEPS:]:
                assert frame["object_distance"] <= DESTINATION_RADIUS
                assert frame["object_speed"] <= SETTLED_SPEED
