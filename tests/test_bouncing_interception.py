import json
import math
from io import StringIO

import pytest

from worldsmith.api import Action
from worldsmith.benchmark.evaluate import evaluate
from worldsmith.benchmark.tasks.bouncing_interception import BouncingInterception
from worldsmith.benchmark.tasks.interception import CAPTURE_RADIUS


@pytest.mark.parametrize("seed", [12, 31, 1020])
def test_target_bounces_off_walls_without_escaping(seed):
    task = BouncingInterception()
    world = task.start(seed)
    walls = world.initial_observation.entities
    left = walls["wall_left"].position[0]
    right = walls["wall_right"].position[0]
    bottom = walls["wall_bottom"].position[1]
    top = walls["wall_top"].position[1]
    initial_velocity = world.ground_truth()["target"]["velocity"]
    vx, vy = initial_velocity
    reversed_x = False

    for _ in range(task.step_limit):
        world.step(Action(0, 0))
        x, y = world.position("target")
        assert left < x < right
        assert bottom < y < top
        next_vx, next_vy = world.ground_truth()["target"]["velocity"]
        assert math.hypot(next_vx, next_vy) <= math.hypot(vx, vy) + 1e-9
        reversed_x |= vx * next_vx < 0
        vx, vy = next_vx, next_vy

    assert reversed_x
    assert math.hypot(vx, vy) < math.hypot(*initial_velocity)


def test_seed_varies_target_and_wall_properties():
    task = BouncingInterception()
    first, second = (task.start(seed).ground_truth() for seed in (12, 31))
    for name in ("position", "velocity"):
        assert first["target"][name] != second["target"][name]
    assert first["wall_left"]["position"] != second["wall_left"]["position"]
    assert first["wall_left"]["shapes"][0]["elasticity"] != second["wall_left"]["shapes"][0]["elasticity"]


def test_capture_boundary_and_episode_limit():
    task = BouncingInterception()
    assert task.succeeded([{"distance": CAPTURE_RADIUS}])
    assert not task.succeeded([{"distance": CAPTURE_RADIUS + 0.01}])

    trace = StringIO()
    metrics = evaluate(task, lambda: lambda observation: Action(0, 0), [12], trace)
    assert metrics["capture_rate"] == 0
    episode = json.loads(trace.getvalue())
    assert not episode["captured"]
    assert episode["steps"] == task.step_limit
