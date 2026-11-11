import json
import sys
from random import Random

import pymunk
import pytest

from worldsmith.__main__ import main
from worldsmith.api import INTERCEPTION_GOAL, Action
from worldsmith.benchmark.simulation import Entity, PhysicalEntity, Simulation
from worldsmith.benchmark.tasks.constants import TASKS
from worldsmith.benchmark.tasks.interception import Interception
from worldsmith.research_feedback import publish_feedback

COAST = Action(0, 0)
RADIUS = 0.25
MASS = 2
MOMENT = pymunk.moment_for_circle(MASS, 0, RADIUS)


def ball(position=(0, 0), velocity=(0, 0)):
    body = pymunk.Body(MASS, MOMENT, body_type=pymunk.Body.DYNAMIC)
    body.position = position
    body.velocity = velocity
    return pymunk.Circle(body, RADIUS)


def wall():
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    return pymunk.Segment(body, (1, -2), (1, 2), 0.05)


def scene(**entities):
    return Simulation(
        {"robot": Entity((-2, 0), (0, 0), 0), **entities},
        "robot", 2, Random(0), goal=INTERCEPTION_GOAL,
    )


@pytest.mark.parametrize(("elasticity", "impact_velocity"), [(0, 0), (1, -1.6)])
def test_force_drives_ball_into_static_wall(elasticity, impact_velocity):
    circle, segment = ball(), wall()
    circle.elasticity = elasticity
    segment.elasticity = 1
    world = scene(ball=PhysicalEntity(circle.body), wall=PhysicalEntity(segment.body))

    for _ in range(8):
        circle.body.apply_force_at_local_point((4, 0))
        world.step(COAST)
    before = world.ground_truth()
    assert before["ball"]["position"] == pytest.approx((0.56, 0))
    assert before["ball"]["velocity"] == pytest.approx((1.6, 0))

    circle.body.apply_force_at_local_point((4, 0))
    world.step(COAST)
    impact = world.ground_truth()
    assert impact["ball"]["position"] == pytest.approx((0.72, 0))
    assert impact["ball"]["velocity"] == pytest.approx((impact_velocity, 0))
    for _ in range(10):
        world.step(COAST)
        assert world.position("ball")[0] <= 0.72 + 1e-9
    assert world.ground_truth()["wall"] == before["wall"]


def test_friction_reduces_sliding_and_spins_ball_at_contact():
    def impact(friction):
        circle, segment = ball(velocity=(2, 1)), wall()
        circle.friction = friction
        segment.friction = 1
        world = scene(ball=PhysicalEntity(circle.body), wall=PhysicalEntity(segment.body))
        for _ in range(4):
            world.step(COAST)
        return world.ground_truth()["ball"]

    smooth, rough = impact(0), impact(1)
    assert smooth["velocity"] == pytest.approx((0, 1))
    assert smooth["angular_velocity"] == pytest.approx(0)
    assert 0 < rough["velocity"][1] < smooth["velocity"][1]
    assert abs(rough["angular_velocity"]) > 0.1


def test_dynamic_circles_exchange_momentum_after_impulse():
    first, second = ball(), ball((0.75, 0))
    first.elasticity = second.elasticity = 1
    world = scene(first=PhysicalEntity(first.body), second=PhysicalEntity(second.body))

    first.body.apply_impulse_at_local_point((MASS, 0))
    assert first.body.velocity == pytest.approx((1, 0))
    for _ in range(3):
        world.step(COAST)
    state = world.ground_truth()
    assert state["first"]["velocity"] == pytest.approx((0, 0))
    assert state["second"]["velocity"] == pytest.approx((1, 0))


def test_kinematic_robot_pushes_dynamic_body_at_commanded_speed():
    robot = pymunk.Circle(pymunk.Body(body_type=pymunk.Body.KINEMATIC), RADIUS)
    circle = ball((1, 0))
    world = scene(robot=PhysicalEntity(robot.body, observable=True), ball=PhysicalEntity(circle.body))

    for _ in range(4):
        observation = world.step(Action(0))
    assert observation.entities["robot"].position == pytest.approx((0.8, 0))
    assert world.position("ball")[0] > 1
    assert circle.body.velocity == pytest.approx((2, 0))
    assert robot.body.velocity == pytest.approx((2, 0))


def test_shape_free_bodies_can_be_hidden_or_observed_with_or_without_noise():
    exact_body = pymunk.Body(MASS, MOMENT)
    exact_body.velocity = (1, 0)
    world = scene(
        hidden=PhysicalEntity(pymunk.Body(MASS, MOMENT), position_noise=0.25),
        exact=PhysicalEntity(exact_body, observable=True),
        noisy=PhysicalEntity(pymunk.Body(MASS, MOMENT), observable=True, position_noise=0.25),
    )

    initial = world.initial_observation
    stepped = world.step(COAST)
    assert "hidden" in world.ground_truth()
    for observation, exact_position in ((initial, (0, 0)), (stepped, (0.1, 0))):
        assert set(observation.entities) == {"robot", "exact", "noisy"}
        assert observation.entities["exact"].position == exact_position
        assert observation.entities["exact"].covariance == ((0, 0), (0, 0))
        assert observation.entities["noisy"].position != (0, 0)
        assert observation.entities["noisy"].covariance == ((0.0625, 0), (0, 0.0625))


def test_physics_stays_in_owner_trace(tmp_path, monkeypatch):
    class PhysicalInterception(Interception):
        step_limit = 1

        def start(self, seed):
            circle, segment = ball(velocity=(0.5, 0)), wall()
            circle.friction = 0.7
            circle.elasticity = 0.3
            return scene(
                target=PhysicalEntity(circle.body, observable=True),
                wall=PhysicalEntity(segment.body),
            )

    monkeypatch.setitem(TASKS, INTERCEPTION_GOAL, PhysicalInterception)
    owner_dir = tmp_path / "owner"
    monkeypatch.setattr(sys, "argv", ["worldsmith", "evaluate", "--task", "all", "--output", str(owner_dir)])
    main()
    public = publish_feedback(owner_dir, tmp_path / "public").episode(0, INTERCEPTION_GOAL)
    with (owner_dir / "episodes.jsonl").open() as trace:
        owner = json.loads(next(trace))

    assert owner["initial_state"]["wall"]["body_type"] == "static"
    assert owner["initial_state"]["target"]["position"] == [0, 0]
    assert owner["frames"][0]["state"]["target"] == {
        "position": [0.05, 0], "velocity": [0.5, 0],
        "body_type": "dynamic", "mass": MASS, "moment": MOMENT,
        "angle": 0, "angular_velocity": 0,
        "shapes": [{"type": "Circle", "friction": 0.7, "elasticity": 0.3}],
    }
    assert public["initial_observation"] == owner["initial_observation"]
    assert set(public) == {"task", "seed", "steps", "captured", "initial_observation", "frames"}
    assert set(public["frames"][0]) == {"action", "observation"}
    assert public["frames"][0]["observation"] == owner["frames"][0]["observation"]
    for observation in (public["initial_observation"], public["frames"][0]["observation"]):
        assert set(observation["entities"]) == {"robot", "target"}
        assert set(observation["entities"]["target"]) == {"position", "covariance"}
