import json
from contextlib import nullcontext
from random import Random

import pymunk
import pytest

from worldsmith.api import INTERCEPTION_GOAL, Action
from worldsmith.benchmark.evaluator import evaluate_to_directory
from worldsmith.benchmark.pymunk_adapter import PymunkAdapter
from worldsmith.benchmark.simulation import Entity, PhysicalEntity, Simulation
from worldsmith.benchmark.tasks.interception import Chase, Interception
from worldsmith.research_feedback import public_episode

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


@pytest.mark.parametrize("layout", ["one_wall", "two_walls", "two_shapes_on_one_wall"])
def test_pairwise_contact_impulse(layout):
    target = pymunk.Body(MASS, MOMENT)
    target.position = (0.65, 0.65)
    target.velocity = (1, 1)
    circle = pymunk.Circle(target, RADIUS)
    side = pymunk.Body(body_type=pymunk.Body.STATIC)
    vertical = pymunk.Segment(side, (1, -2), (1, 2), 0.05)
    circle.elasticity = vertical.elasticity = 1
    entities = {"target": PhysicalEntity(target), "side": PhysicalEntity(side)}
    expected = {"side": 2 * MASS}

    if layout != "one_wall":
        top = side
        if layout == "two_walls":
            top = pymunk.Body(body_type=pymunk.Body.STATIC)
            entities["top"] = PhysicalEntity(top)
            expected["top"] = 2 * MASS
        else:
            expected["side"] += 2 * MASS
        horizontal = pymunk.Segment(top, (-2, 1), (2, 1), 0.05)
        horizontal.elasticity = 1

    world = scene(**entities)
    assert world.contact_impulse("target", "side") == 0
    world.step(COAST)
    for wall_id, expected_impulse in expected.items():
        impulse = world.contact_impulse("target", wall_id)
        assert impulse == pytest.approx(expected_impulse)
        assert world.contact_impulse(wall_id, "target") == impulse
    assert world.contact_impulse("target") == pytest.approx(sum(expected.values()))
    assert world.contact_impulse("target", "robot") == 0
    assert world.contact_impulse("target", "target") == 0

    world.step(COAST)
    for wall_id in expected:
        assert world.contact_impulse("target", wall_id) == 0
    assert world.contact_impulse("target") == 0


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


def test_physics_stays_in_owner_trace(tmp_path):
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

    owner_dir = tmp_path / "owner"
    evaluate_to_directory(
        lambda: nullcontext(PymunkAdapter(PhysicalInterception())), Chase, 1, 0, owner_dir,
    )
    with (owner_dir / "episodes.jsonl").open() as trace:
        owner = json.loads(next(trace))
    public = public_episode(INTERCEPTION_GOAL, "pymunk", owner)

    assert owner["reset_info"]["state"]["wall"]["body_type"] == "static"
    assert owner["reset_info"]["state"]["target"]["position"] == [0, 0]
    assert owner["steps"][0]["info"]["state"]["target"] == {
        "position": [0.05, 0], "velocity": [0.5, 0],
        "body_type": "dynamic", "mass": MASS, "moment": MOMENT,
        "angle": 0, "angular_velocity": 0,
        "shapes": [{"type": "Circle", "friction": 0.7, "elasticity": 0.3}],
    }
    assert public["initial_observation"] == owner["steps"][0]["observation"]
    assert set(public) == {
        "task", "seed", "steps", "captured", "success", "terminated", "truncated",
        "initial_observation", "frames", "final_observation",
    }
    assert set(public["frames"][0]) == {"action", "observation"}
    assert public["frames"][0]["observation"] == owner["steps"][0]["observation"]
    assert public["final_observation"] == owner["final_observation"]
    for observation in (public["initial_observation"], public["final_observation"]):
        assert set(observation["entities"]) == {"robot", "target"}
        assert set(observation["entities"]["target"]) == {"position", "covariance"}
