from dataclasses import replace
from random import Random

import pytest

from autoresearch.controller import make_controller
from worldsmith.api import INTERCEPTION_GOAL, Action, Observation
from worldsmith.benchmark.simulation import TIME_STEP, Entity, Simulation
from worldsmith.benchmark.tasks.interception import TARGET_NOISE, Chase, Interception, Lead


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


def test_reference_controller_metrics_are_unchanged(evaluate_pymunk):
    # Keep the original sample behind these metric expectations.
    seeds = list(range(32))
    chase, _ = evaluate_pymunk(INTERCEPTION_GOAL, Chase, seeds)
    lead, _ = evaluate_pymunk(INTERCEPTION_GOAL, Lead, seeds)
    assert chase["seeds"] == lead["seeds"] == seeds
    assert len(chase["episodes"]) == len(lead["episodes"]) == 32
    assert chase["captures"] == 24
    assert chase["capture_rate"] == 0.75
    assert chase["mean_closest_distance"] == pytest.approx(0.440274733137007)
    assert lead["captures"] == 27
    assert lead["capture_rate"] == 0.84375
    assert lead["mean_closest_distance"] == pytest.approx(0.3012441981496445)


def test_evaluator_passes_goal_to_controller_and_trace(evaluate_pymunk):
    observations = []

    def controller(observation):
        observations.append(observation)
        return Action(0.0, throttle=0.0)

    _, (episode,) = evaluate_pymunk(INTERCEPTION_GOAL, lambda: controller, [12])
    assert [observation.goal for observation in observations] == [INTERCEPTION_GOAL] * Interception.step_limit
    assert all(step["observation"]["goal"] == INTERCEPTION_GOAL for step in episode["steps"])
    assert episode["final_observation"]["goal"] == INTERCEPTION_GOAL


def test_simulation_preserves_the_supplied_goal():
    world = Simulation(
        {"robot": Entity((0.0, 0.0), (0.0, 0.0), 0.0)},
        "robot", 1.0, Random(0), goal="example",
    )
    assert world.initial_observation.goal == "example"
    assert world.step(Action(0.0)).goal == "example"


def test_legacy_observation_constructor_and_chase_behavior(evaluate_pymunk):
    initial = Interception().start(12).initial_observation
    legacy = Observation(initial.time, initial.robot_id, initial.entities, initial.robot_max_speed)
    assert legacy == initial

    seeds = list(range(32))
    chase, chase_trace = evaluate_pymunk(INTERCEPTION_GOAL, Chase, seeds)
    shared, shared_trace = evaluate_pymunk(INTERCEPTION_GOAL, make_controller, seeds)
    assert shared == {**chase, "controller": shared["controller"]}
    assert shared_trace == chase_trace


def test_shared_controller_rejects_unsupported_goal():
    observation = replace(Interception().start(12).initial_observation, goal="unknown")
    with pytest.raises(ValueError, match="Unsupported goal: 'unknown'"):
        make_controller()(observation)
