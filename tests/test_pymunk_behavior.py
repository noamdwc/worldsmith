import json
import math

import numpy as np
import pytest

from worldsmith.api import Action
from worldsmith.benchmark.evaluator import evaluate_to_directory
from worldsmith.benchmark.runner import ControllerError, InvalidAction
from worldsmith.benchmark.task_catalog import TASK_GROUPS


@pytest.mark.parametrize("task", TASK_GROUPS["pymunk"])
def test_planar_action_and_episode_replay(task, check_episode_order, check_native_parallel_artifacts):
    open_adapter = TASK_GROUPS["pymunk"][task]
    with open_adapter() as adapter:
        initial, _ = adapter.reset(0)
        body = adapter.world._bodies[initial.robot_id]
        start = body.position
        for action in (Action(math.inf), Action(0, -0.1), Action(0, 1.1)):
            with pytest.raises(InvalidAction):
                adapter.step(action)
            assert body.position == start

        observation, _, _, _, _ = adapter.step(Action(-math.pi / 2, 0.5))
        speed = initial.robot_max_speed / 2
        assert observation.time == pytest.approx(0.1)
        assert tuple(body.velocity) == pytest.approx((0., -speed))
        assert tuple(body.position - start) == pytest.approx((0., -speed * 0.1))
        assert observation.goal == task
        assert initial.time == 0
        adapter.step(Action(0, 0))
        assert tuple(body.velocity) == (0., 0.)
    check_episode_order(task)
    check_native_parallel_artifacts(task)


def test_numpy_real_actions_are_saved_as_numeric_commands(tmp_path):
    open_adapter = TASK_GROUPS["pymunk"]["interception"]
    output = tmp_path / "evaluation"
    evaluate_to_directory(
        open_adapter, lambda: lambda observation: Action(np.float32(0.5), np.float32(0.5)),
        1, 0, output, task="interception",
    )
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert episode["steps"][0]["action"] == {"heading": 0.5, "throttle": 0.5}


@pytest.mark.parametrize("action", [Action("north"), Action(0, "full")])
def test_malformed_action_uses_controller_error_contract(tmp_path, action):
    open_adapter = TASK_GROUPS["pymunk"]["interception"]
    with pytest.raises(ControllerError) as error:
        evaluate_to_directory(
            open_adapter, lambda: lambda observation: action, 1, 0, tmp_path / "evaluation",
            task="interception",
        )
    assert error.value.public_message == "interception, seed 0, step 1: controller returned an invalid action"
