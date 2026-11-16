"""Opt-in integration checks for the native macOS ManiSkill tasks."""

import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import sapien
from scipy.spatial.transform import Rotation

from worldsmith.benchmark.evaluator import evaluate_to_directory
from worldsmith.benchmark.results import snapshot
from worldsmith.benchmark.runner import ControllerError, run_scored_episode
from worldsmith.benchmark.task_catalog import TASK_CATALOG, TASK_GROUPS
from worldsmith.research_baseline.controller import make_controller

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("pick_cube", "PickCube-v1", "panda", "pd_ee_delta_pos", 4, 50, "obj_pose", "is_grasped"),
    ("stack_cube", "StackCube-v1", "panda_wristcam", "pd_ee_delta_pos", 4, 50,
     "cubeA_pose", "is_cubeA_on_cubeB"),
    ("peg_insertion_side", "PegInsertionSide-v1", "panda_wristcam", "pd_ee_delta_pose", 7, 100,
     "peg_pose", "peg_head_pos_at_hole"),
    ("pull_cube", "PullCube-v1", "panda", "pd_ee_delta_pos", 4, 50, "obj_pose", "success"),
    ("push_t", "PushT-v1", "panda_stick", "pd_ee_delta_pos", 3, 100, "obj_pose", "success"),
]


def test_pick_cube_reset_discards_previous_grasp_and_rollout():
    task = "pick_cube"
    open_adapter, _ = TASK_CATALOG[task]()
    factory = partial(make_controller, task=task)
    with open_adapter() as adapter:
        first, expected = run_scored_episode(adapter, factory, 1, task)
        prior, prior_score = run_scored_episode(adapter, factory, 0, task)
        assert prior_score["success"]
        assert prior["steps"][-1]["info"]["is_grasped"] == [True]

        repeated, actual = run_scored_episode(adapter, factory, 1, task)
        assert repeated["steps"][0]["observation"]["extra"]["is_grasped"] == [False]
        assert actual == expected
        assert repeated == first


def test_pick_cube_parallel_artifacts(check_native_parallel_artifacts):
    check_native_parallel_artifacts("pick_cube")


@pytest.mark.parametrize(
    "task, task_id, robot, control_mode, action_size, limit, object_pose, native_metric",
    TASKS,
    ids=[task[0] for task in TASKS],
)
def test_native_manipulation(
    task, task_id, robot, control_mode, action_size, limit, object_pose, native_metric, tmp_path
):
    open_adapter, baseline = TASK_CATALOG[task]()
    with open_adapter() as adapter:
        env = adapter.env
        native = env.unwrapped
        assert native.device.type == "cpu"
        assert control_mode in native.agent.supported_control_modes

        observation, info = adapter.reset(42)
        initial = snapshot(observation)
        assert info["elapsed_steps"].item() == 0
        assert observation["agent"]["qpos"].shape[0] == 1
        assert observation["extra"]["tcp_pose"].shape == (1, 7)
        assert observation["extra"][object_pose].shape == (1, 7)

        action = baseline()(observation)
        assert action.shape == (action_size,)
        assert action.dtype == np.float32
        assert adapter.encode_action(action) == action.tolist()
        for invalid in (
            action.tolist(),
            np.zeros(action_size + 1, dtype=np.float32),
            np.zeros(action_size, dtype=np.float64),
            np.full(action_size, 1.01, dtype=np.float32),
            np.full(action_size, np.nan, dtype=np.float32),
        ):
            with pytest.raises(ValueError, match="Controller action"):
                adapter.encode_action(invalid)

        observation, reward, terminated, truncated, info = adapter.step(action)
        assert snapshot(observation) != initial
        assert isinstance(reward, float)
        assert terminated is bool(info["success"])
        assert not truncated
        assert snapshot(info[native_metric]) == snapshot(native.evaluate()[native_metric])
        other, _ = adapter.reset(43)
        assert snapshot(other["extra"][object_pose]) != initial["extra"][object_pose]
        repeated, _ = adapter.reset(42)
        assert snapshot(repeated) == initial

    assert native.scene is None

    # Run the public CLI with an injected policy that holds position.
    (tmp_path / "hold_policy.py").write_text(
        "import numpy as np\n\n"
        "def make_controller():\n"
        f"    return lambda observation: np.zeros({action_size}, dtype=np.float32)\n"
    )
    output = tmp_path / "evaluation"
    subprocess.run(
        [sys.executable, "-m", "worldsmith.benchmark.evaluator", "--task", task,
         "--controller", "hold_policy:make_controller", "--episodes", "1", "--seed", "42",
         "--output", str(output)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
    )
    summary = json.loads((output / "summary.json").read_text())
    traces = (output / "episodes.jsonl").read_text().splitlines()
    assert len(traces) == 1
    episode = json.loads(traces[0])

    assert summary["task_id"] == task_id
    assert summary["task_version"] == 1
    assert summary["robot"] == robot
    assert summary["obs_mode"] == "state_dict"
    assert summary["action_mode"] == control_mode
    assert summary["reward_mode"] == "normalized_dense"
    assert summary["sim_backend"] == "physx_cpu"
    assert summary["render_backend"] == "sapien_cpu"
    assert summary["render_mode"] is None
    assert summary["reconfiguration_freq"] == 1
    assert summary["max_episode_steps"] == limit
    assert summary["controller"] == "hold_policy.make_controller"
    assert summary["seeds"] == [42]
    assert summary["episodes"] == [{"seed": 42, "length": limit, "success": False}]
    assert summary["success_rate"] == 0.0

    steps = episode["steps"]
    assert episode["length"] == len(steps) == limit
    assert steps[0]["observation"] == initial
    assert all(step["action"] == [0.0] * action_size for step in steps)
    assert all(not step["terminated"] and not step["truncated"] for step in steps[:-1])
    final = steps[-1]
    assert final["info"]["elapsed_steps"] == [limit]
    assert not final["terminated"] and final["truncated"]
    assert episode["success"] is final["info"]["success"][0] is False
    assert native_metric in final["info"]
    assert episode["final_observation"]["extra"][object_pose]


@pytest.mark.parametrize("task", TASK_GROUPS["maniskill"])
def test_shared_starter_with_native_observations(task, tmp_path):
    open_adapter, _ = TASK_CATALOG[task]()
    output = tmp_path / "run"
    summary = evaluate_to_directory(
        open_adapter, partial(make_controller, task=task), 1, 0, output, task=task,
    )
    episode = json.loads((output / "episodes.jsonl").read_text())
    final_step = episode["steps"][-1]
    assert final_step["terminated"] or final_step["truncated"]
    assert episode["success"] is final_step["info"]["success"][0]
    assert summary["success_rate"] == float(episode["success"])
    assert set(episode["steps"][0]["observation"]) == {"agent", "extra"}
    assert len(final_step["action"]) == summary["action_shape"][0]


def test_invalid_shared_controller_action_fails_native_evaluation(tmp_path):
    open_adapter, _ = TASK_CATALOG["push_cube"]()
    with pytest.raises(ControllerError, match="push_cube, seed 0, step 1: Controller action"):
        evaluate_to_directory(
            open_adapter, lambda: lambda observation: np.full(4, np.nan, dtype=np.float32),
            1, 0, tmp_path / "run", task="push_cube",
        )


@pytest.mark.parametrize(("task", "pose_field", "actor"), [
    ("push_cube", "obj_pose", "obj"),
    ("pick_cube", "obj_pose", "cube"),
    ("stack_cube", "cubeA_pose", "cubeA"),
    ("peg_insertion_side", "peg_pose", "peg"),
    ("pull_cube", "obj_pose", "obj"),
    ("push_t", "obj_pose", "tee"),
])
def test_native_observation_and_delta_commands(
    task, pose_field, actor, check_episode_order, check_native_parallel_artifacts,
):
    open_adapter, _ = TASK_CATALOG[task]()
    with open_adapter() as adapter:
        positions = []
        fingers = []
        for sign in (1, -1):
            before, _ = adapter.reset(0)
            native = adapter.env.unwrapped
            arm = native.agent.controller.controllers["arm"]
            action = np.zeros(adapter.env.action_space.shape, dtype=np.float32)
            action[0] = sign * 0.5
            if task != "push_t":
                action[-1] = sign
            if task == "peg_insertion_side":
                action[3:5] = 1
            for _ in range(3):
                previous = arm.ee_pose_at_base.raw_pose[0].numpy()
                adapter.encode_action(action)
                after, _, terminated, truncated, info = adapter.step(action)
                goal = arm._target_pose.raw_pose[0].numpy()
                np.testing.assert_allclose(goal[:3] - previous[:3], [sign * 0.05, 0, 0], atol=1e-6)
                delta = Rotation.from_quat(goal[3:], scalar_first=True) * Rotation.from_quat(
                    previous[3:], scalar_first=True,
                ).inv()
                expected_rotation = (
                    Rotation.from_euler("XYZ", [-0.1 / np.sqrt(2), -0.1 / np.sqrt(2), 0])
                    if task == "peg_insertion_side" else Rotation.identity()
                )
                np.testing.assert_allclose(delta.as_matrix(), expected_rotation.as_matrix(), atol=1e-6)
                assert terminated == bool(native.evaluate()["success"]) == bool(info["success"])
                assert not truncated

            robot = native.agent.robot
            np.testing.assert_array_equal(after["agent"]["qpos"], robot.get_qpos())
            np.testing.assert_array_equal(after["agent"]["qvel"], robot.get_qvel())
            np.testing.assert_array_equal(after["extra"]["tcp_pose"], native.agent.tcp.pose.raw_pose)
            np.testing.assert_array_equal(after["extra"][pose_field], getattr(native, actor).pose.raw_pose)
            assert "success" not in after and "success" not in after["extra"]
            positions.append(float(after["extra"]["tcp_pose"][0, 0] - before["extra"]["tcp_pose"][0, 0]))
            if task != "push_t":
                fingers.append(float(robot.get_qpos()[0, -2:].sum()))
        assert positions[0] > 0.01 and positions[1] < -0.01
        if fingers:
            assert fingers[0] > fingers[1] + 0.01
    check_episode_order(task)
    if task not in {"pick_cube", "push_cube"}:
        check_native_parallel_artifacts(task)


@pytest.mark.parametrize("task", ["stack_cube", "peg_insertion_side", "pull_cube", "push_t"])
def test_native_goal_states_terminate_with_success(task):
    open_adapter, _ = TASK_CATALOG[task]()
    with open_adapter() as adapter:
        def start_at_goal():
            native = adapter.env.unwrapped
            assert not bool(native.evaluate()["success"])
            if task == "stack_cube":
                position = native.cubeB.pose.p[0].numpy().copy()
                position[2] += 2 * float(native.cube_half_size[2])
                native.cubeA.set_pose(sapien.Pose(position))
            elif task == "peg_insertion_side":
                native.peg.set_pose(native.goal_pose)
            elif task == "pull_cube":
                position = native.obj.pose.p[0].numpy().copy()
                position[:2] = native.goal_region.pose.p[0, :2].numpy()
                native.obj.set_pose(sapien.Pose(position))
            else:
                position = native.tee.pose.p[0].numpy().copy()
                position[:2] = native.goal_tee.pose.p[0, :2].numpy()
                native.tee.set_pose(sapien.Pose(position, native.goal_tee.pose.q[0].numpy()))
            assert bool(native.evaluate()["success"])
            action = np.zeros(adapter.env.action_space.shape, dtype=np.float32)
            return lambda observation: action

        episode, score = run_scored_episode(adapter, start_at_goal, 0, task)
        assert score == {"success": True}
        assert episode["length"] == 1
        assert episode["steps"][0]["terminated"] and not episode["steps"][0]["truncated"]
        assert bool(adapter.env.unwrapped.evaluate()["success"])
