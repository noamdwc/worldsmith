"""Behavioral checks for ManiSkill rolling and locomotion on the pinned CPU stack."""

import numpy as np
import pytest
import sapien

from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark.runner import InvalidAction, run_scored_episode
from worldsmith.benchmark.task_catalog import DEFAULT_CONTROLLERS, TASK_CATALOG
from worldsmith.research_feedback import public_episode


@pytest.mark.parametrize(("task", "goal_key", "horizon"), [
    ("roll_ball", "ball_pose", 80),
    ("quadruped_reach", "goal_pos", 200),
])
def test_native_reset_timeout_and_parallel(
    task, goal_key, horizon, check_episode_order, check_native_parallel_artifacts,
):
    open_adapter = TASK_CATALOG[task]
    idle = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        first, _ = adapter.reset(0)
        other, _ = adapter.reset(801)
        assert not np.allclose(first["extra"][goal_key], other["extra"][goal_key])
        episode, score = run_scored_episode(adapter, idle, 0, task)
        assert score == {"success": False}
        assert episode["length"] == horizon == adapter.metadata()["max_episode_steps"]
        public = public_episode(task, "maniskill", {**episode, **score})
        assert public["truncated"] and not public["terminated"]
        assert set(public["frames"][0]) == {"observation", "action"}
        assert "reset_info" not in public
        with pytest.raises(InvalidAction):
            adapter.encode_action(np.full(adapter.env.action_space.shape, 1.01, dtype=np.float32))
    check_episode_order(task)
    check_native_parallel_artifacts(task)


def test_panda_impulse_rolls_ball_after_arm_stops():
    open_adapter = TASK_CATALOG["roll_ball"]
    idle = DEFAULT_CONTROLLERS["roll_ball"]()
    with open_adapter() as adapter:
        obs, _ = adapter.reset(0)
        native = adapter.env.unwrapped
        initial_ball = obs["extra"]["ball_pose"][0, :3].numpy().copy()
        goal = obs["extra"]["goal_pos"][0, :2].numpy()
        direction = goal - initial_ball[:2]
        direction /= np.linalg.norm(direction)
        behind = initial_ball[:2] - 0.11 * direction
        targets = (
            (*behind, 0.11),
            (*behind, 0.045),
            (*(initial_ball[:2] + 0.15 * direction), 0.045),
        )
        arm = native.agent.controller.controllers["arm"]
        for target in targets:
            for _ in range(16):
                delta = target - obs["extra"]["tcp_pose"][0, :3].numpy()
                # RollBall rotates the Panda base -90 degrees about world Z.
                base_delta = np.array([-delta[1], delta[0], delta[2]])
                action = np.array([*np.clip(base_delta / 0.1, -0.6, 0.6), -1], dtype=np.float32)
                before = arm.ee_pose_at_base.p.numpy().copy()
                assert adapter.encode_action(action) == action.tolist()
                obs, _, terminated, truncated, info = adapter.step(action)
                np.testing.assert_allclose(arm._target_pose.p.numpy() - before, [action[:3] * 0.1], atol=1e-6)
                assert not terminated and not truncated and not bool(info["success"])

        moving_ball = obs["extra"]["ball_pose"][0, :2].numpy().copy()
        assert (moving_ball - initial_ball[:2]) @ direction > 0.2
        assert obs["extra"]["ball_vel"][0, :2].numpy() @ direction > 0.2
        obs, *_ = adapter.step(idle()(obs))
        extra = obs["extra"]
        assert (extra["ball_pose"][0, :2].numpy() - moving_ball) @ direction > 0.005
        np.testing.assert_array_equal(extra["ball_pose"], native.ball.pose.raw_pose)
        np.testing.assert_array_equal(extra["ball_vel"], native.ball.linear_velocity)
        np.testing.assert_array_equal(extra["goal_pos"], native.goal_region.pose.p)
        np.testing.assert_array_equal(extra["tcp_pose"], native.agent.tcp.pose.raw_pose)
        np.testing.assert_array_equal(obs["agent"]["qpos"], native.agent.robot.get_qpos())


def test_quadruped_joint_target_and_proprioception():
    open_adapter = TASK_CATALOG["quadruped_reach"]
    with open_adapter() as adapter:
        before, _ = adapter.reset(0)
        native = adapter.env.unwrapped
        robot = native.agent.robot
        body = native.agent.controller.controllers["body"]
        action = np.zeros(12, dtype=np.float32)
        action[[0, 7, 9]] = [0.3, 0.2, -0.2]
        assert [robot.active_joints[index].name for index in (0, 7, 9)] == [
            "FL_hip_joint", "RR_thigh_joint", "FR_calf_joint",
        ]
        assert adapter.encode_action(action) == action.tolist()
        obs, _, terminated, truncated, info = adapter.step(action)
        np.testing.assert_allclose(body._target_qpos.numpy() - before["agent"]["qpos"].numpy(), [0.7 * action], atol=1e-6)
        assert obs["agent"]["qpos"][0, 0] - before["agent"]["qpos"][0, 0] > 0.03
        np.testing.assert_array_equal(obs["agent"]["qpos"], robot.get_qpos())
        np.testing.assert_array_equal(obs["agent"]["qvel"], robot.get_qvel())
        extra = obs["extra"]
        np.testing.assert_array_equal(extra["root_linear_velocity"], robot.root_linear_velocity)
        np.testing.assert_array_equal(extra["root_angular_velocity"], robot.root_angular_velocity)
        np.testing.assert_array_equal(extra["goal_pos"], native.goal.pose.p[:, :2])
        np.testing.assert_array_equal(extra["robot_to_goal"], native.goal.pose.p[:, :2] - robot.pose.p[:, :2])
        assert set(extra) == {"root_linear_velocity", "root_angular_velocity", "goal_pos", "robot_to_goal"}
        assert not terminated and not truncated and not bool(info["success"])


@pytest.mark.parametrize("task", ["roll_ball", "quadruped_reach"])
def test_native_success_and_timeout_aggregate(task):
    open_adapter = TASK_CATALOG[task]
    idle = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        set_goal_state = iter((True, False))

        def factory():
            native = adapter.env.unwrapped
            assert not bool(native.evaluate()["success"])
            if next(set_goal_state):
                # Exercise the native predicate from a controlled physical state.
                if task == "roll_ball":
                    xy = native.goal_region.pose.p[0, :2].numpy()
                    native.ball.set_pose(sapien.Pose([*xy, native.ball_radius]))
                else:
                    robot = native.agent.robot
                    xy = native.goal.pose.p[0, :2].numpy()
                    robot.set_pose(sapien.Pose([*xy, float(robot.pose.p[0, 2])]))
                assert bool(native.evaluate()["success"])
            return idle()

        traces = []
        result = evaluate_episodes(adapter, factory, [0, 0], lambda episode, _: traces.append(episode), task=task)
        assert result.metrics == {"success_rate": 0.5}
        assert result.episodes == [
            {"seed": 0, "length": 1, "success": True},
            {"seed": 0, "length": adapter.metadata()["max_episode_steps"], "success": False},
        ]
        complete = traces[0]["steps"][-1]
        timeout = traces[1]["steps"][-1]
        assert complete["terminated"] and not complete["truncated"]
        assert not timeout["terminated"] and timeout["truncated"]
        assert complete["observation"] == traces[1]["steps"][0]["observation"]


def test_quadruped_fall_terminates_as_failure():
    open_adapter = TASK_CATALOG["quadruped_reach"]
    idle = DEFAULT_CONTROLLERS["quadruped_reach"]()
    with open_adapter() as adapter:
        def start_upside_down():
            robot = adapter.env.unwrapped.agent.robot
            robot.set_pose(sapien.Pose([0, 0, 0.12], [0, 1, 0, 0]))
            return idle()

        episode, score = run_scored_episode(adapter, start_upside_down, 0, "quadruped_reach")
        last = episode["steps"][-1]
        assert episode["length"] < adapter.metadata()["max_episode_steps"]
        assert last["terminated"] and not last["truncated"]
        assert last["info"]["is_fallen"] == last["info"]["fail"] == [True]
        assert score == {"success": False}
