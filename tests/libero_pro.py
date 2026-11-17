"""Native integration checks. Run with the pinned optional Python environment."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark.adapters.libero_adapter import INSTALL, SOURCE, SOURCE_REVISION, idle
from worldsmith.benchmark.runner import InvalidAction
from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_feedback import public_episode


def test_parallel_native_artifacts(check_native_parallel_artifacts):
    check_native_parallel_artifacts("open_middle_drawer")


def test_pinned_source_checkout():
    git = ["git", "--no-optional-locks", "-C", str(SOURCE)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    assert revision == SOURCE_REVISION

    tracked = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    assert not tracked, f"LIBERO-PRO source has local changes:\n{tracked}"

    allowed = {"MUJOCO_LOG.TXT"}
    for line in (INSTALL / "assets.sha256").read_text().splitlines():
        _, path = line.split()
        allowed.add(f"libero/libero/{path}")
    untracked = subprocess.check_output([*git, "ls-files", "--others", "-z"], text=True)
    for name in untracked.split("\0")[:-1]:
        path = Path(name)
        if path.parent.name == "__pycache__" and path.suffix == ".pyc":
            continue
        assert name in allowed, f"Unexpected file in LIBERO-PRO source: {name}"


@pytest.mark.parametrize(("task", "object_key"), [
    ("open_middle_drawer", "drawer_handle_pos"),
    ("place_bowl_on_plate", "akita_black_bowl_1_pos"),
    ("place_two_items_in_basket", "basket_1_pos"),
])
def test_native_episode(task, object_key):
    open_adapter, factory = TASK_GROUPS["libero"][task]()
    with open_adapter() as adapter:
        observation, reset_info = adapter.reset(0)
        assert observation["robot0_eef_pos"].shape == (3,)
        assert observation["robot0_eef_quat"].shape == (4,)
        assert observation[object_key].shape == (3,)
        assert reset_info == {"init_state_index": 0}

        heldout, heldout_info = adapter.reset(8)
        assert heldout_info == {"init_state_index": 8}
        varied_key = "plate_1_pos" if task == "open_middle_drawer" else object_key
        assert not np.allclose(observation[varied_key], heldout[varied_key])
        repeated, _ = adapter.reset(0)
        for key in observation:
            np.testing.assert_array_equal(observation[key], repeated[key])
        with pytest.raises(ValueError, match="saved layout index"):
            adapter.reset(len(adapter.states))

        for invalid in (np.zeros(6), np.full(7, 2.), np.full(7, np.nan), np.zeros(7, dtype=complex)):
            with pytest.raises(InvalidAction):
                adapter.encode_action(invalid)
        rows = []
        result = evaluate_episodes(adapter, factory, [0], lambda episode, metrics: rows.append({
            **episode, **metrics,
        }), task=task)
        assert result.metrics == {"success_rate": 0.0}
        assert result.episodes == [{"seed": 0, "length": adapter.horizon, "success": False}]
        assert not adapter.env.check_success()
        private = rows[0]
        public = public_episode(task, "libero", private)
        assert public["truncated"] and not public["terminated"]
        assert public["initial_observation"] == private["steps"][0]["observation"]
        assert public["frames"][0]["action"] == idle(None).tolist()
        assert set(public["frames"][0]) == {"observation", "action"}
        assert "reset_info" not in public and "native_suite" not in public
        assert private["reset_info"]["init_state_index"] == 0
        assert adapter.metadata()["native_task_index"] == 0


def test_native_success_terminates_and_scores():
    task = "open_middle_drawer"
    open_adapter, factory = TASK_GROUPS["libero"][task]()
    with open_adapter() as adapter:
        def start_with_open_drawer():
            # A positive control for the native predicate, independent of controller skill.
            joint = "wooden_cabinet_1_middle_level"
            adapter.env.sim.data.set_joint_qpos(joint, -0.159)
            adapter.env.sim.data.set_joint_qvel(joint, 0.0)
            adapter.env.sim.forward()
            assert adapter.env.check_success()
            return factory()

        def record(episode, metrics):
            step = episode["steps"][-1]
            assert step["terminated"] and not step["truncated"]
            assert step["reward"] == 1.0
            assert metrics == {"success": True}

        result = evaluate_episodes(adapter, start_with_open_drawer, [0], record, task=task)
        assert result.episodes == [{"seed": 0, "length": 1, "success": True}]
        assert result.metrics == {"success_rate": 1.0}


@pytest.mark.parametrize(("task", "object_name"), [
    ("open_middle_drawer", "plate_1"),
    ("place_bowl_on_plate", "akita_black_bowl_1"),
    ("place_two_items_in_basket", "basket_1"),
])
def test_native_observation_and_osc_action(
    task, object_name, check_episode_order, check_native_parallel_artifacts,
):
    open_adapter, _ = TASK_GROUPS["libero"][task]()
    with open_adapter() as adapter:
        initial, _ = adapter.reset(0)
        env = adapter.env.env
        sim = env.sim
        robot = env.robots[0]
        command = np.array([0.4, 0., 0., 0., 0., -0.2, 1.])
        for _ in range(4):
            # Refresh kinematics from qpos just as native step does before setting its goal.
            sim.forward()
            before = np.array(sim.data.site_xpos[robot.eef_site_id])
            orientation = Rotation.from_matrix(sim.data.site_xmat[robot.eef_site_id].reshape(3, 3))
            adapter.encode_action(command)
            observation, _, terminated, truncated, info = adapter.step(command)
            np.testing.assert_allclose(robot.controller.goal_pos - before, [0.02, 0, 0], atol=1e-6)
            goal = Rotation.from_matrix(robot.controller.goal_ori)
            np.testing.assert_allclose((goal * orientation.inv()).as_rotvec(), [0, 0, -0.1], atol=1e-6)
            assert terminated == bool(adapter.env.check_success()) == info["success"]
            assert not truncated

        assert np.ptp(observation["robot0_gripper_qpos"]) < np.ptp(initial["robot0_gripper_qpos"])
        # LIBERO returns cached sensor samples from the physics substeps. Sample
        # explicitly here to compare the observation and MuJoCo at the same instant.
        sim.forward()
        observation = adapter.observation(env._get_observations(force_update=True))
        np.testing.assert_array_equal(observation["robot0_joint_pos"], sim.data.qpos[robot._ref_joint_pos_indexes])
        np.testing.assert_array_equal(observation["robot0_joint_vel"], sim.data.qvel[robot._ref_joint_vel_indexes])
        np.testing.assert_array_equal(observation["robot0_eef_pos"], sim.data.site_xpos[robot.eef_site_id])
        tool_orientation = Rotation.from_quat(observation["robot0_eef_quat"])
        np.testing.assert_allclose(
            tool_orientation.as_matrix(), sim.data.get_body_xmat(robot.robot_model.eef_name), atol=1e-7,
        )
        native_body = env.obj_body_id[object_name]
        np.testing.assert_array_equal(observation[f"{object_name}_pos"], sim.data.body_xpos[native_body])
        np.testing.assert_allclose(
            Rotation.from_quat(observation[f"{object_name}_quat"]).as_matrix(),
            sim.data.body_xmat[native_body].reshape(3, 3), atol=1e-7,
        )
        offset = sim.data.body_xpos[native_body] - sim.data.site_xpos[robot.eef_site_id]
        np.testing.assert_allclose(
            observation[f"{object_name}_to_robot0_eef_pos"], tool_orientation.inv().apply(offset), atol=1e-7,
        )
    check_episode_order(task)
    if task != "open_middle_drawer":
        check_native_parallel_artifacts(task)


@pytest.mark.parametrize("task", ["place_bowl_on_plate", "place_two_items_in_basket"])
def test_native_placement_success_stops_episode(task):
    open_adapter, factory = TASK_GROUPS["libero"][task]()
    with open_adapter() as adapter:
        def start_near_goal():
            env = adapter.env.env
            sim = env.sim
            idle = factory()
            assert not adapter.env.check_success()

            def place(name, position):
                joint = env.get_object(name).joints[0]
                sim.data.set_joint_qpos(joint, np.r_[position, 1., 0., 0., 0.])
                sim.data.set_joint_qvel(joint, np.zeros(6))
                sim.forward()

            if task == "place_bowl_on_plate":
                # Release the named bowl just above the plate so MuJoCo establishes contact.
                plate = sim.data.body_xpos[env.obj_body_id["plate_1"]]
                place("akita_black_bowl_1", plate + [0., 0., 0.025])
                return idle

            site = "basket_1_contain_region"
            center = sim.data.get_site_xpos(site).copy()
            axis = sim.data.get_site_xmat(site)[:, 0].copy()
            place("alphabet_soup_1", center + 0.032 * axis)
            assert not adapter.env.check_success()
            place("tomato_sauce_1", center - 0.032 * axis)
            assert adapter.env.check_success()
            return idle

        traces = []
        result = evaluate_episodes(
            adapter, start_near_goal, [0], lambda episode, _: traces.append(episode), task=task,
        )
        episode = traces[0]
        assert 1 <= episode["length"] <= 4
        assert result.episodes == [{"seed": 0, "length": episode["length"], "success": True}]
        assert result.metrics == {"success_rate": 1.0}
        assert episode["steps"][-1]["terminated"] and not episode["steps"][-1]["truncated"]
        assert adapter.env.check_success()
