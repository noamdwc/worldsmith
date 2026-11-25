"""Run with the pinned optional Robosuite environment."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark.runner import InvalidAction, run_scored_episode
from worldsmith.benchmark.task_catalog import DEFAULT_CONTROLLERS, TASK_GROUPS
from worldsmith.research_feedback import public_episode


def test_parallel_native_artifacts(check_native_parallel_artifacts):
    check_native_parallel_artifacts("handover_hammer")


@pytest.mark.parametrize(("task", "native_task", "object_key", "size", "horizon"), [
    ("open_latched_door", "Door", "handle_pos", 7, 300),
    ("assemble_nuts", "NutAssembly", "square_nut_pos", 7, 450),
    ("handover_hammer", "TwoArmHandover", "hammer_handle_pos", 14, 400),
    ("wipe_table", "Wipe", "wipe_centroid", 6, 300),
])
def test_native_episode(task, native_task, object_key, size, horizon):
    open_adapter = TASK_GROUPS["robosuite"][task]
    factory = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        observation, reset_info = adapter.reset(0)
        assert reset_info == {"native_seed": 0}
        assert type(adapter.env).__name__ == native_task
        assert not adapter.env.has_renderer and not adapter.env.has_offscreen_renderer
        assert observation[object_key].shape == (3,)
        assert observation["robot0_eef_quat_site"].shape == (4,)
        assert ("robot1_eef_pos" in observation) == (size == 14)

        heldout, _ = adapter.reset(801)
        assert not np.allclose(observation[object_key], heldout[object_key])
        heldout_state = adapter.env.sim.data.qpos.copy()
        repeated, _ = adapter.reset(0)
        for key, value in observation.items():
            np.testing.assert_array_equal(value, repeated[key])
        adapter.reset(801)
        np.testing.assert_array_equal(heldout_state, adapter.env.sim.data.qpos)

        assert adapter.encode_action(np.zeros(size, dtype=np.float32)) == [0.] * size
        assert adapter.encode_action(np.ones(size)) == [1.] * size
        for invalid in (
            [0.] * size, np.zeros(size - 1), np.zeros((1, size)), np.zeros(size, dtype=int),
            np.full(size, 1.01), np.full(size, np.nan), np.full(size, -np.inf),
        ):
            with pytest.raises(InvalidAction):
                adapter.encode_action(invalid)

        rows = []
        result = evaluate_episodes(
            adapter, factory, [0], lambda episode, metrics: rows.append({**episode, **metrics}), task=task,
        )
        assert result.episodes == [{"seed": 0, "length": horizon, "success": False}]
        assert result.metrics == {"success_rate": 0.0}
        assert not adapter.env._check_success()
        private = rows[0]
        public = public_episode(task, "robosuite", private)
        assert public["truncated"] and not public["terminated"]
        assert public["frames"][0] == {
            key: private["steps"][0][key] for key in ("observation", "action")
        }
        assert "reset_info" not in public
        assert private["reset_info"] == {"native_seed": 0}
        assert adapter.metadata()["native_task"] == native_task
        assert adapter.metadata()["action_shape"] == [size]
    assert adapter.env is None


def test_controller_opens_native_door():
    task = "open_latched_door"
    open_adapter = TASK_GROUPS["robosuite"][task]

    def factory():
        step = 0
        hinge = handle = tangent = down = None

        def control(observation):
            nonlocal step, hinge, handle, tangent, down
            if step == 0:
                # From the initial hinge-to-handle direction, find the opening arc.
                offset = observation["handle_pos"] - observation["hinge_pos"]
                tangent = np.array([-offset[1], offset[0], 0.])
                tangent /= np.linalg.norm(tangent)
                radial = np.cross(tangent, [0, 0, 1])
                tool_z = np.array([0, 0, -1])
                tool_y = np.cross(tool_z, radial)
                down = Rotation.from_matrix(np.column_stack((radial, tool_y, tool_z)))
                hinge = observation["hinge_pos"]
                handle = observation["handle_pos"] - 0.048 * radial - 0.008 * tangent

            # Approach from above, close the fingers, press the latch down, then sweep.
            latch = np.clip((step - 95) / 25, 0, 1)
            swing = np.clip((step - 130) / 45, 0, 1)
            arc = Rotation.from_rotvec([0, 0, 0.55 * swing])
            target = hinge + arc.apply(handle - hinge)
            target[2] += 0.07 if step < 35 else -0.105 * latch
            orientation = arc * Rotation.from_rotvec(0.95 * latch * tangent) * down
            delta_pos = (target - observation["robot0_eef_pos"]) / 0.05
            current = Rotation.from_quat(observation["robot0_eef_quat_site"])
            delta_rot = (orientation * current.inv()).as_rotvec() / 0.5
            grip = -1. if step < 70 else 1.
            step += 1
            return np.r_[np.clip(delta_pos, -1, 1), np.clip(delta_rot, -1, 1), grip]

        return control

    with open_adapter() as adapter:
        def record(episode, metrics):
            final = episode["steps"][-1]
            assert final["terminated"] and not final["truncated"]
            assert final["reward"] == 1.0
            assert metrics == {"success": True}
            assert episode["final_observation"]["hinge_angle"] > 0.3

        result = evaluate_episodes(adapter, factory, [0], record, task=task)
        assert result.metrics == {"success_rate": 1.0}
        assert result.episodes[0]["length"] < adapter.horizon


def handover_controller():
    """Seed-3 AutoResearch policy distilled to a test-only public controller."""
    phase = "approach"
    dwell = 0
    transfer_pickup_pos = None
    advance_when_reached = {"approach": "lower", "lower": "grasp", "lift": "align"}

    def control(observation):
        nonlocal phase, dwell, transfer_pickup_pos
        pickup = observation["robot0_eef_pos"]
        receiver = observation["robot1_eef_pos"]
        handle = observation["hammer_handle_pos"]
        table = observation["table_height"]

        if phase == "approach":
            pickup_target = handle + (0, 0, 0.12)
        elif phase in ("lower", "grasp"):
            pickup_target = handle + (0, 0, 0.005)
        elif phase == "lift":
            pickup_target = pickup + np.array([0, 0, table + 0.28 - handle[2]])
        elif phase == "align":
            pickup_target = pickup + np.array([0, -0.04, table + 0.30]) - handle
        elif phase == "receive":
            pickup_target = transfer_pickup_pos
        else:
            pickup_target = transfer_pickup_pos + (0, -0.15, 0.08)

        receiver_target = np.array([0, 0.12, table + 0.38])
        if phase in ("align", "receive"):
            receiver_target = handle + (0, 0.045, 0.015)
        elif phase == "release":
            receiver_target = handle + (0, 0, 0.05)

        action = np.zeros(14, dtype=np.float32)
        action[:3] = np.clip((pickup_target - pickup) / 0.05, -0.6, 0.6)
        action[6] = 1 if phase in ("grasp", "lift", "align", "receive") else -1
        action[7:10] = np.clip((receiver_target - receiver) / 0.05, -0.6, 0.6)
        action[13] = 1 if phase in ("receive", "release") else -1

        pickup_reached = np.linalg.norm(pickup_target - pickup) < 0.018
        if phase in advance_when_reached and pickup_reached:
            phase = advance_when_reached[phase]
        elif phase == "grasp":
            dwell += 1
            if dwell >= 18:
                phase = "lift"
        elif phase == "align" and pickup_reached and np.linalg.norm(receiver_target - receiver) < 0.02:
            transfer_pickup_pos = pickup.copy()
            phase = "receive"
            dwell = 0
        elif phase == "receive":
            dwell += 1
            if dwell >= 20:
                phase = "release"
        return action

    return control


def test_controller_hands_over_native_hammer():
    open_adapter = TASK_GROUPS["robosuite"]["handover_hammer"]
    with open_adapter() as adapter:
        observation, _ = adapter.reset(3)
        control = handover_controller()
        native = adapter.env
        # The native contact flags are ordered pickup arm, receiving arm.
        grasp_states = [native._get_task_info()[:2]]

        for step in range(1, adapter.horizon + 1):
            action = control(observation)
            adapter.encode_action(action)
            observation, reward, terminated, truncated, info = adapter.step(action)

            # Inspect native contacts from the test, outside the controller's inputs.
            grasp = native._get_task_info()[:2]
            if grasp != grasp_states[-1]:
                grasp_states.append(grasp)
            assert terminated == info["success"] == native._check_success()
            if terminated or truncated:
                break

        assert grasp_states == [(False, False), (True, False), (True, True), (False, True)]
        assert step == 176
        assert action[6] < 0 < action[13]
        assert terminated and not truncated
        assert reward == 1.0


@pytest.mark.parametrize("task", TASK_GROUPS["robosuite"])
def test_native_observation_and_world_frame_actions(
    task, check_episode_order, check_native_parallel_artifacts,
):
    open_adapter = TASK_GROUPS["robosuite"][task]
    with open_adapter() as adapter:
        initial, _ = adapter.reset(0)
        env = adapter.env
        sim = env.sim
        sites = [robot.eef_site_id["right"] for robot in env.robots]
        # Opposite world-frame offsets distinguish the two arms' action slices.
        commands = np.array([[0.4, 0., 0., 0., 0., -0.2, 1.], [-0.4, 0., 0., 0., 0., 0.2, -1.]])
        commands = commands[:len(env.robots)]
        if task == "wipe_table":
            commands = commands[:, :6]
        action = commands.flatten()
        for _ in range(6):
            # Refresh kinematics from qpos just as native step does before setting its goal.
            sim.forward()
            before = sim.data.site_xpos[sites].copy()
            orientations = Rotation.from_matrix(sim.data.site_xmat[sites].reshape(-1, 3, 3))
            adapter.encode_action(action)
            observation, _, terminated, truncated, info = adapter.step(action)
            assert terminated == bool(env._check_success()) == info["success"]
            assert not truncated
            for index, robot in enumerate(env.robots):
                controller = robot.part_controllers["right"]
                np.testing.assert_allclose(
                    controller.goal_pos - before[index], commands[index, :3] * 0.05, atol=1e-6,
                )
                delta = Rotation.from_matrix(controller.goal_ori) * orientations[index].inv()
                np.testing.assert_allclose(delta.as_rotvec(), commands[index, 3:6] * 0.5, atol=1e-6)

        for index, (robot, site) in enumerate(zip(env.robots, sites, strict=True)):
            prefix = f"robot{index}_"
            np.testing.assert_array_equal(
                observation[prefix + "joint_pos"], sim.data.qpos[robot._ref_joint_pos_indexes],
            )
            np.testing.assert_array_equal(
                observation[prefix + "joint_vel"], sim.data.qvel[robot._ref_joint_vel_indexes],
            )
            np.testing.assert_array_equal(observation[prefix + "eef_pos"], sim.data.site_xpos[site])
            np.testing.assert_allclose(
                Rotation.from_quat(observation[prefix + "eef_quat_site"]).as_matrix(),
                sim.data.site_xmat[site].reshape(3, 3), atol=1e-7,
            )
        if task != "wipe_table":
            assert np.ptp(observation["robot0_gripper_qpos"]) < np.ptp(initial["robot0_gripper_qpos"])

        if task == "open_latched_door":
            np.testing.assert_array_equal(observation["handle_pos"], sim.data.site_xpos[env.door_handle_site_id])
            assert observation["hinge_angle"] == sim.data.get_joint_qpos(env.door.joints[0])
        elif task == "assemble_nuts":
            np.testing.assert_array_equal(observation["square_peg_pos"], sim.data.get_body_xpos("peg1"))
            np.testing.assert_array_equal(observation["round_peg_pos"], sim.data.get_body_xpos("peg2"))
        elif task == "handover_hammer":
            np.testing.assert_array_equal(observation["hammer_pos"], sim.data.body_xpos[env.hammer_body_id])
            assert observation["table_height"] == sim.data.site_xpos[env.table_top_id, 2]
        else:
            assert "robot0_gripper_qpos" not in observation
            positions = np.array([
                sim.data.get_body_xpos(marker.root_body) for marker in env.model.mujoco_arena.markers
            ])
            centroid = positions.mean(axis=0)
            np.testing.assert_allclose(observation["wipe_centroid"], centroid)
            assert observation["wipe_radius"] == pytest.approx(np.linalg.norm(positions - centroid, axis=1).max())
    check_episode_order(task)
    if task != "handover_hammer":
        check_native_parallel_artifacts(task)


def test_partial_nut_reward_does_not_end_episode():
    task = "assemble_nuts"
    open_adapter = TASK_GROUPS["robosuite"][task]
    factory = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        def start_placing_nuts():
            env = adapter.env
            sim = env.sim
            placements = iter(zip(env.nuts, ("peg1", "peg2"), strict=True))
            idle = factory()

            def place_next(observation):
                nut, peg = next(placements)
                position = sim.data.get_body_xpos(peg).copy()
                # Seat the nut on the table with its opening centered on its peg.
                position[2] = env.table_offset[2] + 0.012
                sim.data.set_joint_qpos(nut.joints[0], np.r_[position, 1., 0., 0., 0.])
                sim.data.set_joint_qvel(nut.joints[0], np.zeros(6))
                sim.forward()
                return idle(observation)

            return place_next

        traces = []
        result = evaluate_episodes(
            adapter, start_placing_nuts, [0], lambda episode, _: traces.append(episode), task=task,
        )
        assert result.episodes == [{"seed": 0, "length": 2, "success": True}]
        assert result.metrics == {"success_rate": 1.0}
        partial, complete = traces[0]["steps"]
        assert partial["reward"] == 0.5
        assert not partial["terminated"] and not partial["truncated"]
        assert complete["reward"] == 1.0 and complete["terminated"] and not complete["truncated"]
        assert adapter.env._check_success()


def test_wiping_contact_removes_native_markers():
    open_adapter = TASK_GROUPS["robosuite"]["wipe_table"]
    with open_adapter() as adapter:
        observation, _ = adapter.reset(0)
        env = adapter.env
        centroid = observation["wipe_centroid"].copy()
        assert observation["proportion_wiped"] == 0
        assert not observation["robot0_contact"]

        # Approach above the dirt, then lower the passive wiper into table contact.
        for z in (0.10, -0.015):
            target = centroid + (0, 0, z)
            for _ in range(24):
                delta = (target - observation["robot0_eef_pos"]) / 0.05
                action = np.r_[np.clip(delta, -0.6, 0.6), 0., 0., 0.]
                # Native observables sample before the reward call updates coverage.
                sampled_coverage = len(env.wiped_markers) / env.num_markers
                observation, _, terminated, truncated, info = adapter.step(action)
                assert not terminated and not truncated and not info["success"]
                assert observation["proportion_wiped"] == sampled_coverage

        assert observation["robot0_contact"]
        assert observation["robot0_contact"] == env._has_gripper_contact
        assert 0 < len(env.wiped_markers) < env.num_markers
        for marker in env.wiped_markers:
            marker_id = env.sim.model.geom_name2id(marker.visual_geoms[0])
            assert env.sim.model.geom_rgba[marker_id, 3] == 0

        reset, _ = adapter.reset(0)
        assert reset["proportion_wiped"] == 0 and not adapter.env.wiped_markers
        np.testing.assert_array_equal(reset["wipe_centroid"], centroid)


def test_native_wipe_completion_and_timeout_aggregate():
    task = "wipe_table"
    open_adapter = TASK_GROUPS["robosuite"][task]
    idle = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        complete = iter((True, False))

        def factory():
            env = adapter.env
            assert not env._check_success()
            if next(complete):
                # Set the native coverage state independently of the contact test.
                env.wiped_markers.extend(env.model.mujoco_arena.markers)
                assert env._check_success()
            return idle()

        traces = []
        result = evaluate_episodes(adapter, factory, [0, 0], lambda episode, _: traces.append(episode), task=task)
        assert result.metrics == {"success_rate": 0.5}
        assert result.episodes == [
            {"seed": 0, "length": 1, "success": True},
            {"seed": 0, "length": adapter.horizon, "success": False},
        ]
        success, timeout = (trace["steps"][-1] for trace in traces)
        assert success["terminated"] and not success["truncated"]
        assert not timeout["terminated"] and timeout["truncated"]


def test_wipe_joint_limit_terminates_as_failure():
    task = "wipe_table"
    open_adapter = TASK_GROUPS["robosuite"][task]
    idle = DEFAULT_CONTROLLERS[task]()
    with open_adapter() as adapter:
        def start_at_joint_limit():
            env = adapter.env
            robot = env.robots[0]
            index = robot._ref_joint_indexes[0]
            env.sim.data.qpos[robot._ref_joint_pos_indexes[0]] = env.sim.model.jnt_range[index, 1] - 0.01
            env.sim.forward()
            assert robot.check_q_limits()
            return idle()

        episode, score = run_scored_episode(adapter, start_at_joint_limit, 0, task)
        assert adapter.env.robots[0].check_q_limits()
        assert score == {"success": False}
        assert episode["length"] == 1
        last = episode["steps"][-1]
        assert last["terminated"] and not last["truncated"]
