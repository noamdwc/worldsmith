"""Run with the pinned optional Robosuite environment."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark.runner import InvalidAction
from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_feedback import public_episode


@pytest.mark.parametrize(("task", "native_task", "object_key", "size", "horizon"), [
    ("open_latched_door", "Door", "handle_pos", 7, 300),
    ("assemble_nuts", "NutAssembly", "square_nut_pos", 7, 450),
    ("handover_hammer", "TwoArmHandover", "hammer_handle_pos", 14, 400),
])
def test_native_episode(task, native_task, object_key, size, horizon):
    open_adapter, factory = TASK_GROUPS["robosuite"][task]()
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
    open_adapter, _ = TASK_GROUPS["robosuite"][task]()

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
