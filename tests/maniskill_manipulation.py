"""Opt-in integration checks for the native macOS ManiSkill tasks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worldsmith.benchmark.results import snapshot
from worldsmith.benchmark.task_catalog import TASK_CATALOG

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
