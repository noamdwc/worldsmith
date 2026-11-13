"""Native integration checks. Run with the pinned optional Python environment."""

import subprocess

import numpy as np
import pytest

from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark import libero_adapter
from worldsmith.benchmark.libero_adapter import idle
from worldsmith.benchmark.runner import InvalidAction
from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_feedback import public_episode


def test_source_check_allows_assets_and_generated_files(tmp_path, monkeypatch):
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    source = tmp_path / "module.py"
    source.write_text("pass\n")
    subprocess.run([*git, "add", "module.py"], check=True)
    subprocess.run([*git, "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "Pinned source"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(libero_adapter, "SOURCE", tmp_path)
    monkeypatch.setattr(libero_adapter, "SOURCE_REVISION", revision)

    asset = tmp_path / "libero/libero/init_files/task.pruned_init"
    asset.parent.mkdir(parents=True)
    asset.touch()
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").touch()
    (tmp_path / "MUJOCO_LOG.TXT").touch()
    assets = ["init_files/task.pruned_init"]
    libero_adapter.check_source(assets)

    added = tmp_path / "other.py"
    added.touch()
    with pytest.raises(RuntimeError, match="Unexpected file.*other.py"):
        libero_adapter.check_source(assets)
    added.unlink()

    source.write_text("# local edit\n")
    with pytest.raises(RuntimeError, match="local changes"):
        libero_adapter.check_source(assets)
    subprocess.run([*git, "add", "module.py"], check=True)
    with pytest.raises(RuntimeError, match="local changes"):
        libero_adapter.check_source(assets)


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
