"""Opt-in integration checks against the pinned CPU simulator."""

import json

import numpy as np
import pytest

from worldsmith.benchmark.evaluate_push_cube import evaluate, json_state
from worldsmith.benchmark.tasks.push_cube import PushController, make_env


def test_native_reset_and_action():
    env = make_env()
    try:
        observation, info = env.reset(seed=42)
        initial = json_state(observation)
        assert not bool(info["success"])
        assert observation["agent"]["qpos"].shape == (1, 9)
        assert observation["agent"]["qvel"].shape == (1, 9)
        assert observation["extra"]["tcp_pose"].shape == (1, 7)
        assert observation["extra"]["obj_pose"].shape == (1, 7)
        assert observation["extra"]["goal_pos"].shape == (1, 3)
        assert env.action_space.shape == (4,)

        action = PushController().act(observation)
        assert action.dtype == np.float32
        assert env.action_space.contains(action)
        next_observation, _, _, _, _ = env.step(action)
        assert json_state(next_observation) != initial

        other_observation, _ = env.reset(seed=43)
        assert json_state(other_observation["extra"]["obj_pose"]) != initial["extra"]["obj_pose"]
        repeated_observation, _ = env.reset(seed=42)
        assert json_state(repeated_observation) == initial
    finally:
        env.close()


def test_complete_episodes_and_artifacts(tmp_path):
    output = tmp_path / "evaluation"
    summary = evaluate(2, 42, output)
    assert json.loads((output / "summary.json").read_text()) == summary
    assert summary["task_id"] == "PushCube-v1"
    assert summary["task_version"] == 1
    assert summary["obs_mode"] == "state_dict"
    assert summary["action_mode"] == "pd_ee_delta_pos"
    assert summary["sim_backend"] == "physx_cpu"
    assert summary["render_backend"] == "sapien_cpu"
    assert summary["render_mode"] is None
    assert summary["max_episode_steps"] == 50
    assert summary["controller"] == "worldsmith.benchmark.tasks.push_cube.PushController"
    assert summary["versions"]["mani-skill-nightly"] == "2026.8.2.400"
    assert summary["versions"]["sapien"] == "3.0.3"
    assert summary["seeds"] == [42, 43]

    episodes = [
        json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(episodes) == 2
    for row, episode in zip(summary["episodes"], episodes, strict=True):
        assert row == {key: episode[key] for key in ("seed", "length", "success")}
        steps = episode["steps"]
        assert 0 < episode["length"] == len(steps) <= summary["max_episode_steps"]
        for step in steps:
            assert step["observation"]["agent"]["qpos"]
            assert len(step["observation"]["extra"]["obj_pose"][0]) == 7
            assert len(step["action"]) == 4
            assert all(-1 <= value <= 1 for value in step["action"])
        final = steps[-1]
        assert final["terminated"] or final["truncated"]
        assert episode["success"] is final["info"]["success"][0]
        assert episode["final_observation"]["extra"]["obj_pose"]
    assert summary["success_rate"] == sum(episode["success"] for episode in episodes) / 2


@pytest.mark.parametrize(("episodes", "seed"), [(0, 42), (1, -1), (2, 2**32 - 1)])
def test_invalid_seed_sequence(episodes, seed, tmp_path):
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="episode count and seeds"):
        evaluate(episodes, seed, output)
    assert not output.exists()
