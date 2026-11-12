"""Single-environment CPU adapter for native ManiSkill tasks."""

from contextlib import contextmanager
from importlib.metadata import version

import gymnasium as gym
import mani_skill.envs  # Registers native tasks with Gymnasium.
import numpy as np


def make_env(task_id: str, robot: str, control_mode: str) -> gym.Env:
    return gym.make(
        task_id,
        num_envs=1,
        robot_uids=robot,
        obs_mode="state_dict",
        control_mode=control_mode,
        sim_backend="cpu",
        render_backend="cpu",
        render_mode=None,
    )


class ManiSkillAdapter:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def reset(self, seed: int):
        return self.env.reset(seed=seed)

    def encode_action(self, action: np.ndarray) -> list:
        space = self.env.action_space
        if not isinstance(action, np.ndarray) or not space.contains(action):
            raise ValueError(f"Controller action must be a NumPy array in {space}: {action}")
        return action.tolist()

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward.item(), bool(terminated), bool(truncated), info

    def score(self, episode: dict) -> dict:
        # The adapter runs one environment. ManiSkill reports a batched success flag.
        return {"success": episode["steps"][-1]["info"]["success"][0]}

    def summarize(self, episodes: list[dict]) -> dict:
        return {"success_rate": sum(row["success"] for row in episodes) / len(episodes)}

    def metadata(self) -> dict:
        task = gym.spec(self.env.spec.id)
        native = self.env.unwrapped
        backend = native.backend
        return {
            "task_id": task.id,
            "task_version": task.version,
            "max_episode_steps": task.max_episode_steps,
            "obs_mode": native.obs_mode,
            "action_mode": native.control_mode,
            "reward_mode": native.reward_mode,
            "sim_backend": backend.sim_backend,
            "render_backend": backend.render_backend,
            "render_mode": self.env.render_mode,
            "robot": native.robot_uids,
            "reconfiguration_freq": native.reconfiguration_freq,
            "versions": {
                name: version(name)
                for name in ("mani-skill-nightly", "sapien", "pin", "gymnasium", "numpy", "torch")
            },
        }


@contextmanager
def open_maniskill(task_id: str, robot: str, control_mode: str):
    env = make_env(task_id, robot, control_mode)
    try:
        yield ManiSkillAdapter(env)
    finally:
        env.close()
