"""Native ManiSkill environment and a scripted Panda controller."""

from contextlib import contextmanager
from importlib.metadata import version

import gymnasium as gym
import mani_skill.envs  # Registers PushCube-v1 with Gymnasium.
import numpy as np

TASK_ID = "PushCube-v1"
OBS_MODE = "state_dict"
ACTION_MODE = "pd_ee_delta_pos"
ROBOT = "panda"


def make_env() -> gym.Env:
    return gym.make(
        TASK_ID,
        num_envs=1,
        robot_uids=ROBOT,
        obs_mode=OBS_MODE,
        control_mode=ACTION_MODE,
        sim_backend="cpu",
        render_backend="cpu",
        render_mode=None,
    )


class PushCubeAdapter:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def reset(self, seed: int):
        return self.env.reset(seed=seed)

    def encode_action(self, action: np.ndarray) -> list:
        if not self.env.action_space.contains(action):
            raise ValueError(f"Controller action is outside {self.env.action_space}: {action}")
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
        backend = self.env.unwrapped.backend
        return {
            "task_id": task.id,
            "task_version": task.version,
            "max_episode_steps": task.max_episode_steps,
            "obs_mode": OBS_MODE,
            "action_mode": ACTION_MODE,
            "sim_backend": backend.sim_backend,
            "render_backend": backend.render_backend,
            "render_mode": self.env.render_mode,
            "robot": ROBOT,
            "versions": {
                name: version(name)
                for name in ("mani-skill-nightly", "sapien", "pin", "gymnasium", "numpy", "torch")
            },
        }


@contextmanager
def open_push_cube():
    env = make_env()
    try:
        yield PushCubeAdapter(env)
    finally:
        env.close()


class PushController:
    """Move behind the cube, lower the closed gripper, then push toward the goal."""

    def __init__(self) -> None:
        self.waypoint = 0

    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        cube = extra["obj_pose"][0, :3].numpy()
        goal = extra["goal_pos"][0].numpy()

        # The goal starts on the cube's +X side. Approach from -X above the table.
        waypoints = (
            (cube[0] - 0.055, cube[1], 0.10),
            (cube[0] - 0.055, cube[1], 0.025),
            (goal[0], goal[1], 0.025),
        )
        if self.waypoint < 2 and np.linalg.norm(waypoints[self.waypoint] - tcp) < 0.01:
            self.waypoint += 1

        # ManiSkill maps normalized arm commands to ±0.1 m and solves IK.
        delta = (waypoints[self.waypoint] - tcp) / 0.1
        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(delta, -0.5, 0.5)
        action[3] = -1.0
        return action

    __call__ = act


def evaluation_setup():
    return open_push_cube, PushController
