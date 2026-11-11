"""Native ManiSkill environment and a scripted Panda controller."""

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
