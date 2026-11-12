"""ManiSkill PushCube and its scripted Panda controller."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import make_env as make_maniskill_env, open_maniskill

TASK_ID = "PushCube-v1"
ROBOT = "panda"
ACTION_MODE = "pd_ee_delta_pos"

make_env = partial(make_maniskill_env, TASK_ID, ROBOT, ACTION_MODE)
open_push_cube = partial(open_maniskill, TASK_ID, ROBOT, ACTION_MODE)


class PushController:
    """Move behind the cube, lower the closed gripper, then sweep toward the goal."""

    approach_offset_x = -0.055

    def __init__(self) -> None:
        self.waypoint = 0

    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        cube = extra["obj_pose"][0, :3].numpy()
        goal = extra["goal_pos"][0].numpy()

        # Approach above the table on the side opposite the goal.
        approach_x = cube[0] + self.approach_offset_x
        waypoints = (
            (approach_x, cube[1], 0.10),
            (approach_x, cube[1], 0.025),
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
