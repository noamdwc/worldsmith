"""ManiSkill PickCube with a top-down grasp and transport baseline."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import open_maniskill


class PickController:
    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        cube = extra["obj_pose"][0, :3].numpy()
        goal = extra["goal_pos"][0].numpy()

        if extra["is_grasped"][0]:
            target = tcp + goal - cube
            gripper = -1.0
        elif np.linalg.norm(tcp[:2] - cube[:2]) > 0.012:
            target = cube + (0, 0, 0.08)
            gripper = 1.0
        else:
            target = cube
            gripper = -1.0 if np.linalg.norm(tcp - cube) < 0.012 else 1.0
        return np.array([*np.clip((target - tcp) / 0.1, -0.5, 0.5), gripper], dtype=np.float32)

    __call__ = act


def evaluation_setup():
    return partial(open_maniskill, "PickCube-v1", "panda", "pd_ee_delta_pos"), PickController
