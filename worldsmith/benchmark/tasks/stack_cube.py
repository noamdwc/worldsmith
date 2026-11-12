"""ManiSkill StackCube with a scripted grasp, transfer, and release."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import open_maniskill


class StackController:
    def __init__(self) -> None:
        self.waypoint = 0
        self.closing_steps = 0

    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        cube = extra["cubeA_pose"][0, :3].numpy()
        base = extra["cubeB_pose"][0, :3].numpy()

        # Lift clear of both cubes before moving horizontally. Cube side length is 4 cm.
        waypoints = (
            cube + (0, 0, 0.08),
            cube,
            (cube[0], cube[1], 0.14),
            (base[0], base[1], 0.14),
            base + (0, 0, 0.043),
            base + (0, 0, 0.12),
        )
        if self.waypoint == 1:
            if np.linalg.norm(tcp - cube) < 0.012 or self.closing_steps:
                self.closing_steps += 1
                if self.closing_steps == 6:
                    self.waypoint += 1
        elif self.waypoint < 5 and np.linalg.norm(waypoints[self.waypoint] - tcp) < 0.01:
            self.waypoint += 1

        gripper = -1.0 if self.closing_steps and self.waypoint < 5 else 1.0
        delta = (waypoints[self.waypoint] - tcp) / 0.1
        return np.array([*np.clip(delta, -0.5, 0.5), gripper], dtype=np.float32)

    __call__ = act


def evaluation_setup():
    return partial(open_maniskill, "StackCube-v1", "panda_wristcam", "pd_ee_delta_pos"), StackController
