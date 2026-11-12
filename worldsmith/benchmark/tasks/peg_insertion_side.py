"""ManiSkill PegInsertionSide with a scripted pickup attempt."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import open_maniskill


class PegPickupController:
    def __init__(self) -> None:
        self.waypoint = 0
        self.closing_steps = 0

    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        peg = extra["peg_pose"][0, :3].numpy()
        waypoints = (peg + (0, 0, 0.08), peg, (peg[0], peg[1], 0.15))

        if self.waypoint == 0 and np.linalg.norm(tcp - waypoints[0]) < 0.012:
            self.waypoint = 1
        if self.waypoint == 1 and (np.linalg.norm(tcp - peg) < 0.012 or self.closing_steps):
            self.closing_steps += 1
            if self.closing_steps == 6:
                self.waypoint = 2

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((waypoints[self.waypoint] - tcp) / 0.1, -0.5, 0.5)
        action[6] = -1.0 if self.closing_steps else 1.0
        return action

    __call__ = act


def evaluation_setup():
    return partial(
        open_maniskill, "PegInsertionSide-v1", "panda_wristcam", "pd_ee_delta_pose"
    ), PegPickupController
