"""ManiSkill PushT with a translation-only stick push toward the target."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import open_maniskill


class PushTController:
    def __init__(self) -> None:
        self.waypoint = 0

    def act(self, observation: dict) -> np.ndarray:
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        tee = extra["obj_pose"][0, :2].numpy()
        goal = extra["goal_pos"][0, :2].numpy()

        # Set up behind the T along the line from its center to the goal.
        direction = goal - tee
        distance = np.linalg.norm(direction)
        if distance < 0.01:
            return np.zeros(3, dtype=np.float32)
        approach = tee - 0.14 * direction / distance
        waypoints = ((*approach, 0.09), (*approach, 0.018), (*goal, 0.018))
        if self.waypoint < 2 and np.linalg.norm(waypoints[self.waypoint] - tcp) < 0.01:
            self.waypoint += 1

        delta = (waypoints[self.waypoint] - tcp) / 0.1
        return np.clip(delta, -0.5, 0.5).astype(np.float32)

    __call__ = act


def evaluation_setup():
    return partial(open_maniskill, "PushT-v1", "panda_stick", "pd_ee_delta_pos"), PushTController
