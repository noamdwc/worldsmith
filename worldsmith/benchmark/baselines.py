"""Fixed reference controllers for checking that anticipating motion helps."""

import math
from collections import deque

from worldsmith.api import Action, Observation


class Chase:
    def __call__(self, observation: Observation) -> Action:
        rx, ry = observation.robot.position
        tx, ty = observation.target.position
        return Action(math.atan2(ty - ry, tx - rx))


class Lead:
    def __init__(self):
        self._history: deque[Observation] = deque(maxlen=9)

    def __call__(self, observation: Observation) -> Action:
        self._history.append(observation)
        rx, ry = observation.robot.position
        tx, ty = observation.target.position
        dx, dy = tx - rx, ty - ry

        if len(self._history) == self._history.maxlen:
            first = self._history[0]
            old_x, old_y = first.target.position
            elapsed = observation.time - first.time
            # A longer measurement interval reduces noise in the finite difference.
            lookahead = math.hypot(dx, dy) / observation.robot_max_speed
            dx += (tx - old_x) / elapsed * lookahead
            dy += (ty - old_y) / elapsed * lookahead

        return Action(math.atan2(dy, dx))
