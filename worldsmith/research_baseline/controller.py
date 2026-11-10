"""Shared goal-aware starter controller."""

from math import atan2

from worldsmith.api import INTERCEPTION_GOAL, Action, Controller, Observation


def make_controller() -> Controller:
    def control(observation: Observation) -> Action:
        if observation.goal != INTERCEPTION_GOAL:
            raise ValueError(f"Unsupported goal: {observation.goal!r}")

        robot_x, robot_y = observation.entities[observation.robot_id].position
        target_x, target_y = observation.entities["target"].position
        return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

    return control
