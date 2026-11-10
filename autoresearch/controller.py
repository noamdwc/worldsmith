"""Starter controller that aims at the latest target measurement."""

from math import atan2

from worldsmith.api import Action, Controller, Observation


def make_controller() -> Controller:
    def chase(observation: Observation) -> Action:
        robot_x, robot_y = observation.entities[observation.robot_id].position
        target_x, target_y = observation.entities["target"].position
        return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

    return chase
