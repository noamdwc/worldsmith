"""Reactive starter using only the current goal and position measurements."""

from math import atan2, hypot

from worldsmith.api import (
    INTERCEPTION_GOAL, NAVIGATION_GOAL, RENDEZVOUS_GOAL, Action, Controller, Observation,
)


def make_controller() -> Controller:
    def control(observation: Observation) -> Action:
        robot_x, robot_y = observation.entities[observation.robot_id].position
        if observation.goal == INTERCEPTION_GOAL:
            # Preserve the original Chase baseline exactly for interception.
            target_x, target_y = observation.entities["target"].position
            return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

        if observation.goal == RENDEZVOUS_GOAL:
            target_x, target_y = observation.entities["target"].position
            dx, dy = target_x - robot_x, target_y - robot_y
            return Action(atan2(dy, dx), min(1.0, 2.0 * hypot(dx, dy) / observation.robot_max_speed))

        if observation.goal == NAVIGATION_GOAL:
            destination_x, destination_y = observation.entities["destination"].position
            dx, dy = destination_x - robot_x, destination_y - robot_y
            distance = hypot(dx, dy)
            if distance == 0:
                return Action(0.0, 0.0)
            # Turn away from nearby obstacles, without remembering their motion.
            steer_x, steer_y = dx / distance, dy / distance
            for name, measurement in observation.entities.items():
                if name.startswith("obstacle_"):
                    ox, oy = measurement.position[0] - robot_x, measurement.position[1] - robot_y
                    obstacle_distance = hypot(ox, oy)
                    if 0 < obstacle_distance < 1.2:
                        repulsion = 2.0 * (1.2 - obstacle_distance) / obstacle_distance
                        steer_x -= repulsion * ox
                        steer_y -= repulsion * oy
            return Action(
                atan2(steer_y, steer_x),
                min(1.0, 2.0 * distance / observation.robot_max_speed),
            )

        raise ValueError(f"Unsupported goal: {observation.goal!r}")

    return control
