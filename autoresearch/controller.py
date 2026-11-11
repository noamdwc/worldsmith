"""Reactive starter using only the current goal and position measurements."""

from math import atan2, hypot

from worldsmith.api import (
    BOUNCING_INTERCEPTION_GOAL, INTERCEPTION_GOAL, NAVIGATION_GOAL, OBJECT_PUSHING_GOAL,
    RENDEZVOUS_GOAL,
    Action, Controller, Observation,
)


def push_object(observation: Observation, robot_x: float, robot_y: float) -> Action:
    object_x, object_y = observation.entities["object"].position
    destination_x, destination_y = observation.entities["destination"].position
    dx, dy = destination_x - object_x, destination_y - object_y
    distance = hypot(dx, dy)
    if distance < 0.18:
        return Action(0.0, 0.0)

    forward_x, forward_y = dx / distance, dy / distance
    rx, ry = robot_x - object_x, robot_y - object_y
    along = rx * forward_x + ry * forward_y
    lateral = -rx * forward_y + ry * forward_x

    if along < -0.45 and abs(lateral) < 0.17:
        # Aim through the object and ease off so it can coast to the goal.
        return Action(atan2(-ry, -rx), min(1.0, distance / 1.5))

    # Approach a point behind the object. While still in front, stay off
    # the push line so the approach does not bump it away from the goal.
    behind = -0.8 - along
    side = -lateral
    if along > -0.45:
        clearance = 1.0 if lateral >= 0 else -1.0
        side = clearance - lateral
    steer_x = behind * forward_x - side * forward_y
    steer_y = behind * forward_y + side * forward_x
    return Action(
        atan2(steer_y, steer_x),
        min(1.0, 3.0 * hypot(steer_x, steer_y) / observation.robot_max_speed),
    )


def make_controller() -> Controller:
    def control(observation: Observation) -> Action:
        robot_x, robot_y = observation.entities[observation.robot_id].position
        if observation.goal in (INTERCEPTION_GOAL, BOUNCING_INTERCEPTION_GOAL):
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

        if observation.goal == OBJECT_PUSHING_GOAL:
            return push_object(observation, robot_x, robot_y)

        raise ValueError(f"Unsupported goal: {observation.goal!r}")

    return control
