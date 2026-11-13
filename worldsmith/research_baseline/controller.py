"""Shared starter for planar movement and tabletop manipulation tasks."""

from math import atan2, hypot

import numpy as np

from worldsmith.api import (
    BOUNCING_INTERCEPTION_GOAL,
    INTERCEPTION_GOAL,
    NAVIGATION_GOAL,
    OBJECT_PUSHING_GOAL,
    RENDEZVOUS_GOAL,
    Action,
    Observation,
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


def make_controller(*, task=None):
    if task in ("open_middle_drawer", "place_bowl_on_plate", "place_two_items_in_basket"):
        return lambda observation: np.array([0., 0., 0., 0., 0., 0., -1.])

    if task in ("push_cube", "pull_cube", "pick_cube", "stack_cube", "peg_insertion_side", "push_t"):
        return manipulation_controller(task)

    def control(observation: Observation) -> Action:
        goal = task if task is not None else observation.goal
        robot_x, robot_y = observation.entities[observation.robot_id].position
        if goal in (INTERCEPTION_GOAL, BOUNCING_INTERCEPTION_GOAL):
            target_x, target_y = observation.entities["target"].position
            return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

        if goal == RENDEZVOUS_GOAL:
            target_x, target_y = observation.entities["target"].position
            dx, dy = target_x - robot_x, target_y - robot_y
            return Action(atan2(dy, dx), min(1.0, 2.0 * hypot(dx, dy) / observation.robot_max_speed))

        if goal == NAVIGATION_GOAL:
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

        if goal == OBJECT_PUSHING_GOAL:
            return push_object(observation, robot_x, robot_y)

        raise ValueError(f"Unsupported goal: {goal!r}")

    return control


def manipulation_controller(task):
    waypoint = 0
    closing_steps = 0

    def control(observation):
        nonlocal waypoint, closing_steps
        extra = observation["extra"]
        tcp = extra["tcp_pose"][0, :3].numpy()
        gripper = -1.0

        if task in ("push_cube", "pull_cube"):
            cube = extra["obj_pose"][0, :3].numpy()
            goal = extra["goal_pos"][0].numpy()
            approach_x = cube[0] + (-0.055 if task == "push_cube" else 0.055)
            targets = (
                (approach_x, cube[1], 0.10),
                (approach_x, cube[1], 0.025),
                (goal[0], goal[1], 0.025),
            )
            if waypoint < 2 and np.linalg.norm(targets[waypoint] - tcp) < 0.01:
                waypoint += 1
            target = targets[waypoint]

        elif task == "pick_cube":
            cube = extra["obj_pose"][0, :3].numpy()
            goal = extra["goal_pos"][0].numpy()
            if extra["is_grasped"][0]:
                target = tcp + goal - cube
            elif np.linalg.norm(tcp[:2] - cube[:2]) > 0.012:
                target = cube + (0, 0, 0.08)
                gripper = 1.0
            else:
                target = cube
                gripper = -1.0 if np.linalg.norm(tcp - cube) < 0.012 else 1.0

        elif task in ("stack_cube", "peg_insertion_side"):
            # Grasp from above. StackCube also transports and releases the cube.
            peg = task == "peg_insertion_side"
            pose_key = "peg_pose" if peg else "cubeA_pose"
            obj = extra[pose_key][0, :3].numpy()
            lift = 0.15 if peg else 0.14
            targets = (obj + (0, 0, 0.08), obj, (obj[0], obj[1], lift))
            if not peg:
                base = extra["cubeB_pose"][0, :3].numpy()
                targets += ((base[0], base[1], lift), base + (0, 0, 0.043), base + (0, 0, 0.12))

            if waypoint == 1:
                if np.linalg.norm(tcp - obj) < 0.012 or closing_steps:
                    closing_steps += 1
                    if closing_steps == 6:
                        waypoint += 1
            elif waypoint < len(targets) - 1 and np.linalg.norm(tcp - targets[waypoint]) < 0.01:
                waypoint += 1
            target = targets[waypoint]
            gripper = -1.0 if closing_steps and waypoint < 5 else 1.0

        else:  # PushT uses a translation-only stick.
            tee = extra["obj_pose"][0, :2].numpy()
            goal = extra["goal_pos"][0, :2].numpy()
            direction = goal - tee
            distance = np.linalg.norm(direction)
            if distance < 0.01:
                return np.zeros(3, dtype=np.float32)
            approach = tee - 0.14 * direction / distance
            targets = ((*approach, 0.09), (*approach, 0.018), (*goal, 0.018))
            if waypoint < 2 and np.linalg.norm(targets[waypoint] - tcp) < 0.01:
                waypoint += 1
            target = targets[waypoint]

        xyz = np.clip((target - tcp) / 0.1, -0.5, 0.5)
        if task == "push_t":
            return xyz.astype(np.float32)
        if task == "peg_insertion_side":
            return np.array([*xyz, 0, 0, 0, gripper], dtype=np.float32)
        return np.array([*xyz, gripper], dtype=np.float32)

    return control
