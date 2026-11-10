"""Shared goal-aware starter: chase, predict obstacle motion, or match target motion."""

from collections import deque
from math import atan2, cos, hypot, pi, sin

from worldsmith.api import (
    INTERCEPTION_GOAL, NAVIGATION_GOAL, RENDEZVOUS_GOAL, Action, Controller, Observation,
)


def make_controller() -> Controller:
    history: deque[Observation] = deque(maxlen=6)

    def velocity(entity_id: str) -> tuple[float, float]:
        first, last = history[0], history[-1]
        if first.time == last.time:
            return 0.0, 0.0
        x0, y0 = first.entities[entity_id].position
        x1, y1 = last.entities[entity_id].position
        elapsed = last.time - first.time
        return (x1 - x0) / elapsed, (y1 - y0) / elapsed

    def control(observation: Observation) -> Action:
        history.append(observation)
        robot_x, robot_y = observation.entities[observation.robot_id].position
        speed = observation.robot_max_speed

        if observation.goal == INTERCEPTION_GOAL:
            # Preserve the original Chase baseline exactly for interception.
            target_x, target_y = observation.entities["target"].position
            return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

        if observation.goal == RENDEZVOUS_GOAL:
            target_x, target_y = observation.entities["target"].position
            vx, vy = velocity("target")
            # Feed forward estimated target motion, then close the position error.
            vx += 2.0 * (target_x - robot_x)
            vy += 2.0 * (target_y - robot_y)
            return Action(atan2(vy, vx), min(1.0, hypot(vx, vy) / speed))

        if observation.goal == NAVIGATION_GOAL:
            destination_x, destination_y = observation.entities["destination"].position
            dx, dy = destination_x - robot_x, destination_y - robot_y
            heading = atan2(dy, dx)
            throttle = min(1.0, hypot(dx, dy) / (speed * 0.8))
            obstacles = []
            for name, measurement in observation.entities.items():
                if name.startswith("obstacle_"):
                    x, y = measurement.position
                    vx, vy = velocity(name)
                    obstacles.append((x - robot_x, y - robot_y, vx, vy))

            best_cost = float("inf")
            best_action = Action(heading, 0.0)
            # Test a few headings plus waiting. Penalize predicted closest approach
            # over a short horizon, so crossing obstacles can be passed on either side.
            for offset, candidate_throttle in [
                (offset, throttle) for offset in (0, -pi / 6, pi / 6, -pi / 3, pi / 3, -pi / 2, pi / 2)
            ] + [(0.0, 0.0)]:
                angle = heading + offset
                vx = speed * candidate_throttle * cos(angle)
                vy = speed * candidate_throttle * sin(angle)
                cost = hypot(dx - vx * 0.8, dy - vy * 0.8)
                for ox, oy, ovx, ovy in obstacles:
                    rvx, rvy = ovx - vx, ovy - vy
                    relative_speed_squared = rvx * rvx + rvy * rvy
                    closest_time = (
                        max(0.0, min(0.8, -(ox * rvx + oy * rvy) / relative_speed_squared))
                        if relative_speed_squared > 0 else 0.0
                    )
                    clearance = hypot(ox + rvx * closest_time, oy + rvy * closest_time)
                    cost += 20.0 * max(0.0, 1.0 - clearance) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_action = Action(angle, candidate_throttle)
            return best_action

        raise ValueError(f"Unsupported goal: {observation.goal!r}")

    return control
