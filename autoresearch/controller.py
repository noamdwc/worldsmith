"""Estimate constant target motion and steer toward its future position."""

from math import atan2, hypot, sqrt

from worldsmith.api import Action, Controller, Observation


def make_controller() -> Controller:
    target_history: list[tuple[float, float, float]] = []

    def intercept(observation: Observation) -> Action:
        robot_x, robot_y = observation.entities[observation.robot_id].position
        target_x, target_y = observation.entities["target"].position
        target_history.append((observation.time, target_x, target_y))

        # Wait for a short baseline so differenced measurement noise does not
        # dominate the initial velocity estimate. All target variances are equal
        # in the development traces, so ordinary least squares is appropriate.
        if len(target_history) >= 5:
            count = len(target_history)
            mean_time = sum(t for t, _, _ in target_history) / count
            mean_x = sum(x for _, x, _ in target_history) / count
            mean_y = sum(y for _, _, y in target_history) / count
            time_spread = sum((t - mean_time) ** 2 for t, _, _ in target_history)
            velocity_x = sum(
                (t - mean_time) * (x - mean_x) for t, x, _ in target_history
            ) / time_spread
            velocity_y = sum(
                (t - mean_time) * (y - mean_y) for t, _, y in target_history
            ) / time_spread
            target_x = mean_x + velocity_x * (observation.time - mean_time)
            target_y = mean_y + velocity_y * (observation.time - mean_time)
            relative_x = target_x - robot_x
            relative_y = target_y - robot_y
            distance_squared = relative_x**2 + relative_y**2
            speed_margin = observation.robot_max_speed**2 - velocity_x**2 - velocity_y**2
            travel_time = hypot(relative_x, relative_y) / observation.robot_max_speed
            # The positive root exists for a faster pursuer. A noisy estimate
            # can exceed robot speed; keep the finite approximate lead then.
            if speed_margin > 0:
                position_velocity = relative_x * velocity_x + relative_y * velocity_y
                travel_time = (
                    position_velocity
                    + sqrt(position_velocity**2 + speed_margin * distance_squared)
                ) / speed_margin
            # If interception lies beyond the public 45-step deadline, aim at
            # the target's deadline position to minimize the remaining gap.
            travel_time = min(travel_time, 4.5 - observation.time)
            target_x += velocity_x * travel_time
            target_y += velocity_y * travel_time

        return Action(heading=atan2(target_y - robot_y, target_x - robot_x))

    return intercept
