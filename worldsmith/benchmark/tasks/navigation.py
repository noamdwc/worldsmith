"""Reach a fixed destination without touching crossing moving obstacles."""

import math
from random import Random
from statistics import mean

from worldsmith.api import NAVIGATION_GOAL
from worldsmith.benchmark.simulation import Entity, Simulation

DESTINATION_RADIUS = 0.35
COLLISION_DISTANCE = 0.7  # Robot radius 0.15 plus obstacle radius 0.55.


class Navigation:
    objective = "Reach destination within 0.35 units in 80 steps without touching an obstacle."
    step_limit = 80

    def start(self, seed: int) -> Simulation:
        rng = Random(seed)
        # Vary crossing timing and speed so local steering can react too late.
        # The open plane and 8-second budget still allow detours or waiting.
        return Simulation(
            entities={
                "robot": Entity((0.0, 0.0), (0.0, 0.0), 0.02),
                "destination": Entity((rng.uniform(8.0, 10.0), rng.uniform(-1.0, 1.0)), (0.0, 0.0), 0.0),
                "obstacle_0": Entity(
                    (rng.uniform(2.5, 4.0), rng.uniform(-2.5, -1.0)),
                    (0.0, rng.uniform(0.9, 1.7)), 0.04,
                ),
                "obstacle_1": Entity(
                    (rng.uniform(5.0, 6.5), rng.uniform(1.0, 3.0)),
                    (0.0, -rng.uniform(0.8, 1.6)), 0.04,
                ),
            },
            robot_id="robot", robot_max_speed=2.4, rng=rng, goal=NAVIGATION_GOAL,
        )

    def measure(self, world: Simulation) -> dict[str, float]:
        robot = world.position("robot")
        return {
            "destination_distance": math.dist(robot, world.position("destination")),
            "obstacle_clearance": min(
                math.dist(robot, world.position(name)) - COLLISION_DISTANCE
                for name in ("obstacle_0", "obstacle_1")
            ),
        }

    def succeeded(self, measurements: list[dict[str, float]]) -> bool:
        return (
            measurements[-1]["destination_distance"] <= DESTINATION_RADIUS
            and all(frame["obstacle_clearance"] > 0 for frame in measurements)
        )

    def episode_metrics(self, measurements: list[dict[str, float]], success: bool) -> dict:
        return {
            "success": success,
            "collided": any(frame["obstacle_clearance"] <= 0 for frame in measurements),
            "final_destination_distance": measurements[-1]["destination_distance"],
            "minimum_obstacle_clearance": min(frame["obstacle_clearance"] for frame in measurements),
        }

    def summarize(self, episodes: list[dict]) -> dict:
        return {
            "success_rate": mean(episode["success"] for episode in episodes),
            "collision_rate": mean(episode["collided"] for episode in episodes),
            "mean_final_destination_distance": mean(
                episode["final_destination_distance"] for episode in episodes
            ),
            "mean_minimum_obstacle_clearance": mean(
                episode["minimum_obstacle_clearance"] for episode in episodes
            ),
        }
