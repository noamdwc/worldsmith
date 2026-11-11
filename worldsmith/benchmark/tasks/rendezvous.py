"""Approach a moving target and sustain proximity rather than just intercept it."""

import math
from random import Random
from statistics import mean

from worldsmith.api import RENDEZVOUS_GOAL
from worldsmith.benchmark.simulation import TIME_STEP, Entity, Simulation

TRACKING_RADIUS = 0.6
HOLD_STEPS = 20


def longest_tracking_streak(measurements: list[dict[str, float]]) -> int:
    streak = longest = 0
    # The initial measurement does not represent an elapsed control step.
    for frame in measurements[1:]:
        streak = streak + 1 if frame["distance"] <= TRACKING_RADIUS else 0
        longest = max(longest, streak)
    return longest


class Rendezvous:
    goal = "Stay within 0.6 units of target for 20 consecutive steps within 100 steps."
    step_limit = 100

    def start(self, seed: int) -> Simulation:
        rng = Random(seed)
        bearing = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(3.0, 5.0)
        heading = bearing + rng.uniform(-math.pi, math.pi)
        # Faster targets expose steady tracking lag in proportional pursuit.
        # Even the fastest outbound target can be caught within the budget:
        # 5 / (2.4 - 1.7) + 2 seconds holding is less than 10 seconds.
        speed = rng.uniform(0.8, 1.7)
        return Simulation(
            entities={
                "robot": Entity((0.0, 0.0), (0.0, 0.0), 0.02),
                "target": Entity(
                    (radius * math.cos(bearing), radius * math.sin(bearing)),
                    (speed * math.cos(heading), speed * math.sin(heading)), 0.08,
                ),
            },
            robot_id="robot", robot_max_speed=2.4, rng=rng, goal=RENDEZVOUS_GOAL,
        )

    def measure(self, world: Simulation) -> dict[str, float]:
        return {"distance": math.dist(world.position("robot"), world.position("target"))}

    def succeeded(self, measurements: list[dict[str, float]]) -> bool:
        return longest_tracking_streak(measurements) >= HOLD_STEPS

    def episode_metrics(self, measurements: list[dict[str, float]], success: bool) -> dict:
        return {
            "success": success,
            "longest_tracking_seconds": longest_tracking_streak(measurements) * TIME_STEP,
            "mean_distance": mean(frame["distance"] for frame in measurements[1:]),
            "final_distance": measurements[-1]["distance"],
        }

    def summarize(self, episodes: list[dict]) -> dict:
        return {
            "success_rate": mean(episode["success"] for episode in episodes),
            "mean_longest_tracking_seconds": mean(
                episode["longest_tracking_seconds"] for episode in episodes
            ),
            "mean_distance": mean(episode["mean_distance"] for episode in episodes),
            "mean_final_distance": mean(episode["final_distance"] for episode in episodes),
        }
