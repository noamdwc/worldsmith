"""Scene generation and scoring for moving-target interception."""

import math
from collections import deque
from random import Random
from statistics import mean

import pymunk

from worldsmith.api import Action, Observation
from worldsmith.benchmark.simulation import Entity, Simulation

BENCHMARK = "interception-v2"
SPLITS = {"dev": range(32), "heldout": range(1000, 1032)}
CAPTURE_RADIUS = 0.3
TARGET_NOISE = 0.12


class Interception:
    goal = "Get within 0.3 simulation units of the moving target within 45 steps."
    step_limit = 45

    def start(self, seed: int) -> Simulation:
        rng = Random(seed)
        bearing = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(5.0, 8.0)
        target_position = pymunk.Vec2d(radius, 0).rotated(bearing)
        # Sample crossing paths with some variation in their radial motion.
        heading = bearing + rng.choice((-1, 1)) * math.pi / 2 + rng.uniform(-0.25, 0.25)
        speed = rng.uniform(1.1, 1.7)
        target_velocity = pymunk.Vec2d(speed, 0).rotated(heading)
        return Simulation(
            entities={
                "robot": Entity((0.0, 0.0), (0.0, 0.0), 0.04),
                "target": Entity(tuple(target_position), tuple(target_velocity), TARGET_NOISE),
            },
            robot_id="robot",
            robot_max_speed=2.4,
            rng=rng,
        )

    def measure(self, world: Simulation) -> dict[str, float]:
        return {"distance": math.dist(world.position("robot"), world.position("target"))}

    def succeeded(self, measurements: list[dict[str, float]]) -> bool:
        return measurements[-1]["distance"] <= CAPTURE_RADIUS

    def episode_metrics(self, measurements: list[dict[str, float]], success: bool) -> dict:
        return {
            "captured": success,
            "closest_distance": min(frame["distance"] for frame in measurements),
        }

    def summarize(self, episodes: list[dict]) -> dict:
        captures = sum(episode["captured"] for episode in episodes)
        return {
            "captures": captures,
            "capture_rate": captures / len(episodes),
            "mean_closest_distance": mean(episode["closest_distance"] for episode in episodes),
        }


# Reference controllers for checking whether anticipation helps on this task.
class Chase:
    def __call__(self, observation: Observation) -> Action:
        rx, ry = observation.entities[observation.robot_id].position
        tx, ty = observation.entities["target"].position
        return Action(math.atan2(ty - ry, tx - rx))


class Lead:
    def __init__(self):
        self._history: deque[Observation] = deque(maxlen=9)

    def __call__(self, observation: Observation) -> Action:
        self._history.append(observation)
        rx, ry = observation.entities[observation.robot_id].position
        tx, ty = observation.entities["target"].position
        dx, dy = tx - rx, ty - ry

        if len(self._history) == self._history.maxlen:
            first = self._history[0]
            old_x, old_y = first.entities["target"].position
            elapsed = observation.time - first.time
            # A longer measurement interval reduces noise in the finite difference.
            lookahead = math.hypot(dx, dy) / observation.robot_max_speed
            dx += (tx - old_x) / elapsed * lookahead
            dy += (ty - old_y) / elapsed * lookahead

        return Action(math.atan2(dy, dx))
