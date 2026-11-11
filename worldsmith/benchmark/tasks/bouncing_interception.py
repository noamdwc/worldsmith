"""Interception of a dynamic target in a rectangular arena."""

import math
from random import Random

import pymunk

from worldsmith.api import BOUNCING_INTERCEPTION_GOAL
from worldsmith.benchmark.simulation import Entity, PhysicalEntity, Simulation
from worldsmith.benchmark.tasks.interception import TARGET_NOISE, Interception

TARGET_RADIUS = 0.35
WALL_RADIUS = 0.25


class BouncingInterception(Interception):
    goal = "Get within 0.3 simulation units of the bouncing target within 45 steps."

    def start(self, seed: int) -> Simulation:
        rng = Random(seed)
        half_width = rng.uniform(5.5, 6.5)
        half_height = rng.uniform(3.2, 4.2)
        direction = rng.choice((-1, 1))

        target = pymunk.Body(1.0, math.inf)
        target.position = direction * rng.uniform(2.0, 3.2), rng.uniform(-2.0, 2.0)
        # Start with outward motion so an early bounce can create a route to capture.
        heading = rng.uniform(-0.85, 0.85)
        speed = rng.uniform(3.4, 4.4)
        target.velocity = direction * speed * math.cos(heading), speed * math.sin(heading)
        # The target travels at most 0.44 units per step, less than the
        # combined shape radii, so it cannot cross a wall between steps.
        circle = pymunk.Circle(target, TARGET_RADIUS)
        circle.elasticity = 1.0

        entities = {
            "robot": Entity((0.0, 0.0), (0.0, 0.0), 0.04),
            "target": PhysicalEntity(target, observable=True, position_noise=TARGET_NOISE),
        }
        for name, midpoint, offset in (
            ("wall_left", (-half_width, 0), (0, half_height)),
            ("wall_right", (half_width, 0), (0, half_height)),
            ("wall_bottom", (0, -half_height), (half_width, 0)),
            ("wall_top", (0, half_height), (half_width, 0)),
        ):
            wall = pymunk.Body(body_type=pymunk.Body.STATIC)
            wall.position = midpoint
            end = pymunk.Vec2d(*offset)
            segment = pymunk.Segment(wall, -end, end, WALL_RADIUS)
            segment.elasticity = rng.uniform(0.85, 0.97)
            entities[name] = PhysicalEntity(wall, observable=True)

        return Simulation(
            entities, "robot", 2.4, rng, goal=BOUNCING_INTERCEPTION_GOAL,
        )
