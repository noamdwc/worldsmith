"""Push a dynamic circle across an open plane and let it settle."""

import math
from random import Random
from statistics import mean

import pymunk

from worldsmith.api import OBJECT_PUSHING_GOAL
from worldsmith.benchmark.simulation import TIME_STEP, Entity, PhysicalEntity, Simulation
from worldsmith.benchmark.tasks.task import Measurements

ROBOT_RADIUS = 0.22
OBJECT_RADIUS = 0.34
DESTINATION_RADIUS = 0.25
SETTLED_SPEED = 0.15
SETTLE_STEPS = 8


class ObjectPushing:
    objective = (
        "Move the object within 0.25 units of the destination at 0.15 units/second or less "
        "for 8 consecutive steps within 140 steps."
    )
    step_limit = 140

    def start(self, seed: int) -> Simulation:
        rng = Random(seed)
        object_position = pymunk.Vec2d(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
        direction = rng.uniform(-math.pi, math.pi)
        destination = object_position + pymunk.Vec2d(rng.uniform(3.0, 5.0), 0).rotated(direction)

        # Start on the destination side so the robot must get around the object.
        robot_bearing = direction + rng.choice((-1, 1)) * rng.uniform(0.45, 1.55)
        robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        robot.position = object_position + pymunk.Vec2d(rng.uniform(1.5, 3.5), 0).rotated(robot_bearing)
        robot_shape = pymunk.Circle(robot, ROBOT_RADIUS)
        robot_shape.friction = 0.6

        mass = rng.uniform(0.8, 1.6)
        movable = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, OBJECT_RADIUS))
        movable.position = object_position
        object_shape = pymunk.Circle(movable, OBJECT_RADIUS)
        object_shape.friction = rng.uniform(0.35, 0.8)

        # For linear drag F = -c*v, velocity decays as exp(-c*t/mass).
        # Pymunk takes the velocity fraction retained after one second.
        damping = math.exp(-rng.uniform(0.55, 4.4) / mass)
        return Simulation(
            {
                "robot": PhysicalEntity(robot, observable=True, position_noise=0.02),
                "object": PhysicalEntity(movable, observable=True, position_noise=0.04),
                "destination": Entity(tuple(destination), (0.0, 0.0), 0.0),
            },
            "robot", 2.4, rng, goal=OBJECT_PUSHING_GOAL, damping=damping,
        )

    def measure(self, world: Simulation) -> dict[str, float]:
        return {
            "object_distance": math.dist(world.position("object"), world.position("destination")),
            "object_speed": world.speed("object"),
            "contact_impulse": world.contact_impulse("robot", "object"),
        }

    def succeeded(self, measurements: Measurements) -> bool:
        return len(measurements) > SETTLE_STEPS and all(
            frame["object_distance"] <= DESTINATION_RADIUS and frame["object_speed"] <= SETTLED_SPEED
            for frame in measurements[-SETTLE_STEPS:]
        )

    def episode_metrics(self, measurements: Measurements, success: bool) -> dict:
        return {
            "success": success,
            "final_object_distance": measurements[-1]["object_distance"],
            "minimum_object_distance": min(frame["object_distance"] for frame in measurements),
            "final_object_speed": measurements[-1]["object_speed"],
            "contact_seconds": TIME_STEP * sum(frame["contact_impulse"] > 0 for frame in measurements[1:]),
        }

    def summarize(self, episodes: list[dict]) -> dict:
        return {
            "success_rate": mean(episode["success"] for episode in episodes),
            "mean_final_object_distance": mean(episode["final_object_distance"] for episode in episodes),
            "mean_minimum_object_distance": mean(episode["minimum_object_distance"] for episode in episodes),
            "mean_final_object_speed": mean(episode["final_object_speed"] for episode in episodes),
            "mean_contact_seconds": mean(episode["contact_seconds"] for episode in episodes),
        }
