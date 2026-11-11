"""Top-down Pymunk simulation and noisy position sensing."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from random import Random

import pymunk

from worldsmith.api import Action, Observation, PositionMeasurement, Vec2

TIME_STEP = 0.1
_BODY_TYPES = {
    pymunk.Body.KINEMATIC: "kinematic",
    pymunk.Body.DYNAMIC: "dynamic",
    pymunk.Body.STATIC: "static",
}


class InvalidAction(ValueError):
    """A controller command violates the Action contract."""


@dataclass(frozen=True, slots=True)
class Entity:
    position: Vec2
    velocity: Vec2
    position_noise: float


@dataclass(frozen=True, slots=True)
class PhysicalEntity:
    """A fresh Pymunk body with its collision shapes already attached."""

    body: pymunk.Body
    position_noise: float | None = None  # None omits the body from observations.


class Simulation:
    def __init__(
        self, entities: Mapping[str, Entity | PhysicalEntity], robot_id: str,
        robot_max_speed: float, rng: Random, *, goal: str,
    ):
        robot = entities[robot_id]
        if isinstance(robot, PhysicalEntity) and (
            robot.body.body_type != pymunk.Body.KINEMATIC or robot.position_noise is None
        ):
            raise ValueError("The Action-controlled robot must be kinematic and observable")

        self._space = pymunk.Space()
        self._bodies = {}
        self._physical_bodies = {}
        self._sensors = {}
        for entity_id, entity in entities.items():
            if isinstance(entity, PhysicalEntity):
                body = entity.body
                self._space.add(body, *body.shapes)
                self._physical_bodies[entity_id] = body
            else:
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = entity.position
                body.velocity = entity.velocity
                self._space.add(body)
            self._bodies[entity_id] = body
            sigma = entity.position_noise
            if sigma is not None:
                self._sensors[entity_id] = (body, sigma)

        self._robot_id = robot_id
        self._robot_max_speed = robot_max_speed
        self._goal = goal
        self._steps = 0
        self._rng = rng
        self.initial_observation = self._observe()

    def position(self, entity_id: str) -> Vec2:
        x, y = self._bodies[entity_id].position
        return x, y

    def ground_truth(self) -> dict[str, dict[str, object]]:
        state: dict[str, dict[str, object]] = {
            name: {"position": tuple(body.position), "velocity": tuple(body.velocity)}
            for name, body in self._bodies.items()
        }
        for name, body in self._physical_bodies.items():
            state[name].update(
                body_type=_BODY_TYPES[body.body_type],
                angle=body.angle,
                angular_velocity=body.angular_velocity,
                shapes=[
                    {
                        "type": type(shape).__name__,
                        "friction": shape.friction,
                        "elasticity": shape.elasticity,
                    }
                    for shape in body.shapes
                ],
            )
            if body.body_type == pymunk.Body.DYNAMIC:
                # Pymunk uses an infinite moment to lock rotation. Use JSON null in traces.
                moment = body.moment
                state[name].update(mass=body.mass, moment=None if math.isinf(moment) else moment)
        return state

    def step(self, action: Action) -> Observation:
        if not math.isfinite(action.heading) or not 0 <= action.throttle <= 1:
            raise InvalidAction("Action requires a finite heading and throttle in [0, 1]")

        self._bodies[self._robot_id].velocity = pymunk.Vec2d(
            self._robot_max_speed * action.throttle, 0
        ).rotated(action.heading)
        self._space.step(TIME_STEP)
        self._steps += 1
        return self._observe()

    def _observe(self) -> Observation:
        measurements = {}
        for entity_id, (body, sigma) in self._sensors.items():
            x, y = body.position
            measurements[entity_id] = PositionMeasurement(
                position=(self._rng.gauss(x, sigma), self._rng.gauss(y, sigma)),
                covariance=((sigma**2, 0.0), (0.0, sigma**2)),
            )
        return Observation(
            time=self._steps * TIME_STEP,
            robot_id=self._robot_id,
            entities=measurements,
            robot_max_speed=self._robot_max_speed,
            goal=self._goal,
        )
