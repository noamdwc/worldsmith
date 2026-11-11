"""Top-down kinematic simulation and noisy position sensing."""

import math
from dataclasses import dataclass
from random import Random

import pymunk

from worldsmith.api import Action, Observation, PositionMeasurement, Vec2

TIME_STEP = 0.1


class InvalidAction(ValueError):
    """A controller command violates the Action contract."""


@dataclass(frozen=True, slots=True)
class Entity:
    position: Vec2
    velocity: Vec2
    position_noise: float


class Simulation:
    def __init__(
        self, entities: dict[str, Entity], robot_id: str, robot_max_speed: float, rng: Random,
        *, goal: str,
    ):
        self._space = pymunk.Space()
        self._bodies = {}
        self._noise = {}
        for entity_id, entity in entities.items():
            body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            body.position = entity.position
            body.velocity = entity.velocity
            self._space.add(body)
            self._bodies[entity_id] = body
            self._noise[entity_id] = entity.position_noise

        self._robot_id = robot_id
        self._robot_max_speed = robot_max_speed
        self._goal = goal
        self._steps = 0
        self._rng = rng
        self.initial_observation = self._observe()

    def position(self, entity_id: str) -> Vec2:
        x, y = self._bodies[entity_id].position
        return x, y

    def ground_truth(self) -> dict[str, dict[str, Vec2]]:
        return {
            name: {"position": tuple(body.position), "velocity": tuple(body.velocity)}
            for name, body in self._bodies.items()
        }

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
        for entity_id, body in self._bodies.items():
            x, y = body.position
            sigma = self._noise[entity_id]
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
