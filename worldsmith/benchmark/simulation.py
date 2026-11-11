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
    """A Pymunk body outside any space, with optional collision shapes attached."""

    body: pymunk.Body
    observable: bool = False
    position_noise: float = 0.0


class Simulation:
    def __init__(
        self, entities: Mapping[str, Entity | PhysicalEntity], robot_id: str,
        robot_max_speed: float, rng: Random, *, goal: str, damping: float = 1.0,
    ):
        robot = entities[robot_id]
        if isinstance(robot, PhysicalEntity) and (
            robot.body.body_type != pymunk.Body.KINEMATIC or not robot.observable
        ):
            raise ValueError("The Action-controlled robot must be kinematic and observable")

        self._space = pymunk.Space()
        self._space.damping = damping
        self._bodies = {}
        self._physical_bodies = {}
        self._sensors = {}
        for entity_id, entity in entities.items():
            if isinstance(entity, PhysicalEntity):
                body = entity.body
                self._space.add(body, *body.shapes)
                self._physical_bodies[entity_id] = body
                observable = entity.observable
            else:
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = entity.position
                body.velocity = entity.velocity
                self._space.add(body)
                observable = True
            self._bodies[entity_id] = body
            if observable:
                self._sensors[entity_id] = (body, entity.position_noise)

        self._robot_id = robot_id
        self._robot_max_speed = robot_max_speed
        self._goal = goal
        self._steps = 0
        self._rng = rng
        self.initial_observation = self._observe()

    def position(self, entity_id: str) -> Vec2:
        x, y = self._bodies[entity_id].position
        return x, y

    def speed(self, entity_id: str) -> float:
        return self._bodies[entity_id].velocity.length

    def contact_impulse(self, entity_id: str, other_id: str | None = None) -> float:
        """Sum this step's collision impulse magnitudes, optionally for one body pair."""
        # Traverse the same arbiters in the same order regardless of argument order.
        if other_id is not None and other_id < entity_id:
            entity_id, other_id = other_id, entity_id
        body = self._bodies[entity_id]
        other = self._bodies[other_id] if other_id is not None else None
        if other is body:
            return 0.0

        impulse = 0.0

        def add_impulse(arbiter: pymunk.Arbiter) -> None:
            nonlocal impulse
            if other is not None:
                first, second = arbiter.shapes
                if first.body is not other and second.body is not other:
                    return
            impulse += arbiter.total_impulse.length

        body.each_arbiter(add_impulse)
        return impulse

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
                if self._space.damping != 1.0:
                    state[name]["velocity_retention_per_second"] = self._space.damping
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
