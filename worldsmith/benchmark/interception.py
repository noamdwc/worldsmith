"""Fixed top-down interception scenario with noisy position measurements."""

import math
from random import Random

import pymunk

from worldsmith.api import Action, Observation, PositionMeasurement

TIME_STEP = 0.1
STEP_LIMIT = 45
CAPTURE_RADIUS = 0.3
ROBOT_MAX_SPEED = 2.4
ROBOT_NOISE = 0.04
TARGET_NOISE = 0.12


class Interception:
    def __init__(self, seed: int):
        rng = Random(seed)
        self._space = pymunk.Space()
        self._robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._target = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._space.add(self._robot, self._target)

        bearing = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(5.0, 8.0)
        self._target.position = pymunk.Vec2d(radius, 0).rotated(bearing)
        # Sample crossing paths with some variation in their radial motion.
        heading = bearing + rng.choice((-1, 1)) * math.pi / 2 + rng.uniform(-0.25, 0.25)
        speed = rng.uniform(1.1, 1.7)
        self._target.velocity = pymunk.Vec2d(speed, 0).rotated(heading)

        self._steps = 0
        self._rng = rng
        self.initial_observation = self._observe()

    @property
    def distance(self) -> float:
        return self._robot.position.get_distance(self._target.position)

    def step(self, action: Action) -> Observation:
        if not math.isfinite(action.heading) or not 0 <= action.throttle <= 1:
            raise ValueError("Action requires a finite heading and throttle in [0, 1]")

        self._robot.velocity = pymunk.Vec2d(ROBOT_MAX_SPEED * action.throttle, 0).rotated(
            action.heading
        )
        self._space.step(TIME_STEP)
        self._steps += 1
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            time=self._steps * TIME_STEP,
            robot=self._measure(self._robot.position, ROBOT_NOISE),
            target=self._measure(self._target.position, TARGET_NOISE),
            robot_max_speed=ROBOT_MAX_SPEED,
        )

    def _measure(self, position: pymunk.Vec2d, sigma: float) -> PositionMeasurement:
        return PositionMeasurement(
            position=(self._rng.gauss(position.x, sigma), self._rng.gauss(position.y, sigma)),
            covariance=((sigma**2, 0.0), (0.0, sigma**2)),
        )
