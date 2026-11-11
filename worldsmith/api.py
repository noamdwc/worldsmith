"""The observation and action contract shared by controllers and the benchmark."""

from collections.abc import Callable
from dataclasses import dataclass

type Vec2 = tuple[float, float]

INTERCEPTION_GOAL = "interception"
BOUNCING_INTERCEPTION_GOAL = "bouncing_interception"
NAVIGATION_GOAL = "navigation"
RENDEZVOUS_GOAL = "rendezvous"


@dataclass(frozen=True, slots=True)
class PositionMeasurement:
    position: Vec2
    covariance: tuple[Vec2, Vec2]


@dataclass(frozen=True, slots=True)
class Observation:
    time: float
    robot_id: str
    entities: dict[str, PositionMeasurement]
    robot_max_speed: float
    goal: str = INTERCEPTION_GOAL


@dataclass(frozen=True, slots=True)
class Action:
    heading: float  # Radians from the positive x axis.
    throttle: float = 1.0  # Fraction of robot_max_speed, in [0, 1].


type Controller = Callable[[Observation], Action]
type ControllerFactory = Callable[[], Controller]
