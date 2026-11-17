"""Pymunk benchmark tasks and their artifact IDs."""

from worldsmith.api import (
    BOUNCING_INTERCEPTION_GOAL, INTERCEPTION_GOAL, NAVIGATION_GOAL, OBJECT_PUSHING_GOAL,
    RENDEZVOUS_GOAL,
)
from worldsmith.benchmark.tasks.bouncing_interception import BouncingInterception
from worldsmith.benchmark.tasks.interception import BENCHMARK, Interception
from worldsmith.benchmark.tasks.navigation import Navigation
from worldsmith.benchmark.tasks.object_pushing import ObjectPushing
from worldsmith.benchmark.tasks.rendezvous import Rendezvous

TASKS = {
    INTERCEPTION_GOAL: Interception,
    NAVIGATION_GOAL: Navigation,
    RENDEZVOUS_GOAL: Rendezvous,
    BOUNCING_INTERCEPTION_GOAL: BouncingInterception,
    OBJECT_PUSHING_GOAL: ObjectPushing,
}
BENCHMARKS = {
    INTERCEPTION_GOAL: BENCHMARK,
    NAVIGATION_GOAL: "navigation-v1",
    RENDEZVOUS_GOAL: "rendezvous-v1",
    BOUNCING_INTERCEPTION_GOAL: "bouncing-interception-v1",
    OBJECT_PUSHING_GOAL: "object-pushing-v1",
}
