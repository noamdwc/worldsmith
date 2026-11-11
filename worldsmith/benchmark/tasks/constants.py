"""The public benchmark tasks and their fixed evaluation seed sets."""

from worldsmith.api import (
    BOUNCING_INTERCEPTION_GOAL, INTERCEPTION_GOAL, NAVIGATION_GOAL, OBJECT_PUSHING_GOAL,
    RENDEZVOUS_GOAL,
)
from worldsmith.benchmark.tasks.bouncing_interception import BouncingInterception
from worldsmith.benchmark.tasks.interception import BENCHMARK, SPLITS, Interception
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
# Keep the independent interception benchmark unchanged; the combined run uses
# a small subset for quick research iterations. No held-out seeds enter dev.
COMBINED_SPLITS = {"dev": range(8), "heldout": range(1000, 1008)}
TASK_SPLITS = {
    INTERCEPTION_GOAL: SPLITS,
    NAVIGATION_GOAL: COMBINED_SPLITS,
    RENDEZVOUS_GOAL: COMBINED_SPLITS,
    BOUNCING_INTERCEPTION_GOAL: COMBINED_SPLITS,
    OBJECT_PUSHING_GOAL: COMBINED_SPLITS,
}
COMBINED_BENCHMARK = "worldsmith-v3"
