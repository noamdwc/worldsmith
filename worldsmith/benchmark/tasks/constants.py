"""The three public V1 tasks and their fixed evaluation seed sets."""

from worldsmith.api import INTERCEPTION_GOAL, NAVIGATION_GOAL, RENDEZVOUS_GOAL
from worldsmith.benchmark.tasks.interception import BENCHMARK, SPLITS, Interception
from worldsmith.benchmark.tasks.navigation import Navigation
from worldsmith.benchmark.tasks.rendezvous import Rendezvous

TASKS = {
    INTERCEPTION_GOAL: Interception,
    NAVIGATION_GOAL: Navigation,
    RENDEZVOUS_GOAL: Rendezvous,
}
BENCHMARKS = {
    INTERCEPTION_GOAL: BENCHMARK,
    NAVIGATION_GOAL: "navigation-v1",
    RENDEZVOUS_GOAL: "rendezvous-v1",
}
# Keep the independent interception benchmark unchanged; the combined run uses
# a small subset for quick research iterations. No held-out seeds enter dev.
COMBINED_SPLITS = {"dev": range(8), "heldout": range(1000, 1008)}
TASK_SPLITS = {
    INTERCEPTION_GOAL: SPLITS,
    NAVIGATION_GOAL: COMBINED_SPLITS,
    RENDEZVOUS_GOAL: COMBINED_SPLITS,
}
COMBINED_BENCHMARK = "worldsmith-v1"
