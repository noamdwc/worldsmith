"""Explicit CLI task names and their lazy adapter loaders."""

from functools import partial
from importlib import import_module


def _load(module_name: str, *args):
    return import_module(module_name).evaluation_setup(*args)


_PYMUNK = "worldsmith.benchmark.pymunk_adapter"
_MANISKILL_TASKS = {
    "push_cube": partial(_load, "worldsmith.benchmark.tasks.push_cube"),
    "pick_cube": partial(_load, "worldsmith.benchmark.tasks.pick_cube"),
    "stack_cube": partial(_load, "worldsmith.benchmark.tasks.stack_cube"),
    "peg_insertion_side": partial(_load, "worldsmith.benchmark.tasks.peg_insertion_side"),
    "pull_cube": partial(_load, "worldsmith.benchmark.tasks.pull_cube"),
    "push_t": partial(_load, "worldsmith.benchmark.tasks.push_t"),
}
_PYMUNK_TASKS = {
    "interception": partial(_load, _PYMUNK, "interception"),
    "navigation": partial(_load, _PYMUNK, "navigation"),
    "rendezvous": partial(_load, _PYMUNK, "rendezvous"),
    "bouncing_interception": partial(_load, _PYMUNK, "bouncing_interception"),
    "object_pushing": partial(_load, _PYMUNK, "object_pushing"),
}

# Load only the adapters available in each pinned Python environment.
TASK_GROUPS = {"pymunk": _PYMUNK_TASKS, "maniskill": _MANISKILL_TASKS}
TASK_CATALOG = {**_MANISKILL_TASKS, **_PYMUNK_TASKS}
