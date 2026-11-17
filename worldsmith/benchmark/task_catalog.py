"""Explicit CLI task names and their lazy adapter loaders."""

from functools import partial
from importlib import import_module


def _load(module_name: str, *args):
    return import_module(module_name).evaluation_setup(*args)


_PYMUNK = "worldsmith.benchmark.adapters.pymunk_adapter"
_MANISKILL_TASKS = {
    "push_cube": partial(_load, "worldsmith.benchmark.tasks.push_cube"),
    "pick_cube": partial(_load, "worldsmith.benchmark.tasks.pick_cube"),
    "stack_cube": partial(_load, "worldsmith.benchmark.tasks.stack_cube"),
    "peg_insertion_side": partial(_load, "worldsmith.benchmark.tasks.peg_insertion_side"),
    "pull_cube": partial(_load, "worldsmith.benchmark.tasks.pull_cube"),
    "push_t": partial(_load, "worldsmith.benchmark.tasks.push_t"),
    "roll_ball": partial(_load, "worldsmith.benchmark.tasks.roll_ball"),
    "quadruped_reach": partial(_load, "worldsmith.benchmark.tasks.quadruped_reach"),
}
_PYMUNK_TASKS = {
    "interception": partial(_load, _PYMUNK, "interception"),
    "navigation": partial(_load, _PYMUNK, "navigation"),
    "rendezvous": partial(_load, _PYMUNK, "rendezvous"),
    "bouncing_interception": partial(_load, _PYMUNK, "bouncing_interception"),
    "object_pushing": partial(_load, _PYMUNK, "object_pushing"),
}

_LIBERO = "worldsmith.benchmark.adapters.libero_adapter"
_LIBERO_TASKS = {
    "open_middle_drawer": partial(_load, _LIBERO, "open_middle_drawer"),
    "place_bowl_on_plate": partial(_load, _LIBERO, "place_bowl_on_plate"),
    "place_two_items_in_basket": partial(_load, _LIBERO, "place_two_items_in_basket"),
}

_ROBOSUITE = "worldsmith.benchmark.adapters.robosuite_adapter"
_ROBOSUITE_TASKS = {
    "open_latched_door": partial(_load, _ROBOSUITE, "open_latched_door"),
    "assemble_nuts": partial(_load, _ROBOSUITE, "assemble_nuts"),
    "handover_hammer": partial(_load, _ROBOSUITE, "handover_hammer"),
    "wipe_table": partial(_load, _ROBOSUITE, "wipe_table"),
}

# Load only the adapters available in each pinned Python environment.
TASK_GROUPS = {
    "pymunk": _PYMUNK_TASKS,
    "maniskill": _MANISKILL_TASKS,
    "libero": _LIBERO_TASKS,
    "robosuite": _ROBOSUITE_TASKS,
}
TASK_CATALOG = {task: load for group in TASK_GROUPS.values() for task, load in group.items()}
DEFAULT_WORKERS = {"pymunk": 1, "maniskill": 1, "libero": 2, "robosuite": 2}

VALIDATION_ONLY_TASKS = {
    "rendezvous",
    "peg_insertion_side",
    "place_bowl_on_plate",
    "wipe_table",
}
DEV_SEEDS = range(0, 8)
VALIDATION_SEEDS = range(8, 16)

DEV_TASK_GROUPS = {
    group: {task: load for task, load in tasks.items() if task not in VALIDATION_ONLY_TASKS}
    for group, tasks in TASK_GROUPS.items()
}
