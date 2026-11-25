"""Explicit CLI task names and their lazy adapter loaders."""

from functools import partial
from importlib import import_module


def _open(module_name: str, opener: str, *args):
    return getattr(import_module(module_name), opener)(*args)


def _open_maniskill(task: str):
    adapter = import_module("worldsmith.benchmark.adapters.maniskill_adapter")
    return adapter.open_maniskill(*adapter.TASKS[task])


def _controller(module_name: str, name: str, **kwargs):
    factory = getattr(import_module(module_name), name)
    return partial(factory, **kwargs) if kwargs else factory


_PYMUNK = "worldsmith.benchmark.adapters.pymunk_adapter"
_MANISKILL_TASKS = {
    task: partial(_open_maniskill, task) for task in (
        "push_cube", "pick_cube", "stack_cube", "peg_insertion_side",
        "pull_cube", "push_t", "roll_ball", "quadruped_reach",
    )
}
_PYMUNK_TASKS = {
    task: partial(_open, _PYMUNK, "open_pymunk", task) for task in (
        "interception", "navigation", "rendezvous", "bouncing_interception", "object_pushing",
    )
}
_LIBERO_TASKS = {
    task: partial(_open, "worldsmith.benchmark.adapters.libero_adapter", "open_libero", task)
    for task in ("open_middle_drawer", "place_bowl_on_plate", "place_two_items_in_basket")
}
_ROBOSUITE_TASKS = {
    task: partial(_open, "worldsmith.benchmark.adapters.robosuite_adapter", "open_robosuite", task)
    for task in ("open_latched_door", "assemble_nuts", "handover_hammer", "wipe_table")
}

# Defaults are loaded only when explicitly selected, never to open a simulator.
DEFAULT_CONTROLLERS = {
    "interception": partial(_controller, "worldsmith.benchmark.tasks.interception", "Chase"),
    **{task: partial(_controller, "autoresearch.controller", "make_controller") for task in (
        "navigation", "rendezvous", "bouncing_interception", "object_pushing",
    )},
    **{task: partial(_controller, f"worldsmith.benchmark.tasks.{task}", name) for task, name in (
        ("push_cube", "PushController"), ("pick_cube", "PickController"),
        ("stack_cube", "StackController"), ("peg_insertion_side", "PegPickupController"),
        ("pull_cube", "PullController"), ("push_t", "PushTController"),
        ("roll_ball", "idle_controller"), ("quadruped_reach", "idle_controller"),
    )},
    **{task: partial(_controller, "worldsmith.benchmark.adapters.libero_adapter", "idle_controller")
       for task in _LIBERO_TASKS},
    **{task: partial(_controller, "worldsmith.benchmark.adapters.robosuite_adapter", "idle_controller", arms=arms)
       for task, arms in (("open_latched_door", 1), ("assemble_nuts", 1), ("handover_hammer", 2))},
    "wipe_table": partial(_controller, "worldsmith.benchmark.adapters.robosuite_adapter", "idle_wiper"),
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
