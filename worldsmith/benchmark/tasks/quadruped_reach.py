"""Native Unitree Go2 locomotion on flat ground."""

from functools import partial

import numpy as np

from worldsmith.benchmark.maniskill_adapter import open_maniskill


def idle_controller():
    return lambda observation: np.zeros(12, dtype=np.float32)


def evaluation_setup():
    return partial(
        open_maniskill,
        "UnitreeGo2-Reach-v1",
        "unitree_go2_simplified_locomotion",
        "pd_joint_delta_pos",
    ), idle_controller
