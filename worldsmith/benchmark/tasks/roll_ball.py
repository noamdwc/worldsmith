"""Native Panda task for rolling a ball into a goal region."""

from functools import partial

import numpy as np

from worldsmith.benchmark.adapters.maniskill_adapter import open_maniskill


def idle_controller():
    return lambda observation: np.array([0., 0., 0., 1.], dtype=np.float32)


def evaluation_setup():
    return partial(open_maniskill, "RollBall-v1", "panda", "pd_ee_delta_pos"), idle_controller
