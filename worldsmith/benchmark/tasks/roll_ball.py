"""Native Panda task for rolling a ball into a goal region."""

import numpy as np



def idle_controller():
    return lambda observation: np.array([0., 0., 0., 1.], dtype=np.float32)
