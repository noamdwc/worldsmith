"""Native Unitree Go2 locomotion on flat ground."""

import numpy as np



def idle_controller():
    return lambda observation: np.zeros(12, dtype=np.float32)
