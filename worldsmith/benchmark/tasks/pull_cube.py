"""ManiSkill PullCube with a closed-gripper sweep toward the robot."""

from functools import partial

from worldsmith.benchmark.maniskill_adapter import open_maniskill
from worldsmith.benchmark.tasks.push_cube import PushController


class PullController(PushController):
    # The goal lies on the cube's -X side, so lower the gripper on +X.
    approach_offset_x = 0.055


def evaluation_setup():
    return partial(open_maniskill, "PullCube-v1", "panda", "pd_ee_delta_pos"), PullController
