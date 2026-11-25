"""ManiSkill PullCube with a closed-gripper sweep toward the robot."""

from worldsmith.benchmark.tasks.push_cube import PushController


class PullController(PushController):
    # The goal lies on the cube's -X side, so lower the gripper on +X.
    approach_offset_x = 0.055
