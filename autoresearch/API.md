# Public World API

Define `make_controller(*, task=None)` in [controller.py](controller.py). AutoResearch passes the task ID when it creates a fresh controller for each episode. Return a callable that takes the current observation and returns one action. Keep episode memory in that callable or its closure and treat observations as read-only.

## Planar movement

Import `Action` and `Observation` from `worldsmith.api`. See [TASKS.md](TASKS.md) for each task’s entities and limits.

| Observation field | Type | Meaning |
| --- | --- | --- |
| `time` | `float` | Simulated seconds since the start of the episode |
| `robot_id` | `str` | ID of the robot you control |
| `entities` | `dict[str, PositionMeasurement]` | Position measurements keyed by entity ID |
| `robot_max_speed` | `float` | Maximum commanded speed in world units per second |
| `goal` | `str` | Task ID, constant throughout the episode |

Each `PositionMeasurement` has a position tuple `(x, y)` in world units and a covariance tuple `((xx, xy), (yx, yy))`. The diagonal entries are variances in squared world units.

Return `Action(heading, throttle=1.0)`. Heading is a finite angle in radians counterclockwise from the positive x axis. Throttle is a fraction of `robot_max_speed` between `0` (stop) and `1` (full speed). When the factory is called without a task ID, use `observation.goal` for planar tasks.

## Tabletop manipulation

Observations are nested dictionaries of tensors with a leading batch dimension of one. The `agent` entry provides joint state, including `qpos` and `qvel`. The `extra` entry contains task measurements listed in [TASKS.md](TASKS.md). For example, `observation["extra"]["tcp_pose"][0, :3]` gives the tool center point position. Pose vectors use `[x, y, z, qw, qx, qy, qz]`, with position in meters.

Return a NumPy `float32` array with entries in `[-1, 1]`. The first three entries command XYZ displacement in the robot base frame, with 0.1 m per axis at full scale. For the peg task, the next three entries form an XYZ rotation command in the robot base frame. The command vector is clipped to unit length, then multiplied by −0.1 to obtain XYZ Euler increments in radians. On tasks with a gripper, the last entry commands it, with `-1` closing and `+1` opening. PushT uses a stick and takes only XYZ commands.

## Arm manipulation with varied starting layouts

Observations are flat dictionaries of NumPy arrays without a batch dimension. Positions are world XYZ in meters and quaternions use XYZW order. `robot0_eef_pos` and `robot0_eef_quat` describe the tool pose. `robot0_joint_pos`, `robot0_joint_vel`, `robot0_gripper_qpos`, and `robot0_gripper_qvel` provide joint readings. Movable objects have `<name>_pos` and `<name>_quat` fields in world coordinates, and `<name>_to_robot0_eef_pos` and `<name>_to_robot0_eef_quat` fields in the tool frame. The drawer task also gives a world-space handle position and a scalar slide displacement in meters. See [TASKS.md](TASKS.md) for object names.

Return a floating-point NumPy array of shape `(7,)` with values in `[-1, 1]`. The first three entries command world XYZ displacement, scaled by 0.05 m per axis. The next three entries form a world-frame axis-angle rotation vector, scaled by 0.5 radians per component. The last entry commands the gripper. Negative opens, positive closes, and zero holds the current gripper command.

## Door opening, nut assembly, and hammer handover

Observations are flat dictionaries without a batch dimension. World XYZ positions are in meters and quaternions are in XYZW order. For each arm, `robot0_eef_pos` and `robot0_eef_quat_site` give the tool position and orientation. `robot0_joint_pos` and `robot0_joint_vel` give joint positions and velocities. `robot0_gripper_qpos` and `robot0_gripper_qvel` give finger positions and velocities. The receiving arm in handover uses the same fields prefixed with `robot1_`. See [TASKS.md](TASKS.md) for scene measurements.

Return a floating-point NumPy array with finite entries in `[-1, 1]`. Each arm uses seven consecutive entries `[dx, dy, dz, rx, ry, rz, grip]`. Translation sets a target offset from the current tool position, scaled by 0.05 m per component. Rotation sets a world-frame axis-angle offset, scaled by 0.5 radians per component. Negative gripper commands open the fingers, positive commands close them, and zero holds the current gripper command. Use shape `(7,)` for one arm and `(14,)` for handover, with the pickup arm first and receiving arm second. Commands update at 20 Hz.

## Development feedback

`evaluate_controller()` returns the benchmark ID, snapshot hash, development seeds and episode counts, success rates and changes from the retained controller, task metrics when available, and elapsed time.

`get_episode_trace(seed, task)` returns the episode outcome and the observations and actions in order. Each frame pairs the observation available when the controller acted with its action. `initial_observation` is the first input and `final_observation` is the state observed after the last action. Tensor values appear as lists. The trace reports whether the episode succeeded and whether it terminated or reached its step limit.
