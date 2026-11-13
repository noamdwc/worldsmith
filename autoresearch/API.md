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

## Development feedback

`evaluate_controller()` returns the benchmark ID, snapshot hash, development seeds and episode counts, success rates and changes from the retained controller, task metrics when available, and elapsed time.

`get_episode_trace(seed, task)` returns the episode outcome and the observations and actions in order. Each frame pairs the observation available when the controller acted with its action. `initial_observation` is the first input and `final_observation` is the state observed after the last action. Tensor values appear as lists. The trace reports whether the episode succeeded and whether it terminated or reached its step limit.
