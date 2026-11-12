# Public World API

Import `Action`, `Observation`, and `Controller` from `worldsmith.api`. See [controller.py](controller.py) for a working example.

## Pymunk observation

| Field | Type | Meaning |
|---|---|---|
| `time` | `float` | Simulated seconds since the episode started |
| `robot_id` | `str` | Entity ID of the robot you control |
| `entities` | `dict[str, PositionMeasurement]` | Noisy position measurements keyed by entity ID |
| `robot_max_speed` | `float` | Maximum commanded speed in simulation units per second |
| `goal` | `str` | Machine-readable task identifier for the controller. One of `"interception"`, `"navigation"`, `"rendezvous"`, `"bouncing_interception"`, or `"object_pushing"` |

The simulator supplies the same goal in the initial observation and every subsequent observation. Existing four-argument calls to `Observation(...)` default to `"interception"`.

Evaluation summaries use the `objective` key for the human-readable `task.objective`. Combined summaries map task identifiers to their objective descriptions under that key.

Each `PositionMeasurement` has a `position` tuple `(x, y)` in simulation units and a `covariance` tuple `((xx, xy), (yx, yy))` describing position measurement noise. The diagonal entries are variances in squared simulation units. Treat measurements as read-only. Entity positions are measurements, and velocity is unobserved. In bouncing interception, the four `wall_*` entities give exact wall midpoint positions with zero covariance. Object pushing provides noisy positions for `robot` and `object` and an exact position for `destination`. See [TASKS.md](TASKS.md) for task-specific entities and scoring.

## Pymunk action

Return `Action(heading, throttle=1.0)`. `heading` is a finite angle in radians measured counterclockwise from the positive x axis. `throttle` is a fraction of `robot_max_speed` between `0` (stop) and `1` (full speed).

## ManiSkill observation and action

The controller receives the original `state_dict`, a nested dictionary of CPU tensors with a leading batch dimension of one. `agent` provides robot joint state. `extra` provides the task observations in [TASKS.md](TASKS.md). Treat observations as read-only. For example, `observation["extra"]["tcp_pose"][0, :3]` is the TCP position.

Return a NumPy `float32` array of the documented action size with values in `[-1, 1]`. The first three values command XYZ displacements in the robot base frame, with 0.1 m per axis at full scale. In pose mode, the next three values command rotation. On gripper-equipped robots, the last value commands the gripper, with `-1` closing it. PushT uses a stick and takes only XYZ commands.

## Shared controller

Define one `make_controller(*, task=None)` in `controller.py`. AutoResearch supplies the catalog task ID when it creates a fresh controller for each episode. The returned Python callable receives the native observation at each step and returns the corresponding native action. The optional default preserves the zero-argument Pymunk CLI contract through `observation.goal`.

Store episode history on the returned object or in its closure. The file and any common helpers must import under Python 3.11 and 3.12. Import runtime-specific dependencies inside the corresponding branch, as the starter does. The controller runs without a runtime LLM.

## Development feedback

The summary includes the snapshot hash, benchmark and task configuration, development seeds and episode counts, success rates, deltas from the retained controller, native metrics when available, and wall-clock runtime. Pymunk traces pair each action with its resulting observation, starting from `initial_observation`. ManiSkill traces pair each action with the observation that produced it and include `final_observation`, success, and native termination flags. The `state_dict` values are serialized as lists. Raw simulator `info`, reward, ground truth, and evaluator diagnostics stay in owner artifacts.
