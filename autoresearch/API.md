# Public World API

Import `Action`, `Observation`, and `Controller` from `worldsmith.api`. See [controller.py](controller.py) for a working example.

## Observation

| Field | Type | Meaning |
|---|---|---|
| `time` | `float` | Simulated seconds since the episode started |
| `robot_id` | `str` | Entity ID of the robot you control |
| `entities` | `dict[str, PositionMeasurement]` | Noisy position measurements keyed by entity ID |
| `robot_max_speed` | `float` | Maximum commanded speed in simulation units per second |
| `goal` | `str` | Task name for the episode. V1 supplies `"interception"`, `"navigation"`, or `"rendezvous"` |

The simulator supplies the same goal in the initial observation and every subsequent observation. Existing four-argument calls to `Observation(...)` default to `"interception"`.

Each `PositionMeasurement` has a `position` tuple `(x, y)` in simulation units and a `covariance` tuple `((xx, xy), (yx, yy))` describing position measurement noise. The diagonal entries are variances in squared simulation units. Treat measurements as read-only. Entity positions are measurements, and velocity is unobserved.

## Action

Return `Action(heading, throttle=1.0)`. `heading` is a finite angle in radians measured counterclockwise from the positive x axis. `throttle` is a fraction of `robot_max_speed` between `0` (stop) and `1` (full speed).

## Controller

Define one `make_controller() -> Controller` in `controller.py` for all tasks. The evaluator calls this zero-argument factory once per episode. The returned Python callable receives the goal through `observation.goal` each step and returns one `Action`. V1 uses no runtime LLM. Store episode history on the returned object or in its closure so each episode starts with fresh memory.
