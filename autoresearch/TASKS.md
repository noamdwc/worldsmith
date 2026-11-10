# Public task brief

## Moving-target interception

Control the robot identified by `observation.robot_id`. Get within 0.3 simulation units of the entity named `target` within 45 steps. The target moves at constant velocity in an open 2D plane. Each step is 0.1 simulated seconds.

The [World API](API.md) gives you noisy entity positions and their 2×2 measurement covariances, the observation time, and the robot's maximum speed. Target velocity is unobserved.

Capture rate is the primary metric. Mean closest distance and per-step distance in development traces help diagnose near misses.
