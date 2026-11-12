# Worldsmith development tasks

One shared `make_controller(task=...)` handles all 11 tasks. The researcher can read these task descriptions and all development traces. The benchmark uses no runtime LLM and does not require task blindness.

## Pymunk

These five tasks use 2D scenes and 0.1-second steps. The [World API](API.md) supplies noisy positions and measurement covariances, observation time, robot ID, maximum speed (2.4 units/second), and the goal identifier. Entity velocity is unobserved. The first three tasks use constant-velocity targets or obstacles.

## Moving-target interception

`observation.goal == "interception"`. Get within 0.3 units of `target` within 45 steps. Touching the capture region once is enough; matching the target's motion is unnecessary.

Metrics: success rate (also called capture rate), captures, and mean closest distance.

## Moving-obstacle navigation

`observation.goal == "navigation"`. Reach the stationary `destination` within 0.35 units in 80 steps while avoiding `obstacle_0` and `obstacle_1`, which cross the route in opposite directions. Robot and obstacle radii are 0.15 and 0.55 units. A center distance of 0.7 units or less counts as a collision and makes the episode unsuccessful, even if the robot later reaches the destination.

Contact is scored at the initial state and each simulation step; bodies do not bounce or block each other. There are no walls. Predicting crossing traffic, choosing a route, and waiting can help.

Destinations are 8–10 units ahead with a small lateral offset. Obstacle crossing speeds vary from 0.9–1.7 and 0.8–1.6 units/second, with independently sampled starting positions. Turning away only when an obstacle is nearby can be too late for faster crossings.

Metrics: success rate, collision rate, mean final destination distance, and mean minimum obstacle clearance (center distance minus 0.7).

## Rendezvous / tracking

`observation.goal == "rendezvous"`. Approach `target` and remain within 0.6 units for 20 consecutive control steps (2 seconds) within a 100-step episode. Leaving the radius resets the consecutive-step count. The initial observation does not count as a control step. Success requires sustained proximity, so controlling speed and matching motion matter.

Targets start 3–5 units away and move at 0.8–1.7 units/second in a sampled direction. The robot is always faster, with enough time to catch even an outbound target and sustain proximity. Slowing down based only on current distance can leave a persistent tracking lag; estimating motion can help match speed while remaining close.

Metrics: success rate, mean longest tracking duration, mean distance over control steps, and mean final distance.

## Bouncing-object interception

`observation.goal == "bouncing_interception"`. Get within 0.3 units of `target` within 45 steps. The target starts 2–3.2 units off center horizontally with outward motion at 3.4–4.4 units/second. Static walls redirect and slow the target on impact. Arena dimensions, starting position, speed, heading, and wall restitution vary by episode. The robot commands heading and throttle as usual, and its motion does not interact with the walls or target.

The target position has 0.12 units of measurement noise (standard deviation per axis). The exact positions of `wall_left`, `wall_right`, `wall_bottom`, and `wall_top` mark the four wall midpoints. The arena is axis-aligned, with each wall extending to the adjacent walls. The target radius is 0.35 units and the wall segments have a 0.25-unit radius. Restitution and target velocity are unobserved. Cutting across the target's return path can help when it is moving faster than the robot.

Metrics: success rate (also called capture rate), captures, and mean closest distance.

## Object pushing

`observation.goal == "object_pushing"`. Move `object` within 0.25 units of the stationary `destination` at a speed of at most 0.15 units/second for 8 consecutive control steps (0.8 seconds). The episode allows 140 steps (14 seconds). Leaving the radius or exceeding the speed limit resets the consecutive-step count.

The object starts at rest, 3–5 units from the destination. The robot starts on the destination side of the object and can push it by contact between their circles (radii 0.22 and 0.34 units, respectively). The plane is open. The object retains momentum after release and slows under linear drag. Mass, contact friction, and drag vary across episodes.

Position noise has a standard deviation per axis of 0.02 units for the robot and 0.04 units for the object. The destination is exact. Velocity, contact impulse, and physical parameters are unobserved. Repositioning around the object, slowing before release, and correcting after a push can help it settle.

Metrics include success rate, mean final and minimum object-to-destination distances, mean final object speed, and mean time spent transferring collision impulse.

## ManiSkill

The six manipulation tasks use native ManiSkill success flags and the structured `state_dict` contract in [API.md](API.md). Episodes stop on native termination or truncation. Success rate is the published metric for each task.

| Task | Native objective | Task observations in `extra` | Mode | Action size | Step limit |
| --- | --- | --- | --- | ---: | ---: |
| `push_cube` | Push the cube to the goal | TCP pose, object pose, goal position | `pd_ee_delta_pos` | 4 | 50 |
| `pick_cube` | Pick up the cube and bring it to the goal | TCP pose, object pose, goal position, grasp flag | `pd_ee_delta_pos` | 4 | 50 |
| `stack_cube` | Stack the first cube on the second and release it | TCP pose and both cube poses | `pd_ee_delta_pos` | 4 | 50 |
| `peg_insertion_side` | Insert the peg head into the hole from the side | TCP pose, peg pose, hole pose, peg dimensions | `pd_ee_delta_pose` | 7 | 100 |
| `pull_cube` | Pull the cube toward its goal on the robot side | TCP pose, object pose, goal position | `pd_ee_delta_pos` | 4 | 50 |
| `push_t` | Push the T into alignment with the goal outline | TCP pose, object pose, goal position | `pd_ee_delta_pos` | 3 | 100 |

PushT scores overlap, so position and orientation both matter. PegInsertionSide samples peg and hole geometry at reset and evaluates the peg head in the hole's frame.

## Evaluation and splits

`evaluate_controller()` uses 8 fixed development seeds per task (0–7). The full suite, `worldsmith-all-v1`, has 88 episodes. The ManiSkill-only suite, `worldsmith-maniskill-v1`, has 48. Each episode starts with a fresh controller from the same snapshot. The aggregate is the equal-weight mean of the evaluated task success rates. Secondary metrics support diagnosis. Compare results within the same benchmark and split.

`get_episode_trace(seed, task)` accepts task IDs in the active suite and returns public feedback from the latest successful development run. The owner maintains held-out seeds separately. Researchers use development feedback for every candidate decision.

### Starter development scores

The starter succeeded in 53 of 88 episodes for a `worldsmith-all-v1` score of 60.23%.

| Task | Successes / 8 |
| --- | ---: |
| `interception` | 6 |
| `navigation` | 5 |
| `rendezvous` | 6 |
| `bouncing_interception` | 2 |
| `object_pushing` | 4 |
| `push_cube` | 8 |
| `pick_cube` | 8 |
| `stack_cube` | 6 |
| `peg_insertion_side` | 0 |
| `pull_cube` | 8 |
| `push_t` | 0 |
