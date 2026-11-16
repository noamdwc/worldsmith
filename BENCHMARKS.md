# Worldsmith v2 benchmark inventory

Worldsmith v2 has 20 tasks across four CPU simulator families. The phase column describes a typical solution, including cases where native scoring checks only the final state.

| Task | Simulator | Embodiment and control | Observation | Physical interaction and phases |
| --- | --- | --- | --- | --- |
| `interception` | Pymunk | Planar kinematic robot, heading and throttle | Noisy XY and covariance | Single capture of a target moving at constant velocity |
| `navigation` | Pymunk | Planar kinematic robot, heading and throttle | Noisy XY and covariance | Single goal with two moving obstacles and collision avoidance |
| `rendezvous` | Pymunk | Planar kinematic robot, heading and throttle | Noisy XY and covariance | Approach and then track a moving target for consecutive steps |
| `bouncing_interception` | Pymunk | Planar kinematic robot, heading and throttle | Noisy target XY, covariance, wall locations | Single capture of a dynamic target whose velocity changes on wall contact |
| `object_pushing` | Pymunk | Planar kinematic pusher, heading and throttle | Noisy XY and covariance | Reposition, push a dynamic disc with variable drag, and let it settle at the goal |
| `push_cube` | ManiSkill | Panda with fingers, base XYZ delta and gripper | Joint/TCP, cube pose and goal | Single tabletop sweep with sliding contact |
| `pick_cube` | ManiSkill | Panda with fingers, base XYZ delta and gripper | Joint/TCP, cube pose, goal and grasp reading | Grasp and carry a cube to a 3D goal |
| `stack_cube` | ManiSkill | Panda with fingers, base XYZ delta and gripper | Joint/TCP and two cube poses | Grasp, align, stack, and release |
| `peg_insertion_side` | ManiSkill | Panda with fingers, base pose delta and gripper | Joint/TCP, peg and hole geometry | Grasp and orient a peg, then insert with tight clearance |
| `pull_cube` | ManiSkill | Panda with fingers, base XYZ delta and gripper | Joint/TCP, cube pose and goal | Reach over the cube, then sweep it back |
| `push_t` | ManiSkill | Panda with rigid stick, base XYZ delta | Joint/TCP, T pose and goal center | Planar pushing with sliding and rotation to align a shape |
| `roll_ball` | ManiSkill | Panda with fingers, base XYZ delta and gripper | Joint/TCP, ball pose and velocity, goal | Approach, strike, and allow the ball to roll freely to a distant goal |
| `quadruped_reach` | ManiSkill | Floating-base Unitree Go2, 12 leg joint deltas | Leg proprioception, base velocities and goal offset | Coordinated gait with repeated foot-ground contact and fall termination |
| `open_middle_drawer` | LIBERO-PRO | Panda with fingers, world pose delta and gripper | Joint/tool and structured scene poses, drawer slide | Reach and pull a constrained sliding joint |
| `place_bowl_on_plate` | LIBERO-PRO | Panda with fingers, world pose delta and gripper | Joint/tool and structured scene poses | Identify the target bowl, grasp, and place on the plate |
| `place_two_items_in_basket` | LIBERO-PRO | Panda with fingers, world pose delta and gripper | Joint/tool and structured scene poses | Two independent pick-and-place stages into the same basket |
| `open_latched_door` | Robosuite | Panda with fingers, world pose delta and gripper | Joint/tool, hinge and handle measurements | Grasp or press the latch, then swing the hinged door |
| `assemble_nuts` | Robosuite | Panda with fingers, world pose delta and gripper | Joint/tool, nut poses, handles and peg locations | Precision alignment, seating and release for two matching nut-and-peg pairs |
| `handover_hammer` | Robosuite | Two Pandas with fingers, a world pose delta and gripper command per arm | Both arms' joint/tool measurements, hammer pose and handle | Pickup, coordinated contact transfer, and release |
| `wipe_table` | Robosuite | Panda with passive wiper, world pose delta | Joint/tool, binary contact and remaining-dirt summary | Make and sustain surface contact while clearing distributed targets |

The shared runner creates a fresh controller per episode and reports native success for the 3D tasks. AutoResearch compares controllers on the 16 shared tasks with eight fixed development seeds per task and averages task success rates with equal weight. Owner validation includes all 20 tasks on seeds 8–15. Use the same suite ID and split when comparing runs. The development guides cover shared task field names, action limits, and horizons. [Development tasks](autoresearch/TASKS.md) · [Controller API](autoresearch/API.md)

The 3D tasks use structured state observations with no cameras. Most still use fixed-base arms. Controllers are Python code evaluated in the simulator process. The observation and feedback interfaces filter private diagnostics, and process isolation remains outside their scope.

## Owner reference for validation-only tasks

### Pymunk `rendezvous`

Approach `target` and remain within 0.6 units for 20 consecutive control steps (2 seconds) in a 100-step episode. Leaving the radius resets the consecutive-step count. The count starts with the first control step. Targets start 3–5 units away and move at 0.8–1.7 units per second in a sampled direction. The robot is faster than the target. Estimating target velocity can help the robot match speed and avoid a persistent tracking lag. Metrics include success rate, mean longest tracking duration, mean distance over control steps, and mean final distance.

### ManiSkill `peg_insertion_side`

Insert the peg within 100 steps. In the hole frame, the peg head must reach X ≥ −0.015 m, with Y and Z within the hole radius. Peg and hole geometry varies by episode. `extra` includes `peg_pose`, `peg_half_size`, `box_hole_pose`, and `box_hole_radius` alongside the tool pose. Return a seven-entry `float32` action with XYZ translation, three base-frame rotation commands, and a gripper command. The rotation vector is clipped to unit length and multiplied by −0.1 to obtain XYZ Euler increments in radians.

### LIBERO-PRO `place_bowl_on_plate`

Place the designated black bowl between the plate and the ramekin onto the plate within 220 steps. The observation includes `akita_black_bowl_1_pos`, `akita_black_bowl_2_pos`, `plate_1_pos`, and `glazed_rim_porcelain_ramekin_1_pos`. It uses the same world-frame pose and gripper commands as the other LIBERO-PRO tasks.

### Robosuite `wipe_table`

Clear all dirt marks with the passive wiping tool within 300 steps. Arm collisions and joint limits can end the episode early. The observation includes joint and tool readings, binary `robot0_contact`, `wipe_centroid` in world XYZ, `wipe_radius` to the furthest remaining mark from the centroid, and `proportion_wiped`. Return six entries using the standard Robosuite translation and rotation conventions for the wiping tool.
