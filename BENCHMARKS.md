# Worldsmith v2 benchmark inventory

Worldsmith v2 has 20 tasks across four CPU simulator families. The rows identify physical and control differences that can inform task-level splits in PR21. The phase column describes a typical solution, including cases where native scoring checks only the final state.

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

The shared runner creates a fresh controller per episode and reports native success for the 3D tasks. AutoResearch compares controllers on eight fixed development seeds per task and averages the task success rates with equal weight. Keep owner held-out seeds private and use the same suite ID and split when comparing runs. See the [public task guide](autoresearch/TASKS.md) and [controller API](autoresearch/API.md) for field names, action limits, and horizons.

The 3D tasks use structured state observations with no cameras. Most still use fixed-base arms. Controllers are Python code evaluated in the simulator process. The observation and feedback interfaces filter private diagnostics, and process isolation remains outside their scope. The suite measures simulation performance and currently separates development and evaluation by seed within each task. PR21 will introduce task-level separation before the v3 online-agent work.
