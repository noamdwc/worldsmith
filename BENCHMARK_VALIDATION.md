# Benchmark logic validation and sanity testing

Validation used the PR 18 merge (`c5569a7`) and the repository's pinned CPU environments on Apple M2 Pro. The replay tests exercise two seeds per task in both orders on a reused adapter and compare serial and two-worker summaries and complete episode traces. The original ManiSkill regression still runs a successful grasp between two copies of the same episode.

## Test execution

All 162 tests passed, including 30 new checks. Sixteen tasks passed the planned behavioral checks. Hammer handover passed the replay, action, observation and timeout checks. Its positive grasp path remains unverified.

| Environment | Command | Passed |
| --- | --- | ---: |
| Default | `.venv/bin/python -m pytest -q` | 115 |
| ManiSkill | `maniskill/.venv/bin/python -m pytest -q tests/maniskill_push_cube.py tests/maniskill_manipulation.py` | 27 |
| LIBERO-PRO | `libero/.venv/bin/python -m pytest -q tests/libero_pro.py` | 11 |
| Robosuite | `robosuite/.venv/bin/python -m pytest -q tests/robosuite_native.py` | 9 |

ManiSkill used the Vulkan configuration in its README and required macOS graphics access. The pinned ManiSkill and LIBERO dependencies emitted deprecation warnings.

## Changes

- Pymunk accepted NumPy scalar commands in the simulator but could abort when writing a trace containing a `float32` command. It now snapshots these commands into JSON-compatible numbers.
- Text in a Pymunk action's heading or throttle raised an evaluator exception. The simulator now rejects those values as invalid actions, so feedback identifies the task, seed, and step.
- Task-level development sessions use the same seeds, scoring, and feedback as the full suite. Each session can use a dedicated workspace. The held-out seed-file option remains owner-only.

## Shared checks

- Serial and parallel controller failures preserve only the completed episode prefix in owner artifacts and produce no summary. Existing tests also cover worker crashes, stalls, and cleanup.
- The researcher-facing MCP tests check task and seed attribution, score aggregation, terminal observations, retained scores after failures, and rejection of held-out feedback. Public traces contain controller inputs, actions and documented outcomes. Native rewards, reset info, simulator diagnostics and private errors are kept in owner artifacts.
- Researcher information boundaries were checked through the controller arguments and MCP feedback. The controller runs as Python in the evaluator process, so host filesystem and interpreter isolation remain outside that guarantee.
- On the pinned LIBERO stack, sensor samples can precede the last physics substep. The tests synchronize measurements explicitly for comparisons against native state, and use the native cached observations for replay and scoring.

## Task results

Every row includes the replay checks above. The validation column covers the behaviors tested. Pymunk tests compare heading, throttle and stop commands to native motion and exercise the task-specific criteria. Manipulation tests compare native joint, tool and object state, bounded actions, and native success and timeout behavior.

| Family | Task | Task-specific checks and limitations | Validation | Starter | Final | Improvement | Loop |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Pymunk | `interception` | Capture radius, seeded sensor noise, malformed and NumPy scalar actions | Passed | 6/8 (75%) | — | — | Blocked |
| Pymunk | `navigation` | Destination distance, obstacle contact, exact public trace and outcomes | Passed | 5/8 (62.5%) | — | — | Blocked |
| Pymunk | `rendezvous` | Consecutive tracking time and timeout | Passed | 6/8 (75%) | — | — | Blocked |
| Pymunk | `bouncing_interception` | Wall rebounds, capture boundary and timeout | Passed | 2/8 (25%) | — | — | Blocked |
| Pymunk | `object_pushing` | Contact, coasting and consecutive slow steps at goal | Passed | 4/8 (50%) | — | — | Blocked |
| ManiSkill | `push_cube` | Native cube pose and successful reference pushes | Passed | 8/8 (100%) | — | — | Blocked |
| ManiSkill | `pick_cube` | Grasp and pose reset, successful native grasp and placement | Passed | 8/8 (100%) | — | — | Blocked |
| ManiSkill | `stack_cube` | Native cube pose and a placed, released stack | Passed | 6/8 (75%) | — | — | Blocked |
| ManiSkill | `peg_insertion_side` | Native peg pose, clipped base-frame rotation, inserted peg | Passed | 0/8 (0%) | — | — | Blocked |
| ManiSkill | `pull_cube` | Native cube pose and placement in the goal | Passed | 8/8 (100%) | — | — | Blocked |
| ManiSkill | `push_t` | Three-channel stick command, native T pose and aligned goal | Passed | 0/8 (0%) | — | — | Blocked |
| LIBERO-PRO | `open_middle_drawer` | World and tool-frame poses, native open-drawer success | Passed | 0/8 (0%) | — | — | Blocked |
| LIBERO-PRO | `place_bowl_on_plate` | Bowl pose, tool-frame offset and positive contact on plate | Passed | 0/8 (0%) | — | — | Blocked |
| LIBERO-PRO | `place_two_items_in_basket` | Basket pose and both required items inside the native region | Passed | 0/8 (0%) | — | — | Blocked |
| Robosuite | `open_latched_door` | Native handle and hinge state, real controller opens door | Passed | 0/8 (0%) | — | — | Blocked |
| Robosuite | `assemble_nuts` | Peg locations, half reward after one nut, success after two | Passed | 0/8 (0%) | — | — | Blocked |
| Robosuite | `handover_hammer` | Both arm action slices and hammer state. Successful receiving grasp remains unverified | Not Verified | 0/8 (0%) | — | — | Blocked |

## AutoResearch execution

After the validation suites passed, all 17 local starter evaluations completed on seeds 0–7, each from a fresh workspace with the same starter snapshot. The results are recorded under `runs/pr19-baselines/<task>/baseline.json` and `runs/pr19-baselines/<task>/workspace/runs/`. Every starter score was checked against its eight public episode outcomes.

Automatic approval review rejected the signed-in Codex CLI launch because the experiment would send project material to the hosted model without explicit authorization for that destination. No candidate evaluation ran. Final success rates and improvement are unmeasured for all 17 tasks. The planned budget is one candidate per task, followed by a fresh evaluation of the retained controller on the same development seeds.

## Follow-up

Add a physically successful two-arm handover fixture and enforce process isolation if the researcher or generated Python must be treated as adversarial. Keep sanity scores separate from benchmark validity and reserve held-out seeds for a separate evaluation.
