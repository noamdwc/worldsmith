# Benchmark logic validation and sanity testing

Validation used the PR 18 merge (`c5569a7`) and the repository's pinned CPU environments on Apple M2 Pro. The replay tests exercise two seeds per task in both orders on a reused adapter and compare serial and two-worker summaries and complete episode traces. The original ManiSkill regression still runs a successful grasp between two copies of the same episode.

## Test execution

Across the validation runs, 163 tests passed, including 31 new checks. All 17 tasks passed the planned behavioral checks. The added hammer-handover regression uses a seed-3 controller driven by public observations and actions.

| Environment | Command | Passed |
| --- | --- | ---: |
| Default | `.venv/bin/python -m pytest -q` | 115 |
| ManiSkill | `maniskill/.venv/bin/python -m pytest -q tests/maniskill_push_cube.py tests/maniskill_manipulation.py` | 27 |
| LIBERO-PRO | `libero/.venv/bin/python -m pytest -q tests/libero_pro.py` | 11 |
| Robosuite | `robosuite/.venv/bin/python -m pytest -q tests/robosuite_native.py` | 10 |

ManiSkill used the Vulkan configuration in its README and required macOS graphics access. The pinned ManiSkill and LIBERO dependencies emitted deprecation warnings.

The final verification reran the default and Robosuite suites and four targeted LIBERO drawer and bowl checks.

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

| Family | Task | Task-specific checks and limitations | Validation | Starter | Candidate | Final | Improvement | Loop |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Pymunk | `interception` | Capture radius, seeded sensor noise, malformed and NumPy scalar actions | Passed | 6/8 (75%) | 6/8 (75%) | 6/8 (75%) | 0 pp | Completed |
| Pymunk | `navigation` | Destination distance, obstacle contact, exact public trace and outcomes | Passed | 5/8 (62.5%) | 8/8 (100%) | 8/8 (100%) | +37.5 pp | Completed |
| Pymunk | `rendezvous` | Consecutive tracking time and timeout | Passed | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) | +25 pp | Completed |
| Pymunk | `bouncing_interception` | Wall rebounds, capture boundary and timeout | Passed | 2/8 (25%) | 8/8 (100%) | 8/8 (100%) | +75 pp | Completed |
| Pymunk | `object_pushing` | Contact, coasting and consecutive slow steps at goal | Passed | 4/8 (50%) | 8/8 (100%) | 8/8 (100%) | +50 pp | Completed |
| ManiSkill | `push_cube` | Native cube pose and successful reference pushes | Passed | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 0 pp | Completed |
| ManiSkill | `pick_cube` | Grasp and pose reset, successful native grasp and placement | Passed | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 0 pp | Completed |
| ManiSkill | `stack_cube` | Native cube pose and a placed, released stack | Passed | 6/8 (75%) | 6/8 (75%) | 6/8 (75%) | 0 pp | Completed |
| ManiSkill | `peg_insertion_side` | Native peg pose, clipped base-frame rotation, inserted peg | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Completed |
| ManiSkill | `pull_cube` | Native cube pose and placement in the goal | Passed | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 0 pp | Completed |
| ManiSkill | `push_t` | Three-channel stick command, native T pose and aligned goal | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Completed |
| LIBERO-PRO | `open_middle_drawer` | World and tool-frame poses, native open-drawer success | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Rerun completed |
| LIBERO-PRO | `place_bowl_on_plate` | Bowl pose, tool-frame offset and positive contact on plate | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Rerun completed |
| LIBERO-PRO | `place_two_items_in_basket` | Basket pose and both required items inside the native region | Passed | 0/8 (0%) | 8/8 (100%) | 8/8 (100%) | +100 pp | Completed |
| Robosuite | `open_latched_door` | Native handle and hinge state, real controller opens door | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Rerun completed |
| Robosuite | `assemble_nuts` | Peg locations, half reward after one nut, success after two | Passed | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) | 0 pp | Completed |
| Robosuite | `handover_hammer` | Both arm action slices and native contact transfer from pickup to receiver, with successful termination at step 176 on seed 3 | Passed | 0/8 (0%) | 1/8 (12.5%) | 1/8 (12.5%) | +12.5 pp | Completed |

## AutoResearch execution

Part A finished before the sanity runs started. Each of the 17 sessions received its task's verified starter summary and public development traces, along with the benchmark guides and starter controller. Each session had its own workspace and MCP server. The budget was one candidate per task, with strict improvement required for retention. Starter, candidate, and owner re-evaluation used seeds 0–7.

In the initial round, six of the 14 scored candidates improved and were retained. The other eight tied. The owner re-evaluated the selected snapshot for every task and reproduced its score and complete public trace. The handover candidate's successful rollout on development seed 3 was used to build the new regression test. The test reads native contacts and success for its assertions while the controller acts solely on public observations.

For 10 sessions, the owner archived leftover backup files and restored the exact selected code snapshot before the final verification. The earlier outputs and researcher journals remain in their task directories.

- `open_middle_drawer` candidate setup failed at seed 0 with `ModuleNotFoundError: No module named 'autoresearch.drawer'`. The starter was retained.
- `place_bowl_on_plate` candidate setup failed at seed 0 with `ModuleNotFoundError: No module named 'autoresearch.bowl'`. The starter was retained.
- `open_latched_door` candidate setup failed at seed 0 with `ModuleNotFoundError: No module named 'autoresearch.door'`. The starter was retained.

Those sessions placed helper modules one directory below the package root. The workspace guide now shows where to place importable helpers and where to keep temporary backups.

Each of those three tasks was rerun in a fresh workspace with the corrected guide, one candidate attempt, and the same development seeds 0–7. All three candidates evaluated successfully and tied their starters at 0/8. The owner re-evaluated the exact starter snapshots and reproduced all three scores and public traces. Across the latest scored attempt for each task, six candidates improved and eleven tied.

The original local baselines are under `runs/pr19-baselines/<task>/baseline.json`. The session records, candidate snapshots, journals and final results live under `runs/pr19-sanity/<task>/` for the initial round and `runs/pr19-sanity-rerun/<task>/` for the reruns. Each run directory has an `audit.json` index at its root. These paired development scores measure the sanity workflow on eight fixed seeds.

## Follow-up

Enforce process isolation if the researcher or generated Python must be treated as adversarial. Keep sanity scores separate from benchmark validity and reserve held-out seeds for a separate evaluation.

## PR20 task expansion

PR20 adds three native tasks to the [benchmark inventory](BENCHMARKS.md). Validation used the pinned CPU runtimes on Apple M2 Pro. The new tasks replayed identically in both seed orders and produced byte-identical CLI summaries and traces with one and two workers.

| Environment | Command | Passed |
| --- | --- | ---: |
| Default | `.venv/bin/python -m pytest -q` | 116 |
| ManiSkill | `maniskill/.venv/bin/python -m pytest -q tests/maniskill_push_cube.py tests/maniskill_manipulation.py tests/maniskill_diversity.py` | 36 |
| LIBERO-PRO | `libero/.venv/bin/python -m pytest -q tests/libero_pro.py` | 11 |
| Robosuite | `robosuite/.venv/bin/python -m pytest -q tests/robosuite_native.py` | 15 |

- `roll_ball` checks base-frame action mapping, native ball and goal observations, and continued motion after contact sets the ball rolling.
- `quadruped_reach` checks joint channel ordering, target offsets and actual leg motion, native proprioception and goal offsets, and termination after body-ground contact.
- `wipe_table` checks world-frame tool motion, native contact and marker removal, coverage sampling, reset after wiping, and early failure at a joint limit.

Each new task checks native success from a controlled state, timeout at the configured horizon, and 50% evaluator aggregation across one success and one timeout. The controlled success checks exercise the adapter and native predicates independently of the controller's ability to solve the whole task.

The full `worldsmith-all-v4` starter evaluation finished in 196.4 seconds on the M2 Pro and scored 53/160 (33.125%). All 17 existing tasks reproduced their PR19 starter metrics and complete public traces on the same development seeds. The full starter record is under `runs/pr20-full-starter/`.

### PR20 AutoResearch sanity checks

Each task had an independent workspace and development-only evaluation server. The baseline and each candidate evaluation used seeds 0–7. The fixed budget was three evaluation attempts per task, with strict improvement required for retention.

| Task | Baseline | Attempt 1 | Attempt 2 | Attempt 3 | Best retained | Retained run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `roll_ball` | 0/8 | 0/8 (unchanged source) | 2/8 | 3/8 | 3/8 (37.5%) | `mcp-c29b674faea54abcb1c9f05aaa2a813b` |
| `quadruped_reach` | 0/8 | 0/8 | 4/8 | 0/8 | 4/8 (50%) | `mcp-21b2fdf025ab4028b160d2872e48380c` |
| `wipe_table` | 0/8 | 0/8 (unchanged source) | 0/8 | 0/8 | 0/8 (0%) | `mcp-630a6af9a13d48bd9d69450f2fb28196` |

RollBall retained a staged strike with a closer approach waypoint. Quadruped Reach retained a higher-gain diagonal trot. Wipe's final candidate made sustained contact and cleared 1–35% of the markers across the eight layouts. It tied the baseline on native task success, so the baseline was retained.

The first launch could not call the local evaluator with approvals disabled. With review enabled, all three baselines completed. Review initially blocked a public quadruped trace request after mistaking a score of 0% for an evaluation error. The experiment resumed from the saved successful baseline. A Codex usage limit interrupted RollBall and Wipe, and both resumed from their journals. Each had already spent one attempt re-evaluating unchanged source after a shell edit failed because `python` was unavailable. The remaining attempts used verified source changes. All 12 evaluation batches scored successfully once submitted to the evaluator. The interrupted sessions resumed within the original budget.

An owner replay of each retained snapshot reproduced its summary and full native trace byte for byte. Local workspaces, session logs, and public artifacts are under `runs/pr20-sanity/approved/<task>/`. The retained scores are measurements on the development seeds.
