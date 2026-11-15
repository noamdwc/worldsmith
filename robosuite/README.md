# Native Robosuite tasks on Apple Silicon

From the repository root, with arm64 Python 3.12 and uv installed:

```sh
uv sync --project robosuite --locked --python 3.12
make autoresearch-robosuite
```

This environment pins Robosuite 1.5.2 and MuJoCo 3.3.7. LIBERO-PRO has its own incompatible pins for Robosuite 1.4.0 and MuJoCo 2.3.7. The Robosuite wheel supplies the Panda and scene assets. Evaluation uses CPU physics with camera observations, the viewer, and offscreen rendering disabled. The pinned interpreter defaults to `robosuite/.venv/bin/python`. To use another installation of the same environment, export `WORLDSMITH_ROBOSUITE_PYTHON` with its absolute interpreter path. [Robosuite release](https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.2)

| Worldsmith task | Native environment | Setup | Control steps |
| --- | --- | --- | ---: |
| `open_latched_door` | Door | One Panda, native latch enabled | 300 |
| `assemble_nuts` | NutAssembly | One Panda, both square and round nuts | 450 |
| `handover_hammer` | TwoArmHandover | Two opposed Pandas, hammer starts on the table | 400 |

Worldsmith uses the native scene and success predicate, with sparse native rewards in the owner trace. It supplies structured observations and configures the Panda's operational-space controller for world-frame pose deltas. The Worldsmith budgets cap rollouts at 15, 22.5, and 20 simulated seconds. Compare scores within this protocol. See the [public task guide](../autoresearch/TASKS.md) and [action conventions](../autoresearch/API.md).

Each seed constructs a fresh native model, including randomized hammer geometry, and seeds object placement and arm initialization. Development uses seeds 0–7. For held-out evaluation, create a private JSON file mapping the three Worldsmith task IDs to the first seed of three distinct eight-seed ranges. Every range must lie within 8–4294967295. From a saved controller snapshot's `code/` directory, use absolute paths:

```sh
PYTHONPATH="$repo" "$repo/robosuite/.venv/bin/python" -B -m worldsmith.research_evaluation \
  robosuite "$private_output" --heldout-seed-file "$private_seed_file"
```

The development tool publishes task scores, controller observations and actions, and episode outcomes. Owner artifacts retain the upstream environment names, runtime versions, seeds, rewards, and worker logs under `runs/owner/`.

## Verification

```sh
robosuite/.venv/bin/python -m pytest -q tests/robosuite_native.py
robosuite/.venv/bin/python -m worldsmith.benchmark.evaluator \
  --task open_latched_door --episodes 1 --seed 0 --output runs/door-dev
```

The native tests exercise same-seed resets across episode order, action validation, horizon truncation, and a seed-0 controller that opens the latched door using public observations and commands. The shared AutoResearch starter holds its pose with open grippers.

## Local runtime

Measured on an Apple M2 Pro with 16 GB RAM, macOS 27.2.1, and Python 3.12.4. The 24-episode dedicated starter evaluation took **66.8 s** end to end, including publication. The per-task timings below are from the combined run. They include resets and owner trace writes, with interpreter startup included in the first task.

| Task | Starter successes / 8 | Eight episodes | Mean per episode |
| --- | ---: | ---: | ---: |
| Door | 0 / 8 | 13.9 s | 1.73 s |
| Nut assembly | 0 / 8 | 22.9 s | 2.86 s |
| Hammer handover | 0 / 8 | 31.1 s | 3.89 s |

In the same combined run, the Robosuite worker took 68.1 s and the LIBERO-PRO worker took 168.1 s for 24 episodes each, under their respective task budgets. Adding the new worker cost about 68 s plus the additional feedback publication. The full 136-episode `worldsmith-all-v3` run took **257.3 s** end to end, scored 53 successes (38.97%), and finished within the six-minute budget. The first run with a cold Numba cache can take longer.
