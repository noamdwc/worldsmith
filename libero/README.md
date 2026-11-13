# LIBERO-PRO on Apple Silicon

From the repository root, with arm64 Python 3.11, uv, and curl installed:

```sh
./libero/setup.sh
make autoresearch-libero
```

Setup fetches the pinned [LIBERO-PRO source](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec), the six task files from the [official dataset revision](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/tree/c86fc3b8293185a6f373677018ff3e37f8391602), and an isolated Python environment. The checkout and environment use about 2.3 GiB after installation. The LIBERO-PRO tests verify the pinned checkout and allow the six downloaded assets, Python bytecode caches, and the MuJoCo log. The evaluator verifies task file checksums before using them, including before deserializing legacy NumPy initial states stored by PyTorch.

The native environments use MuJoCo 2.3.7, robosuite 1.4.0, and the stock Panda operational-space pose controller. An Apple M2 Pro on macOS 27.2.1 failed to import the default CGL backend because the OpenGL framework could not be loaded. With `MUJOCO_GL=disable` set before import and both camera observations and offscreen rendering disabled, the original environments ran locally on CPU. PyTorch loads the initial-state files. The simulation path needs no CUDA, policy checkpoints, demonstration datasets, or training stack. ASPIRE's published agent workflow additionally uses GPU perception and gated SAM3 weights. [ASPIRE LIBERO-PRO workflow](https://github.com/NVlabs/ASPIRE/blob/main/docs/libero-pro.md)

## Evaluation protocol

| Worldsmith task | Official suite and task index | Native goal | Control steps |
| --- | --- | --- | ---: |
| `open_middle_drawer` | `libero_goal_swap:0` | Open the middle drawer | 300 |
| `place_bowl_on_plate` | `libero_spatial_swap:0` | Put the designated bowl on the plate | 220 |
| `place_two_items_in_basket` | `libero_10_swap:0` | Put both requested objects inside the basket | 520 |

These three position-swap tasks cover articulated manipulation, spatial selection and placement, and a multi-object goal. Each reset applies the official saved layout at the requested index and settles for ten open-gripper hold steps before the controller acts. The native success flag terminates the episode. The upstream OpenVLA evaluation budgets cap policy steps. Scores are binary per episode, averaged within each task and then equally across tasks. [LIBERO-PRO evaluation protocol](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec#libero-pro-evaluation), [OpenVLA-OFT settling action](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/libero_utils.py)

Worldsmith exposes structured arm and object observations, plus handle position and drawer displacement for the drawer task. This changes the policy's observation modality relative to the published vision workflows. Compare results within this Worldsmith suite. The starter issues a stationary open-gripper command for all three tasks.

Development uses layout indices 0–7. For an owner-only held-out evaluation, create a private JSON file mapping the three task IDs to distinct eight-index ranges wholly within 8–49. An out-of-range seed fails rather than wrapping to a development layout. From a saved controller snapshot's `code/` directory, run the same evaluator with the pinned Python and absolute paths:

```sh
PYTHONPATH="$repo" "$repo/libero/.venv/bin/python" -B -m worldsmith.research_evaluation \
  libero "$private_output" --heldout-seed-file "$private_seed_file"
```

The development tool publishes the per-task scores, outcomes, and the exact controller inputs and actions. The owner artifacts retain native task identifiers, version pins, layout indices, rewards, and native step information under `runs/owner/`.

## Verification

```sh
libero/.venv/bin/python -m pytest -q tests/libero_pro.py
libero/.venv/bin/python -m worldsmith.benchmark.evaluator \
  --task open_middle_drawer --episodes 1 --seed 0 --output runs/libero-drawer
```

Native tests exercise deterministic resets, distinct layouts, input validation, step-limit truncation, and the published trace shape. A positive control places the drawer in an open state and checks that the next native step terminates successfully and receives reward 1.

On the Apple M2 Pro with 16 GB RAM, the starter completed all 24 development episodes in **210.5 s** end to end, including worker startup and feedback publication. Per-task timings below cover the eight resets and rollouts plus writing their owner traces. The public trace file was 29 MiB.

| Task | Successes / 8 | Eight rollouts | Mean per rollout |
| --- | ---: | ---: | ---: |
| Middle drawer | 0 / 8 | 48.7 s | 6.1 s |
| Bowl on plate | 0 / 8 | 53.0 s | 6.6 s |
| Two items in basket | 0 / 8 | 72.4 s | 9.1 s |

The dedicated suite gives its worker five minutes per evaluation. The historical `worldsmith-all-v2` suite completed 112 episodes in 190.5 s on the same machine in a separate run.

To add tasks, pin and verify each task's upstream BDDL and saved states, check that it has enough distinct layouts for the split, document any additional observations, and time the full development slice. Visual perturbations require a working renderer and separate validation. For a future Colab run, validate a Linux CPU lock for state-only evaluation first. A visual evaluation additionally needs a functioning MuJoCo rendering backend.
