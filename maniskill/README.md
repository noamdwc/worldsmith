# ManiSkill manipulation

Evaluate six native ManiSkill tasks with the Worldsmith evaluator on Apple Silicon.

## Run on Apple Silicon

Use an arm64 Python 3.11 interpreter and [uv](https://docs.astral.sh/uv/) in a local macOS login session. The commands below were tested on an Apple M2 Pro with macOS 27.2.1, Python 3.11.9, MoltenVK 1.4.2, and Vulkan Loader 1.4.363.0. ManiSkill initializes SAPIEN's Vulkan renderer on macOS even for `state_dict` with `render_mode=None`. The run opens no viewer and captures no frames. SAPIEN's device discovery calls AppKit and can abort in a sandbox that blocks macOS application registration. [ManiSkill macOS installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html)

From the repository root, with Python 3.11 and Homebrew available:

```sh
brew install molten-vk vulkan-loader
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix vulkan-loader)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

uv sync --project maniskill --locked --python 3.11
uv run --project maniskill --locked --python 3.11 python -m worldsmith.benchmark.evaluator \
  --task push_cube --episodes 3 --seed 42 --output runs/push-cube-benchmark
```

The optional ManiSkill environment pins ManiSkill nightly `2026.8.2.400`, SAPIEN `3.0.3`, Pinocchio `4.1.0`, PyTorch `2.15.0`, Gymnasium `1.4.0`, and NumPy `2.4.6`. SAPIEN uses Pinocchio for the Panda's native CPU inverse kinematics. Use a new output path for each run. Episode seeds are consecutive integers starting at `--seed`. The opt-in integration tests cover same-seed resets on this stack. Trajectories can vary across simulator versions and hardware.

## Tasks

All tasks use one environment, `state_dict` observations, PhysX CPU simulation, the SAPIEN CPU render backend, and `render_mode=None`. The pinned packages supply the robot and scene assets for all six tasks. PegInsertionSide rebuilds the scene at each reset to sample its peg and hole geometry, following the native single-environment default.

| CLI task | Official environment | Robot | Control mode | Action size | Step limit | Scripted baseline |
| --- | --- | --- | --- | ---: | ---: | --- |
| `push_cube` | `PushCube-v1` | `panda` | `pd_ee_delta_pos` | 4 | 50 | Sweep the cube toward the goal with the closed gripper |
| `pick_cube` | `PickCube-v1` | `panda` | `pd_ee_delta_pos` | 4 | 50 | Grasp and carry the cube to the goal |
| `stack_cube` | `StackCube-v1` | `panda_wristcam` | `pd_ee_delta_pos` | 4 | 50 | Grasp the first cube, lift it above the second, then lower and release |
| `peg_insertion_side` | `PegInsertionSide-v1` | `panda_wristcam` | `pd_ee_delta_pose` | 7 | 100 | Approach, grasp, and lift the peg |
| `pull_cube` | `PullCube-v1` | `panda` | `pd_ee_delta_pos` | 4 | 50 | Reach over the cube and sweep it back with the closed gripper |
| `push_t` | `PushT-v1` | `panda_stick` | `pd_ee_delta_pos` | 3 | 100 | Push the T toward the target center using its observed position |

Use `--controller module:factory` to select an importable, zero-argument Python factory. The evaluator creates a new controller for each episode and passes it ManiSkill's original nested dictionary of CPU tensors with a leading batch dimension of one. `agent` contains robot joint state. `extra` contains task observations such as TCP pose, object pose, and target position.

Return a NumPy `float32` array with the action size above and values in `[-1, 1]`. The first three values command XYZ displacement in the robot base frame with a full scale of 0.1 m per axis. The pose mode adds three rotation commands before the gripper command. On gripper-equipped arms, the final value commands the gripper, with `-1` closing it.

The evaluator uses each environment's native success flag and stops on native termination or truncation. PegInsertionSide scores the peg head in the hole's frame. PushT scores overlap with the goal T, so reaching its center alone can still fail.

## Artifacts and tests

- `summary.json` records simulator and task versions, runtime configuration, seeds, lengths, per-episode success, and the overall success rate.
- `episodes.jsonl` contains one record per episode. Each step stores the exact structured observation passed to the controller and the returned action, plus the simulator's reward, termination flags, and `info`. The final state and reset diagnostics are stored separately from the controller inputs.

On seeds 42, 43, and 44, the PushCube, PickCube, StackCube, and PullCube baselines succeeded in all three episodes. The PegInsertionSide baseline lifted the peg in each episode and timed out before insertion. The PushT baseline moved the T in each episode and timed out below the required overlap. These are three-episode checks of the local evaluation path.

```sh
uv run --project maniskill --locked --python 3.11 python -m pytest \
  tests/maniskill_push_cube.py tests/maniskill_manipulation.py
uv run --locked python -m pytest
```
