# ManiSkill PushCube

Run ManiSkill's `PushCube-v1` with a Panda arm, PhysX CPU simulation, and a scripted Python controller.

## Run on Apple Silicon

Use an arm64 Python 3.11 interpreter and [uv](https://docs.astral.sh/uv/) in a local macOS login session. The commands below were tested on an Apple M2 Pro with macOS 27.2.1, Python 3.11.9, MoltenVK 1.4.2, and Vulkan Loader 1.4.363.0. ManiSkill initializes SAPIEN's Vulkan renderer on macOS even for `state_dict` with `render_mode=None`. The run opens no viewer and captures no frames. SAPIEN's device discovery calls AppKit and can abort in a sandbox that blocks macOS application registration. [ManiSkill macOS installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html)

From the repository root, with Python 3.11 and Homebrew available:

```sh
brew install molten-vk vulkan-loader
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix vulkan-loader)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

uv sync --project maniskill --locked --python 3.11
uv run --project maniskill --locked --python 3.11 python -m worldsmith.benchmark.evaluate_push_cube \
  --episodes 3 --seed 42 --output runs/push-cube-benchmark
```

The optional ManiSkill environment pins ManiSkill nightly `2026.8.2.400`, SAPIEN `3.0.3`, Pinocchio `4.1.0`, PyTorch `2.15.0`, Gymnasium `1.4.0`, and NumPy `2.4.6`. SAPIEN uses Pinocchio for the Panda's native CPU inverse kinematics. Use a new output path for each run. Episode seeds are consecutive integers starting at `--seed`. The opt-in integration checks cover same-seed resets in the tested stack. Trajectories can vary across simulator versions and hardware.

## Controller and scoring

- **Observation.** The controller receives the original nested `state_dict` of CPU tensors with a leading batch dimension of one. `agent` contains joint positions and velocities. `extra` contains tool-center and cube poses as XYZ plus quaternion and the goal's XYZ position.
- **Action.** The Panda's `pd_ee_delta_pos` mode accepts four `float32` values in `[-1, 1]`. The first three specify displacement along the robot base axes, with full scale corresponding to 0.1 m on each axis. The fourth commands the gripper, with `-1` closing it. Worldsmith's scripted baseline uses the observed poses to approach from behind the cube, lower the closed gripper, and push toward the goal.
- **Success.** The task reports success when the cube's XY position is within 0.10 m of the goal center and its center height is below 0.025 m. ManiSkill terminates on success and truncates at 50 steps.

## Artifacts and tests

- `summary.json` records simulator and task versions, runtime configuration, seeds, lengths, per-episode success, and the overall success rate.
- `episodes.jsonl` contains one record per episode. Each step stores the exact structured observation passed to the controller and the returned action, plus the simulator's reward, termination flags, and `info`. The final state and reset diagnostics are stored separately from the controller inputs.

The example run succeeded on seeds 42, 43, and 44 in 16, 17, and 16 steps. Its episode trace was about 50 kB. This sample verifies the evaluation path.

```sh
uv run --project maniskill --locked --python 3.11 python -m pytest tests/maniskill_push_cube.py
uv run --locked python -m pytest
```
