# ManiSkill on Apple Silicon

Use arm64 Python 3.11, uv, and Homebrew in a local macOS login session. SAPIEN initializes Vulkan even for headless evaluation. Its device discovery calls AppKit and can abort in a sandbox that blocks macOS application registration.

From the repository root, install Vulkan, prepare the locked environment, and download the Go2 assets.

```sh
brew install molten-vk vulkan-loader
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix vulkan-loader)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

uv sync --project simulation_environments/maniskill --locked --python 3.11
uv run --project simulation_environments/maniskill --locked --python 3.11 python -m mani_skill.utils.download_asset unitree_go2 -y
```

Keep the Vulkan exports in the shell used for testing or launching research.

```sh
simulation_environments/maniskill/.venv/bin/python -m pytest -q \
  tests/maniskill_push_cube.py tests/maniskill_manipulation.py tests/maniskill_diversity.py
make autoresearch-maniskill
```
