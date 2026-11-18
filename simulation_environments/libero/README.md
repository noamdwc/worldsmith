# LIBERO-PRO on Apple Silicon

From the repository root, with arm64 Python 3.11, uv, and curl installed, run the setup script. It installs the locked Python environment, checks out the pinned LIBERO-PRO source, downloads the task files, and verifies their checksums. The evaluator verifies the assets before loading the saved layouts.

```sh
./simulation_environments/libero/setup.sh
simulation_environments/libero/.venv/bin/python -m pytest -q tests/libero_pro.py
make autoresearch-libero
```

The evaluator disables the MuJoCo graphics backend, camera observations, and offscreen rendering. LIBERO-PRO uses its own Robosuite and MuJoCo pins.

For a historical seed-file replay, keep each task's eight distinct layout indices within 8–49, with no overlap between tasks. Out-of-range seeds fail.
