# Robosuite on Apple Silicon

From the repository root, with arm64 Python 3.12 and uv installed, prepare the locked environment and run the native tests.

```sh
uv sync --project robosuite --locked --python 3.12
robosuite/.venv/bin/python -m pytest -q tests/robosuite_native.py
make autoresearch-robosuite
```

The runtime pins Robosuite 1.5.2 and MuJoCo 3.3.7 separately from the LIBERO-PRO environment. Evaluation uses CPU physics with camera observations and rendering disabled. The wheel supplies the Panda and scene assets.
