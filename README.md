# Worldsmith

See [project.md](project.md) for the research idea. The first benchmark tests whether a simulated robot can intercept a moving target using noisy position measurements.

## Run the benchmark

```sh
uv sync --locked
uv run --locked python -m worldsmith evaluate --controller chase --output runs/chase-dev
uv run --locked python -m worldsmith evaluate --controller lead --output runs/lead-dev
uv run --locked python -m pytest
```

Each run writes `summary.json` and `episodes.jsonl` to a new output directory and prints the summary as JSON on stdout. Each trace contains the initial observation and the sequence of actions, resulting observations, and distances used by the scorer. Use `--split heldout` for comparisons after development, or `--seed 12` to replay a single scenario. Both splits have 32 fixed seeds.

The robot and target move in an obstacle-free plane. The robot commands heading and throttle. The target moves at constant velocity sampled for each episode. Capture means coming within 0.3 simulation units of the target within 45 steps. `chase` aims at the latest measurement. `lead` estimates motion from recent measurements and aims ahead.

## Try a candidate

Put candidate code in `worldsmith/world/`. The entry point is a zero-argument factory that creates a fresh controller for each episode. The controller takes an `Observation` and returns an `Action` from `worldsmith.api`. Observations contain time, the robot's maximum speed, and noisy robot and target positions with 2×2 measurement noise covariances. A controller can retain observations to build its own representation, estimator, model, or skill.

```sh
# After adding worldsmith/world/candidate.py with a make_controller factory
uv run --locked python -m worldsmith evaluate \
  --controller worldsmith.world.candidate:make_controller --output runs/candidate-dev
```

For AutoResearch comparisons, allow edits under `worldsmith/world/` and keep the API, CLI, simulator, references, and scorer fixed. The CLI imports trusted local Python code in-process.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
