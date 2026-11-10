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

## AutoResearch workspace

Write research controllers and reusable skills in [autoresearch/](autoresearch/). The [research run guide](autoresearch/RESEARCH.md) explains how to launch an interactive or noninteractive Codex session, run three candidate iterations through the project-scoped MCP tools, and record results in the [journal](autoresearch/JOURNAL.md).

## Task and simulator boundary

`benchmark/simulation.py` advances the bodies and produces observations. `benchmark/tasks/` defines the `Task` interface. Its `interception.py` defines the scene, goal, time budget, scoring, and the two task-specific reference controllers. The evaluator runs the episode and records the measurements and scores provided by the task. Interception is the single task exposed by the CLI.

The benchmark ID is `interception-v2`. In v2 traces, positions are under `observation.entities`, and the summary includes the task goal.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
