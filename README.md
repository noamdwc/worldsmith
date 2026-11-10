# Worldsmith

See [project.md](project.md) for the research idea. The first benchmark tests whether a simulated robot can intercept a moving target using noisy position measurements.

## Run the benchmark

```sh
uv sync --locked
make chase
make lead
uv run --locked python -m pytest
```

Each Make target evaluates one controller on the development split and uses a timestamped directory under `runs/`. To choose the directory, run `make chase OUTPUT=runs/chase-dev` with a path that does not yet exist.

Each run writes `summary.json` and `episodes.jsonl` to a new output directory and prints the summary as JSON on stdout. Each trace contains the initial observation and the sequence of actions, resulting observations, and distances used by the scorer. Use `--split heldout` for comparisons after development, or `--seed 12` to replay a single scenario. Both splits have 32 fixed seeds.

The robot and target move in an obstacle-free plane. The robot commands heading and throttle. The target moves at constant velocity sampled for each episode. Capture means coming within 0.3 simulation units of the target within 45 steps. `chase` aims at the latest measurement. `lead` estimates motion from recent measurements and aims ahead.

## AutoResearch workspace

Run `make autoresearch` from the repo root to start the noninteractive Codex session. The [research run guide](autoresearch/RESEARCH.md) covers workspace trust, interactive launch, and the three-iteration workflow. Results are recorded in the [journal](autoresearch/JOURNAL.md).

### Session lifecycle

From the repo root, the workspace owner runs:

```sh
uv run --locked python -m worldsmith.research_lifecycle start
make autoresearch
# End the Codex session before archiving or resetting.
archive_id=$(uv run --locked python -m worldsmith.research_lifecycle archive)
uv run --locked python -m worldsmith.research_lifecycle reset "$archive_id"
```

`start` prepares the original Chase controller and a fresh journal. If the workspace contains research changes or results from earlier work, archive and reset it before starting.

Each MCP evaluation saves its source under `runs/<run-id>/code/`, status in `evaluation.json`, and metrics and traces in `results/` when available. A record left in `preparing` or `running` indicates an interrupted evaluation. Inspect archived journals, final code, and evaluations directly in `research-history/<archive-id>/workspace/`. The archive's `metadata.json` records the Git commit and UTC timestamp at archive time.

Reset checks that every current research file is present and unchanged in a completed archive before restoring the original controller and journal. If there are edits after archiving, archive again. Close the research session and wait for any evaluation to finish before running lifecycle commands.

## Task and simulator boundary

`benchmark/simulation.py` advances the bodies and produces observations. `benchmark/tasks/` defines the `Task` interface. Its `interception.py` defines the scene, goal, time budget, scoring, and the two task-specific reference controllers. The evaluator runs the episode and records the measurements and scores provided by the task. Interception is the single task exposed by the CLI.

The benchmark ID is `interception-v2`. In v2 traces, positions are under `observation.entities`, and the summary includes the task goal.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
