# Worldsmith

See [project.md](project.md) for the research idea. In V1, one shared Python controller receives the task name and sensor measurements at runtime. The combined benchmark tests interception, moving-obstacle navigation, and rendezvous / tracking using noisy position measurements.

## Run the benchmark

```sh
uv sync --locked
make chase
make lead
uv run --locked python -m worldsmith evaluate --task all --output runs/combined-dev
uv run --locked python -m pytest
```

Each Make target evaluates one controller on the development split and uses a timestamped directory under `runs/`. To choose the directory, run `make chase OUTPUT=runs/chase-dev` with a path that does not yet exist.

Each run writes `summary.json` and `episodes.jsonl` to a new output directory and prints the summary as JSON on stdout. Each trace contains the initial observation and the sequence of actions, resulting observations, and task measurements used by the scorer. `initial_observation.goal` identifies its task. Use `--task interception`, `--task navigation`, or `--task rendezvous` for independent evaluations; omitting `--task` preserves interception evaluation and its Chase default. Other tasks and `--task all` default to the shared starter in `autoresearch/controller.py`. Select any shared factory with `--controller module:make_controller`.

The combined benchmark (`worldsmith-v1`) uses 8 fixed seeds per task, 24 episodes total. `summary.json` includes per-task metrics under `tasks` and `aggregate_score`, the equal-weight average of the three success rates. Compare those task rows between runs to see improvements and regressions. Interception retains its capture metrics and also exposes `success_rate` in the combined summary.

Use `--split heldout` for owner comparisons after development, or `--seed 0` to replay a scenario (one episode per task with `--task all`). Development seeds are 0–7 and held-out seeds are 1000–1007 for the combined benchmark and the new tasks. Independent interception retains its original 32-seed splits (0–31 and 1000–1031) and benchmark ID `interception-v2`.

For interception, the robot and target move in an obstacle-free plane. The robot commands heading and throttle. The target moves at constant velocity sampled for each episode. Capture means coming within 0.3 simulation units of the target within 45 steps. `chase` aims at the latest measurement. `lead` estimates motion from recent measurements and aims ahead.

## AutoResearch workspace

Run `make autoresearch` from the repo root or invoke [scripts/run-autoresearch.sh](scripts/run-autoresearch.sh) directly. The script starts a clean workspace, runs Codex noninteractively, archives its results, and resets to the original controller. If Codex exits with an error, the script archives the work and leaves the workspace in place. The [research run guide](autoresearch/RESEARCH.md) covers workspace trust, interactive launch, and the three-iteration workflow.

### Session lifecycle

For a manually launched session, run these lifecycle commands from the repo root:

```sh
uv run --locked python -m worldsmith.research_lifecycle start
# Launch Codex as described in the research run guide, then wait for it to exit.
archive_id=$(uv run --locked python -m worldsmith.research_lifecycle archive)
uv run --locked python -m worldsmith.research_lifecycle reset "$archive_id"
```

`start` prepares the shared starter controller and a fresh journal. Its interception behavior remains Chase; navigation predicts obstacle crossings and rendezvous estimates target velocity and adjusts speed. If the workspace contains research changes or results from earlier work, archive and reset it before starting.

`evaluate_controller()` runs all three tasks together by default. `get_episode_trace(seed, task="interception")` selects a task and seed from the latest successful evaluation; the task can also be `"navigation"` or `"rendezvous"`. Each MCP evaluation saves its source under `runs/<run-id>/code/`, status in `evaluation.json`, and metrics and traces in `results/` when available. A record left in `preparing` or `running` indicates an interrupted evaluation. Inspect archived journals, final code, and evaluations directly in `research-history/<archive-id>/workspace/`. The archive's `metadata.json` records the Git commit and UTC timestamp at archive time.

Reset checks that every current research file is present and unchanged in a completed archive before restoring the original controller and journal. If there are edits after archiving, archive again. Close the research session and wait for any evaluation to finish before running lifecycle commands.

## Task and simulator boundary

`benchmark/simulation.py` advances the bodies and produces observations. `benchmark/tasks/` defines the `Task` interface. Its `interception.py` defines the scene, goal, time budget, scoring, and the two task-specific reference controllers. The evaluator runs the episode and records the measurements and scores provided by the task. The CLI exposes all three tasks independently or together with `--task all`. Navigation scores destination arrival without obstacle contact; rendezvous scores 20 consecutive steps within a tracking radius. Both reuse the existing Pymunk simulation, Observation/Action contract, and episode evaluator.

In traces, positions are under `observation.entities` and the task name is under `observation.goal`. The summary's `goal` field contains the human-readable objective (a mapping of task names to objectives for combined runs). Read the [public task brief](autoresearch/TASKS.md) for success criteria, limits, and metrics. See the [World API](autoresearch/API.md) for the controller contract.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
