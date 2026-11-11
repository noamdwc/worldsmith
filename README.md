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

Each owner CLI run writes `summary.json` and `episodes.jsonl` to a new output directory and prints the summary as JSON on stdout. Owner traces include observations, actions, true positions and velocities, and the per-step measurements used by the scorer. `initial_observation.goal` identifies the task. Use `--task interception`, `--task navigation`, or `--task rendezvous` for independent evaluations. Omitting `--task` selects interception and its Chase default. Other tasks and `--task all` default to the shared starter in `autoresearch/controller.py`. Select any shared factory with `--controller module:make_controller`.

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

`start` prepares the shared reactive starter controller and a fresh journal. Its interception behavior remains Chase; navigation aims at the destination and turns away from nearby obstacles, while rendezvous approaches the current target position and slows down nearby. It uses no observation history, motion estimation, or prediction. If the workspace contains research changes or results from earlier work, archive and reset it before starting.

Run `make clear-autoresearch` to archive the current research work and reset the workspace to the shared starter controller and fresh journal. The command prints the archive location. Close the research session and wait for evaluations to finish before clearing.

Reset checks that every current research file is present and unchanged in a completed archive before restoring the original controller and journal. If there are edits after archiving, archive again. Close the research session and wait for any evaluation to finish before running lifecycle commands.

### Evaluation feedback and owner results

`evaluate_controller()` runs the combined development split. `get_episode_trace(seed, task="interception")` selects a task and seed from the latest successful run in the current MCP session. Use `"navigation"` or `"rendezvous"` for the other tasks. Run held-out evaluations through the owner CLI with `--split heldout`.

Each MCP evaluation writes two views with the same run ID and controller snapshot:

- `autoresearch/runs/<run-id>/` stores the research snapshot, status, and public results. `results/episodes.jsonl` contains the initial observation, each action and resulting observation, and the episode outcome. `results/summary.json` contains aggregate and per-task metrics. Archives preserve these files under `research-history/<archive-id>/workspace/`.
- `runs/owner/<run-id>/` stores the exact executed snapshot, status, raw results, and process diagnostics when available. The owner trace includes true positions and velocities at the initial state and every step, scoring measurements, and episode diagnostics. These files survive workspace resets.

[research_feedback.py](worldsmith/research_feedback.py) defines which fields are published to the researcher. MCP serves traces from that published view. Controller errors include debugging details, and evaluator tracebacks are saved with owner results. A status of `preparing` or `running` indicates an interrupted evaluation.

The directories and MCP tools separate the two views in the normal workflow. The local launcher uses the same OS account for the researcher and evaluator. Shell tools and controller code with filesystem read access can still read the owner files and evaluator source. This layout provides no OS-level confidentiality.

## Task and simulator boundary

`benchmark/simulation.py` advances the bodies and produces observations. `benchmark/tasks/` defines the `Task` interface. Its `interception.py` defines the scene, goal, time budget, scoring, and the two task-specific reference controllers. The evaluator runs the episode and records the measurements and scores provided by the task. The CLI exposes all three tasks independently or together with `--task all`. Navigation scores destination arrival without obstacle contact; rendezvous scores 20 consecutive steps within a tracking radius. Both reuse the existing Pymunk simulation, Observation/Action contract, and episode evaluator.

In traces, positions are under `observation.entities` and the task name is under `observation.goal`. The summary's `goal` field contains the human-readable objective (a mapping of task names to objectives for combined runs). Read the [public task brief](autoresearch/TASKS.md) for success criteria, limits, and metrics. See the [World API](autoresearch/API.md) for the controller contract.

Existing scenes use `Entity` for shape-free kinematic motion. Wrap a Pymunk body that is outside any space in a `PhysicalEntity`. Shape-free bodies are valid. For contact interactions, attach a `pymunk.Circle` or `pymunk.Segment` before constructing the simulation. A dynamic body responds to forces, impulses, and collisions according to its mass and moment of inertia. Static bodies provide fixed collision geometry. A shaped kinematic body can move dynamic objects while keeping its prescribed velocity. Set elasticity and contact friction on both shapes in a collision pair. Scene code can keep a body reference to apply Pymunk forces each step or apply an impulse once.

Physical entities are hidden by default. Set `observable=True` to include a position measurement. `position_noise` sets the standard deviation, with the default `0.0` giving a perfect measurement. Owner traces include the physical body's type, pose, velocities, dynamic mass and moment, and shape materials. An infinite moment for locked rotation is recorded as `null`. `Action(heading, throttle)` still sets the velocity of the observed kinematic robot. The simulator takes one discrete physics step every 0.1 seconds, so scene authors should choose speeds and shape sizes that keep bodies from tunneling through contacts.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
