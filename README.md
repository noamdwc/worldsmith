# Worldsmith

See [project.md](project.md) for the research idea. One shared Python controller receives the task name and sensor measurements at runtime. The combined benchmark tests interception, moving-obstacle navigation, rendezvous / tracking, bouncing-object interception, and object pushing using noisy position measurements.

[ManiSkill PushCube](maniskill/README.md) runs a 3D robotic manipulation task with a Panda arm.

The full Worldsmith v2 research suite also covers tabletop manipulation, tool use, dual-arm transfer, and quadruped locomotion. The [20-task inventory](BENCHMARKS.md) describes each task’s embodiment, controls, observations, and physical interactions.

## Run the benchmark

The shared evaluator runs individual tasks from all four simulation families and writes
`summary.json` and `episodes.jsonl`. Use `--controller module:factory` to supply a
factory that creates a fresh controller for each episode.

```sh
uv run --locked python -m worldsmith.benchmark.evaluator \
  --task navigation --episodes 3 --seed 42 --output runs/navigation
```

See [ManiSkill PushCube](maniskill/README.md) for the same command in the pinned
ManiSkill environment. The V1 research commands below use their established seed
splits, combined benchmark, and trace format.

Python callers migrating from `worldsmith.benchmark.evaluate` use
`worldsmith.benchmark.evaluator.evaluate` with an ordered mapping of task names
to open adapters, a controller factory, explicit seeds, and a trace recorder.
The research CLI supplies its recording and summary policies from
`worldsmith.benchmark.pymunk_adapter`.

```sh
uv sync --locked
make chase
make lead
uv run --locked python -m worldsmith evaluate --task all --output runs/combined-dev
uv run --locked python -m pytest
```

Each Make target evaluates one controller on the development split and uses a timestamped directory under `runs/`. To choose the directory, run `make chase OUTPUT=runs/chase-dev` with a path that does not yet exist.

Each owner CLI run writes `summary.json` and `episodes.jsonl` to a new output directory and prints the summary as JSON on stdout. Owner traces include observations, actions, true positions and velocities, and the per-step measurements used by the scorer. `initial_observation.goal` identifies the task. Use `--task interception`, `--task navigation`, `--task rendezvous`, `--task bouncing_interception`, or `--task object_pushing` for independent evaluations. Omitting `--task` selects interception and its Chase default. Other tasks and `--task all` default to the shared starter in `autoresearch/controller.py`. Select any shared factory with `--controller module:make_controller`.

The combined benchmark (`worldsmith-v3`) uses 8 fixed seeds per task, 40 episodes total. `summary.json` includes per-task metrics under `tasks` and `aggregate_score`, the equal-weight average of the five success rates. Compare those task rows between runs to see improvements and regressions. Both interception tasks report capture metrics and also expose `success_rate` in the combined summary.

Use `--split heldout` for owner comparisons after development, or `--seed 0` to replay a scenario (one episode per task with `--task all`). Development seeds are 0–7 and held-out seeds are 1000–1007 for the combined benchmark and independent navigation, rendezvous, bouncing interception, and object pushing. Independent interception retains its original 32-seed splits (0–31 and 1000–1031) and benchmark ID `interception-v2`.

For interception, the robot and target move in an obstacle-free plane. The robot commands heading and throttle. The target moves at constant velocity sampled for each episode. Capture means coming within 0.3 simulation units of the target within 45 steps. `chase` aims at the latest measurement. `lead` estimates motion from recent measurements and aims ahead.

### Parallel evaluation

The shared benchmark CLI, the Pymunk split CLI, the owner group CLI, and
AutoResearch default to **1 worker** for Pymunk and ManiSkill and **2 workers**
for LIBERO-PRO and Robosuite. The full AutoResearch suite uses each group's
default, just as dedicated suites do. Run each simulator with its pinned Python.

```sh
uv run --locked python -m worldsmith evaluate --task all --workers 2 --output runs/combined-parallel
libero/.venv/bin/python -m worldsmith.benchmark.evaluator \
  --task open_middle_drawer --episodes 8 --workers 4 --output runs/drawer-parallel
robosuite/.venv/bin/python -B -m worldsmith.owner_evaluation \
  robosuite runs/robosuite-parallel --workers 4
WORLDSMITH_EVALUATION_WORKERS=2 make autoresearch
```

`--workers N` overrides the default in the evaluation CLIs.
`WORLDSMITH_EVALUATION_WORKERS` overrides the count for every group in an
AutoResearch session, including the full suite.
The MCP tool still evaluates the development split. Owner commands accept the
same held-out seed options as before. The suite deadlines also apply to
parallel runs.

Workers use fresh Python processes and own their simulator instances. Each
task uses at most one worker per episode, and tasks run in the established
order. The evaluator assigns the original seeds and writes traces and
aggregates scores in seed order. Controllers should use episode-local state
and deterministic logic for reproducible comparisons. Python callers can
pass `workers=N` to `evaluate_to_directory` with a picklable, importable
adapter opener and controller factory. The factory can still return a closure.
Use the standard `if __name__ == "__main__":` guard in calling scripts.

Add workers if the workload benefits and the machine can fit the
additional simulator memory. ManiSkill uses CPU physics and initializes
Vulkan on macOS. It now rebuilds the physics scene for every seed to discard
prior contacts. This fix can change earlier exploratory results. LIBERO-PRO
and Robosuite run with rendering disabled. Robosuite rebuilds the model for
each seed to preserve its seeded geometry.

An unexpected worker exit or a ten-minute wait for a worker response aborts
the owner task and cleans up its workers. AutoResearch applies its existing,
shorter suite deadlines to the entire process group. Failed runs can leave
partial owner traces without a completed summary.

#### Local timings

These runs used the pinned environments on an Apple M2 Pro with 16 GiB of
RAM and macOS 27.2.1.
Wall times include process startup, simulator initialization, rollouts, and
owner artifact writing. Speedup is relative to one worker in the same row.

| Workload | Episodes | 1 worker | 2 workers | 4 workers |
| --- | ---: | ---: | ---: | ---: |
| Pymunk object pushing, seeds 0–31 | 32 | 0.41 s | 0.43 s (0.95×) | 0.43 s (0.95×) |
| ManiSkill full development group, with clean resets | 48 | 19.52 s | 33.75 s (0.58×) | 31.83 s (0.61×) |
| LIBERO-PRO drawer opening, seeds 0–7 | 8 | 48.31 s | 26.96 s (1.79×) | 16.28 s (2.97×) |
| Robosuite full development group | 24 | 67.09 s | 39.49 s (1.70×) | 24.90 s (2.69×) |

For each workload, the three runs produced byte-identical owner summaries
and episode traces, including lengths, native outcomes, and scores. The
ManiSkill comparison uses clean resets at every worker count.

In fresh Python processes, loading the task adapter and reaching the first
seeded reset took approximately 0.15 s for object pushing, 3.04 s for
ManiSkill PushCube, 6.45 s for the LIBERO-PRO drawer, and 2.42 s for the
Robosuite door. Each spawned worker pays startup costs for its task.
ManiSkill's short rollouts and native startup made extra workers slower on
this machine. The measurements cover the pinned state-only runtimes on
macOS. GPU physics, camera observations, and other platforms were outside
this validation.

## AutoResearch workspace

Run `make autoresearch` from the repo root for the 16 development tasks across Pymunk, ManiSkill, LIBERO-PRO, and Robosuite. Use `make autoresearch-maniskill`, `make autoresearch-libero`, or `make autoresearch-robosuite` for a dedicated suite. The script starts a clean workspace, runs Codex noninteractively, archives its results, and resets to the original controller. If Codex exits with an error, the script archives the work and leaves the workspace in place. The [research run guide](autoresearch/RESEARCH.md) covers workspace trust, interactive launch, and the three-iteration workflow.

Run `./libero/setup.sh` before launching the full suite or the LIBERO-PRO-only suite. The [LIBERO-PRO setup and evaluation notes](libero/README.md) cover the pinned runtime, measured cost, and owner evaluation.

Run `uv sync --project robosuite --locked --python 3.12` before launching the full suite or the Robosuite-only suite. See the [Robosuite guide](robosuite/README.md) for measurements, tests, and owner evaluation.

On Apple Silicon, prepare the pinned Python 3.11 environment, Vulkan loader, and Unitree Go2 asset using the [ManiSkill setup](maniskill/README.md). Export `VK_ICD_FILENAMES` and `DYLD_LIBRARY_PATH` as shown there before launching Codex. The MCP server runs Pymunk under its Python 3.12 interpreter and defaults to `maniskill/.venv/bin/python` for ManiSkill. Set `WORLDSMITH_MANISKILL_PYTHON` to an absolute interpreter path to use another installation of the pinned environment.

### Session lifecycle

For a manually launched session, run these lifecycle commands from the repo root:

```sh
uv run --locked python -m worldsmith.research_lifecycle start
# Use the same low reasoning effort as make autoresearch.
(cd autoresearch && codex -c 'model_reasoning_effort="low"' "Run the three-iteration AutoResearch session in RESEARCH.md.")
archive_id=$(uv run --locked python -m worldsmith.research_lifecycle archive)
uv run --locked python -m worldsmith.research_lifecycle reset "$archive_id"
```

`start` prepares the shared reactive starter controller and a fresh journal. It uses Chase for both interception goals. Navigation aims at the destination and turns away from nearby obstacles, while rendezvous approaches the current target position and slows down nearby. For object pushing, it moves behind the object, pushes while aligned, and eases off near the destination. The ManiSkill branches use simple waypoint and grasp routines with the public state observations. If the workspace contains research changes or results from earlier work, archive and reset it before starting.

Run `make clear-autoresearch` to archive the current research work and reset the workspace to the shared starter controller and fresh journal. The command prints the archive location. Close the research session and wait for evaluations to finish before clearing.

Reset checks that every current research file is present and unchanged in a completed archive before restoring the original controller and journal. If there are edits after archiving, archive again. Close the research session and wait for any evaluation to finish before running lifecycle commands.

For a manual ManiSkill-only session, export `WORLDSMITH_RESEARCH_SUITE=maniskill` before launching Codex and ask it to work on tabletop manipulation and quadruped locomotion.

### Evaluation feedback and owner results

Each `evaluate_controller()` call uses seeds 0–7 per shared task in the active suite. The default `worldsmith-all-v5` suite evaluates 128 episodes with the same controller snapshot. The ManiSkill suite (`worldsmith-maniskill-v3`) evaluates 56 episodes, LIBERO-PRO (`worldsmith-arm-position-v2`) evaluates 16, and Robosuite (`worldsmith-arm-skills-v3`) evaluates 24. All suites score the equal-weight mean of their task success rates. Earlier benchmark IDs identify historical task sets.

`get_episode_trace(seed, task)` selects an episode from the latest successful MCP evaluation. With no task specified, it returns `interception` for the full suite or `push_cube` for ManiSkill-only. Each run writes two views with the same run ID:

- `autoresearch/runs/<run-id>/` contains a copy of the snapshot, status, and public results. `results/summary.json` records configuration, runtime, aggregate, and per-task scores and deltas. `results/episodes.jsonl` contains controller-visible observations and actions and permitted outcomes. Archives preserve these files under `research-history/<archive-id>/workspace/`.
- `runs/owner/<run-id>/` contains the executed snapshot, raw artifacts organized by runtime and task, and process logs. These files survive workspace resets.

The publisher exposes only development feedback. Evaluator tracebacks and raw simulator diagnostics stay in owner files. A status of `preparing` or `running` indicates an interrupted evaluation. The local launcher uses the same OS account for the researcher and evaluator, so the directory layout and MCP allowlist separate feedback in normal use. Restrict OS access separately if running untrusted controller code.

### Final validation

Once research ends, use the run ID for the retained controller and evaluate its saved `code/` snapshot. Run these commands from the repository root with the pinned simulators set up.

```sh
repo=$PWD
selected_run=mcp-... # Run ID of the retained development controller
output="$repo/runs/validation/$selected_run"
(
  cd "$repo/runs/owner/$selected_run/code" || exit 1
  export PYTHONPATH="$repo"
  "$repo/.venv/bin/python" -B -m worldsmith.owner_evaluation pymunk "$output/pymunk" --validation || exit 1
  for group in maniskill libero robosuite; do
    "$repo/$group/.venv/bin/python" -B -m worldsmith.owner_evaluation "$group" "$output/$group" --validation || exit 1
  done
)
```

Validation uses seeds 8–15 for all 20 tasks. Read the shared-task summaries for seed generalization and the four validation-only task summaries for task generalization. The validation-only tasks are `rendezvous`, `peg_insertion_side`, `place_bowl_on_plate`, and `wipe_table`. The separation applies to new research sessions. Earlier benchmark runs included these tasks. [Owner task reference](BENCHMARKS.md#owner-reference-for-validation-only-tasks)

Final validation writes raw per-task summaries and traces to the owner output directories. The research MCP uses the development partition for every candidate evaluation. For debugging, `worldsmith.benchmark.evaluator` accepts any catalog task with an explicit seed and episode count. Historical replays can still use `--heldout-seed-file` with the owner group CLI or `--split heldout` with the Pymunk CLI.

## Task and simulator boundary

`benchmark/simulation.py` advances the bodies and produces observations. `benchmark/tasks/` defines the `Task` interface. Its `interception.py` defines the scene, objective, time budget, scoring, and the two task-specific reference controllers. The evaluator runs the episode and records the measurements and scores provided by the task. The CLI exposes all five tasks independently or together with `--task all`. Navigation scores destination arrival without obstacle contact; rendezvous scores 20 consecutive steps within a tracking radius. Bouncing interception uses a dynamic Pymunk circle and static arena walls with sampled restitution. It exposes the wall midpoints through the existing Observation contract. Object pushing uses a shaped kinematic robot and a dynamic circle, with Pymunk space damping modeling linear drag. Owner traces record per-step collision impulse, object speed, and distance to the destination.

In traces, positions are under `observation.entities` and the controller's machine-readable task identifier is under `observation.goal`. The summary's `objective` field contains the human-readable `task.objective` (a mapping of task identifiers to objectives for combined runs). Read the [public task brief](autoresearch/TASKS.md) for success criteria, limits, and metrics. See the [World API](autoresearch/API.md) for the controller contract.

Existing scenes use `Entity` for shape-free kinematic motion. Wrap a Pymunk body that is outside any space in a `PhysicalEntity`. Shape-free bodies are valid. For contact interactions, attach a `pymunk.Circle` or `pymunk.Segment` before constructing the simulation. A dynamic body responds to forces, impulses, and collisions according to its mass and moment of inertia. Static bodies provide fixed collision geometry. A shaped kinematic body can move dynamic objects while keeping its prescribed velocity. Set elasticity and contact friction on both shapes in a collision pair. Scene code can keep a body reference to apply Pymunk forces each step or apply an impulse once.

Physical entities are hidden by default. Set `observable=True` to include a position measurement. `position_noise` sets the standard deviation, with the default `0.0` giving a perfect measurement. Owner traces include the physical body's type, pose, velocities, dynamic mass and moment, and shape materials. An infinite moment for locked rotation is recorded as `null`. `Action(heading, throttle)` still sets the velocity of the observed kinematic robot. The simulator takes one discrete physics step every 0.1 seconds, so scene authors should choose speeds and shape sizes that keep bodies from tunneling through contacts.

## Track a run

Add `--wandb offline` to record the seed selection and metrics locally with W&B. Use `--wandb online` with an authenticated W&B account to send metrics to the `worldsmith` project. W&B is disabled by default. Summary and trace files are written locally in every mode and can be uploaded once the existing R2 utilities are integrated.
