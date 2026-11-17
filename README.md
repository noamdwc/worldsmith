# Worldsmith

Worldsmith tests whether a coding agent can improve a shared Python controller by developing reusable representations and skills through simulation experiments. The controller receives the task and observations at runtime and acts without an LLM. The owner inventory describes the 20 tasks across Pymunk, ManiSkill, LIBERO-PRO, and Robosuite. [Benchmark inventory](BENCHMARKS.md)

## Setup and local checks

From the repository root, install the pinned Pymunk environment and run the fast tests.

```sh
uv sync --locked
uv run --locked python -m pytest
```

The other simulators use separate environments. Follow their setup guides to install the runtime and run its opt-in tests. [ManiSkill](simulation_environments/maniskill/README.md) · [LIBERO-PRO](simulation_environments/libero/README.md) · [Robosuite](simulation_environments/robosuite/README.md)

For single-task debugging, use the task's pinned interpreter and an explicit seed and episode count.

```sh
uv run --locked python -m worldsmith.benchmark.evaluator \
  --task navigation --episodes 3 --seed 42 --output runs/navigation
```

Supply a different zero-argument controller factory with `--controller module:factory`. Use a new output directory for each run. The evaluator writes `summary.json` and `episodes.jsonl`. Pymunk and ManiSkill default to one worker, and LIBERO-PRO and Robosuite default to two. Use `--workers N` for owner commands or set `WORLDSMITH_EVALUATION_WORKERS` before launching AutoResearch. Parallel workers require an importable factory and write episodes in seed order.

## AutoResearch

Run `make autoresearch` from the repository root after configuring Google Colab and Cloudflare R2. The launcher starts a Colab L4 runtime before starting Codex, reuses it for the session's evaluations, and stops it when the research process exits. Use `make autoresearch-maniskill`, `make autoresearch-libero`, or `make autoresearch-robosuite` for one simulator. [Colab setup and verification](COLAB.md)

The launcher starts from the clean starter, runs the research session, archives it under `research-history/`, and restores the starter after success. If Codex exits with an error, the archive and workspace are both kept. After closing a session and waiting for evaluations to finish, use `make clear-autoresearch` to archive and reset the workspace manually.

### Manual sessions

Run these commands from the repository root for a local debugging session. You can also use `WORLDSMITH_EVALUATION_BACKEND=local make autoresearch` with the local simulators installed. The server uses the root interpreter for Pymunk and defaults to `simulation_environments/<simulator>/.venv/bin/python` for the other simulators. Set `WORLDSMITH_MANISKILL_PYTHON`, `WORLDSMITH_LIBERO_PYTHON`, or `WORLDSMITH_ROBOSUITE_PYTHON` to an absolute path to override it.

For a single-task session, set `WORLDSMITH_RESEARCH_SUITE` to a development task ID before launching Codex, for example `export WORLDSMITH_RESEARCH_SUITE=navigation`. Close the session and wait for evaluations to finish before archiving and resetting.

```sh
uv run --locked python -m worldsmith.research_lifecycle start
# Use the same low reasoning effort as make autoresearch.
(cd autoresearch && codex -c 'model_reasoning_effort="low"' "Run the three-iteration AutoResearch session in RESEARCH.md.")
archive_id=$(uv run --locked python -m worldsmith.research_lifecycle archive)
uv run --locked python -m worldsmith.research_lifecycle reset "$archive_id"
```

### Evaluation results

`evaluate_controller()` uses development seeds 0–7 for each active task. The full `worldsmith-all-v5` suite covers 16 tasks and 128 episodes. The aggregate is the equal-weight mean of the task success rates. Each evaluation writes two views under the same run ID.

- `autoresearch/runs/<run-id>/` contains a code copy, status, and filtered results. The research archive preserves this view.
- `runs/owner/<run-id>/` contains the executed code snapshot, raw traces organized by simulator and task, and diagnostics. These files survive a workspace reset.

The tool and saved research results expose development feedback. The launcher and evaluator run under the same OS account, so access to other repository files also relies on the workspace instructions. [Research workflow](autoresearch/RESEARCH.md) · [Controller API](autoresearch/API.md)

## Owner validation and replay

From the repository root, run final validation against the retained controller snapshot from development. It evaluates all 20 tasks on seeds 8–15. Use the shared tasks to assess seed generalization and the validation-only tasks to assess task generalization. [Owner task reference](BENCHMARKS.md#owner-reference-for-validation-only-tasks)

The development and validation task separation applies to sessions after PR21. Earlier research sessions exposed all four validation-only tasks to the researcher.

```sh
repo=$PWD
selected_run=mcp-... # Run ID of the retained controller
output="$repo/runs/validation/$selected_run"
(
  cd "$repo/runs/owner/$selected_run/code" || exit 1
  export PYTHONPATH="$repo"
  "$repo/.venv/bin/python" -B -m worldsmith.evaluation.owner_evaluation pymunk "$output/pymunk" --validation || exit 1
  for group in maniskill libero robosuite; do
    "$repo/simulation_environments/$group/.venv/bin/python" -B -m worldsmith.evaluation.owner_evaluation "$group" "$output/$group" --validation || exit 1
  done
)
```

For a historical owner seed-file replay, replace `--validation` with `--heldout-seed-file /absolute/path/to/private-seeds.json` and use a fresh output directory. The file maps task IDs to the first seed in each distinct eight-seed range. Pymunk and ManiSkill use a combined file. Pre-expansion files replay their original task sets.

### V1 Pymunk artifacts

Use the `v2-pre-cleanup` tag to reproduce the original Chase and Lead evaluations, the fixed V1 seed splits, and the legacy Pymunk summary and trace format.

```sh
git worktree add --detach ../worldsmith-v1-replay v2-pre-cleanup
cd ../worldsmith-v1-replay
uv sync --locked
make chase
make lead
uv run --locked python -m worldsmith evaluate --task all --output runs/combined-dev
uv run --locked python -m worldsmith evaluate --task all --split heldout --output runs/combined-heldout
uv run --locked python -m worldsmith evaluate --task interception --split heldout --output runs/interception-heldout
```
