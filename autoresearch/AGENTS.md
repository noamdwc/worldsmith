# Worldsmith AutoResearch

Improve the controller through simulation experiments. Look for reusable representations, estimators, models, or skills that explain a failure and help across development episodes.

## Workspace rules

- Read [TASKS.md](TASKS.md) for the public task brief and [API.md](API.md) for the World API. Use the `worldsmith` MCP tools and development run files for feedback.
- The current working directory is the AutoResearch workspace. Import the public World API and local research code from controllers. Place controller helpers next to `controller.py`. The evaluator snapshots the workspace as the `autoresearch` package, so a sibling file named `drawer.py` is importable as `autoresearch.drawer`. Keep reusable skills and experiment notes in this workspace. Put generated run files and temporary controller backups under `runs/`, which is excluded from code snapshots.
- Use the public task descriptions, MCP tools, and saved development feedback for evaluation. Leave simulator internals, task implementations, scoring logic, reference controller source, tests, owner diagnostics, and held-out evaluations to the project owner.
- Base actions on the public observations and episode memory. Use traces for offline diagnosis. Do not read traces, scenario seeds, or evaluator state from a running controller.

## Evaluation workflow

Write one shared task-aware Python controller in [controller.py](controller.py) that runs for the entire episode. There is no runtime LLM.

Follow [RESEARCH.md](RESEARCH.md) for the evaluation budget and keep-or-reject rules. Record every candidate and the final findings in [JOURNAL.md](JOURNAL.md). `evaluate_controller()` snapshots the current code and evaluates the active development suite. It returns aggregate and per-task results and saves `summary.json` and `episodes.jsonl` in the returned directory. Use `get_episode_trace(seed, task)` with a task ID from the baseline summary to inspect a public trace from the latest successful evaluation. Failed evaluations report a run ID and keep their snapshot and status under `runs/`. Leave the session metadata and saved evaluations intact for the owner to archive.
