# Worldsmith AutoResearch

Improve the controller through simulation experiments. Look for reusable representations, estimators, models, or skills that explain a failure and help across development episodes.

## Workspace rules

- Read [TASKS.md](TASKS.md) for the public task brief and [API.md](API.md) for the World API. Use the `worldsmith` MCP tools and development run files for feedback.
- Keep controller code, reusable skills, and experiment notes under `autoresearch/`. Import the public World API and local research code from controllers. Keep generated run files under `autoresearch/runs/`.
- Use the public task descriptions, MCP tools, and saved development feedback for evaluation. Leave simulator internals, task implementations, scoring logic, reference controller source, tests, owner diagnostics, and held-out evaluations to the project owner.
- Base actions on the public observations and episode memory. Use traces for offline diagnosis. Do not read traces, scenario seeds, or evaluator state from a running controller.

## Evaluation workflow

Write one shared goal-aware Python controller in [controller.py](controller.py) that runs for the entire episode. There is no runtime LLM.

Follow [RESEARCH.md](RESEARCH.md) for the evaluation budget and keep-or-reject rules. Record every candidate and the final findings in [JOURNAL.md](JOURNAL.md). `evaluate_controller()` snapshots the current code and runs the combined five-task development benchmark. It returns per-task metrics and an aggregate score, and saves `summary.json` and `episodes.jsonl` in the returned results directory. The traces contain the complete observations, actions, and episode outcomes. Failed evaluations report a run ID and keep their snapshot and status under `runs/`. `get_episode_trace(seed, task="interception")` returns a task episode from the latest successful run in the current MCP session. Use `task="navigation"`, `task="rendezvous"`, `task="bouncing_interception"`, or `task="object_pushing"` for the other tasks. Leave session metadata and saved evaluations intact for the owner to archive.
