# Worldsmith V0 AutoResearch

Improve the controller through simulation experiments. Look for reusable representations, estimators, models, or skills that explain a failure and help across development episodes.

## Workspace rules

- Read [TASKS.md](TASKS.md) for the public task brief and [API.md](API.md) for the World API. Use the `worldsmith` MCP tools and development run files for feedback.
- Keep controller code, reusable skills, and experiment notes under `autoresearch/`. Import the public World API and local research code from controllers. Keep generated run files under `autoresearch/runs/`.
- Do not inspect or modify simulator internals, benchmark task implementations, scoring logic, reference controller source, or tests. Use the MCP tools for evaluations and leave held-out evaluation to the project owner.
- Base actions on the public observations and episode memory. Use traces for offline diagnosis. Do not read traces, scenario seeds, or evaluator state from a running controller.

## Evaluation workflow

In V0, write a Python controller in [controller.py](controller.py) that runs for the entire episode. There is no runtime LLM.

Follow [RESEARCH.md](RESEARCH.md) for the evaluation budget and keep-or-reject rules. Record every candidate and the final findings in [JOURNAL.md](JOURNAL.md). `evaluate_controller()` saves `summary.json` and `episodes.jsonl`. `get_episode_trace(seed)` reads the latest successful run in the current MCP session.
