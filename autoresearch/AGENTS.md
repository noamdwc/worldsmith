# Worldsmith V0 AutoResearch

Improve the controller through simulation experiments. Look for reusable representations, estimators, models, or skills that explain a failure and help across development episodes.

## Workspace rules

- Read [TASKS.md](TASKS.md) for the public task brief and [API.md](API.md) for the World API. Use the `worldsmith` MCP tools and development run files for feedback.
- Keep controller code, reusable skills, and experiment notes under `autoresearch/`. Import the public World API and local research code from controllers. Keep generated run files under `autoresearch/runs/`.
- Do not inspect or modify simulator internals, benchmark task implementations, scoring logic, reference controller source, or tests. Use the MCP tools for evaluations and leave held-out evaluation to the project owner.
- Base actions on the public observations and episode memory. Use traces for offline diagnosis. Do not read traces, scenario seeds, or evaluator state from a running controller.

## Evaluation workflow

In V0, write a Python controller in [controller.py](controller.py) that runs for the entire episode. There is no runtime LLM.

Start Codex from inside `autoresearch/` so it loads the workspace MCP config.

1. Call `evaluate_controller()` and record the starter metrics. The tool returns the run directory, which contains `summary.json` and `episodes.jsonl`. Find a failed seed in the run file and call `get_episode_trace(seed)` to inspect its observations, actions, and distance feedback.
2. Pick a concrete failure mode and write down a hypothesis. Make a small code change, extracting a reusable function when the logic applies beyond this controller.
3. Call `evaluate_controller()` again. Compare captures and mean closest distance, check which seeds improved or regressed, and record the result before iterating.

`get_episode_trace(seed)` reads the latest successful evaluation in the current MCP session. Earlier runs remain in `runs/`.
