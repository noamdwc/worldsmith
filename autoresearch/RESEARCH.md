# Worldsmith research run

Improve one shared task-aware controller across the tasks in the active development suite using their public observations and reusable Python strategies. Read [TASKS.md](TASKS.md); task descriptions and detailed development traces are available to you. Use only the public files in this workspace and the `worldsmith` MCP tools for evaluation. [AGENTS.md](AGENTS.md) defines the workspace rules.

The launch selects either the full development suite (11 tasks, 88 episodes) or the tabletop manipulation suite (six tasks, 48 episodes). Use the benchmark ID and task list returned by the baseline for the whole session.

## Budget and process

1. Call `evaluate_controller()` once before editing to establish a baseline. Record the aggregate score, every evaluated task’s success rate and secondary metrics, and run directory in [JOURNAL.md](JOURNAL.md). Stop and report the error if the baseline fails.
2. Run up to **three candidate iterations**, with one evaluation per iteration. Use `get_episode_trace(seed, task)` with a task ID from the active suite or the saved development traces to identify a failure, write a hypothesis, and make one focused change. Trace lookups do not use the evaluation budget. A failed or timed-out candidate evaluation uses one iteration.
3. Compare each candidate with the best retained controller on the same development split. Keep it if the aggregate score (the equal-weight average of the evaluated task success rates) increases; otherwise reject it, including ties, and undo only that iteration’s code edits. Record each task’s success-rate change and secondary metrics, including any regressions hidden by an aggregate improvement. Keep its journal entry and run artifacts. A controller error or timeout rejects the candidate. The evaluator has a 90-second evaluation limit, and trace lookup continues to use the latest successful run.
4. End with the best retained controller and summarize the baseline and final metrics, what you tried, what you learned, and what remains uncertain. The scores are from the same 8 fixed development seeds per task. Do not use held-out episodes during research. Secondary metrics support diagnosis; they do not change the keep-or-reject rule.
