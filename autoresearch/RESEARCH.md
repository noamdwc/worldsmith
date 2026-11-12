# Worldsmith research run

Improve one shared task-aware controller across the tasks in the active development suite using their public observations and reusable Python strategies. Read [TASKS.md](TASKS.md); task descriptions and detailed development traces are available to you. Use only the public files in this workspace and the `worldsmith` MCP tools for evaluation. [AGENTS.md](AGENTS.md) defines the workspace rules.

`make autoresearch` uses `worldsmith-all-v1` (11 tasks, 88 episodes). `make autoresearch-maniskill` uses `worldsmith-maniskill-v1` (six ManiSkill tasks, 48 episodes). Use the benchmark ID and task list returned by the baseline for the whole session.

## Budget and process

1. Call `evaluate_controller()` once before editing to establish a baseline. Record the aggregate score, every evaluated task’s success rate and secondary metrics, and run directory in [JOURNAL.md](JOURNAL.md). Stop and report the error if the baseline fails.
2. Run up to **three candidate iterations**, with one evaluation per iteration. Use `get_episode_trace(seed, task)` with a task ID from the active suite or the saved development traces to identify a failure, write a hypothesis, and make one focused change. Trace lookups do not use the evaluation budget. A failed or timed-out candidate evaluation uses one iteration.
3. Compare each candidate with the best retained controller on the same development split. Keep it if the aggregate score (the equal-weight average of the evaluated task success rates) increases; otherwise reject it, including ties, and undo only that iteration’s code edits. Record each task’s success-rate change and secondary metrics, including any regressions hidden by an aggregate improvement. Keep its journal entry and run artifacts. A controller error or timeout rejects the candidate. The evaluator has a 90-second evaluation limit, and trace lookup continues to use the latest successful run.
4. End with the best retained controller and summarize the baseline and final metrics, what you tried, what you learned, and what remains uncertain. The scores are from the same 8 fixed development seeds per task. Do not use held-out episodes during research. Secondary metrics support diagnosis; they do not change the keep-or-reject rule.

## Start a session

From the `autoresearch/` directory, after completing the [owner setup](../README.md#autoresearch-workspace) and trusting the Codex project, start an interactive session:

```sh
codex "Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings."
```

For a noninteractive session from the same directory:

```sh
codex exec --sandbox workspace-write "Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings."
```

For a manual ManiSkill-only session, set `WORLDSMITH_RESEARCH_SUITE=maniskill` in the launch environment and specify the ManiSkill-only suite in the prompt. Use a fresh session for each run. The trusted workspace config requires the `worldsmith` MCP server to start successfully and preapproves its two tools so they can run without an interactive approval prompt. See the [official Codex noninteractive guide](https://learn.chatgpt.com/docs/non-interactive-mode) for launch options.
