# Worldsmith V0 research run

Improve the interception controller using noisy observations and reusable Python skills. Use only the public files in this workspace and the `worldsmith` MCP tools for evaluation. [AGENTS.md](AGENTS.md) defines the workspace rules.

## Budget and process

1. Call `evaluate_controller()` once before editing to establish a baseline. Record the metrics and run directory in [JOURNAL.md](JOURNAL.md). Stop and report the error if the baseline fails.
2. Run up to **three candidate iterations**, with one evaluation per iteration. Use `get_episode_trace(seed)` or the saved development traces to identify a failure, write a hypothesis, and make one focused change. Trace lookups do not use the evaluation budget. A failed or timed-out candidate evaluation uses one iteration.
3. Compare each candidate with the best retained controller on the same development split. Keep it if it captures more targets, or if captures tie and mean closest distance decreases. Otherwise, undo only that iteration’s code edits. Keep its journal entry and run artifacts. A controller error or timeout rejects the candidate. The evaluator has a 30-second limit, and trace lookup continues to use the latest successful run.
4. End with the best retained controller and summarize the baseline and final metrics, what you tried, what you learned, and what remains uncertain. The scores are from development episodes.

## Start a session

From the `autoresearch/` directory, with `uv` available and the Codex project trusted, start an interactive session:

```sh
codex "Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings."
```

For a noninteractive session from the same directory:

```sh
codex exec --sandbox workspace-write "Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings."
```

Use a fresh session for the run. The local config requires the `worldsmith` MCP server to start successfully. See the [official Codex noninteractive guide](https://learn.chatgpt.com/docs/non-interactive-mode) for launch options.
