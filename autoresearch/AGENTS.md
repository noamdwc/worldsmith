# Worldsmith V0 AutoResearch

Improve the controller through simulation experiments. Look for reusable representations, estimators, models, or skills that explain a failure and help across development episodes.

## Workspace rules

- Read [TASKS.md](TASKS.md) for the public task brief and [API.md](API.md) for the World API. Use evaluator stdout and development run files for feedback.
- Keep controller code, reusable skills, and experiment notes under `autoresearch/`. Import the public World API and local research code from controllers. Keep generated run files under `autoresearch/runs/`.
- Do not inspect or modify simulator internals, benchmark task implementations, scoring logic, reference controller source, or tests. Treat the evaluator as a CLI. Use the development split during research and leave held-out evaluation to the project owner.
- Base actions on the public observations and episode memory. Use traces for offline diagnosis. Do not read traces, scenario seeds, or evaluator state from a running controller.

## Evaluation workflow

In V0, write a Python controller in [controller.py](controller.py) that runs for the entire episode. There is no runtime LLM.

Run from inside `autoresearch/`. Start with the supplied controller, then use a new run name for each candidate:

```sh
./worldsmith-eval starter-dev
```

1. Record the starter result, then inspect failed episodes in `runs/starter-dev/episodes.jsonl`. Each row contains one episode with the initial observation, actions, resulting observations, and distance feedback.
2. Pick a concrete failure mode and write down a hypothesis. Make a small code change, extracting a reusable function when the logic applies beyond this controller.
3. Run the same development split again. Compare captures and mean closest distance in `summary.json`, which is also printed on stdout. Check which seeds improved or regressed and record the result before iterating.

To replay a development episode, run `./worldsmith-eval replay-12 --seed 12` with a seed from a development trace.
