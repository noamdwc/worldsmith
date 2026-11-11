# AutoResearch journal

Append an entry for each session and keep the entries for rejected candidates. Use paths relative to this workspace so results can be found in `runs/`.

## Session template

- **Date and baseline** — shared controller, run directory, aggregate score, and the per-task table below. The combined development benchmark has 8 episodes per task.
- **Iteration** — number, hypothesis, evidence from a trace identified by task and seed, and focused code change. Note any expected benefit across tasks.
- **Result and decision** — run directory and aggregate score or error; report each task’s success-rate delta from the retained controller, secondary metrics, and keep/reject decision. Highlight regressions even when aggregate score improves.
- **Findings** — useful discoveries, remaining uncertainty, and baseline versus final aggregate and per-task results. Label these as development results.

| Task | Successes / 8 | Success rate | Change vs retained | Secondary metrics |
|---|---|---|---|---|
| Interception | | | | Mean closest distance |
| Navigation | | | | Collision rate, mean final destination distance, mean minimum clearance |
| Rendezvous | | | | Mean longest tracking seconds, mean distance, mean final distance |
| Bouncing interception | | | | Mean closest distance |

The aggregate score is the mean of the four success rates. Keep only an aggregate improvement; reject ties. Preserve entries and artifacts for each attempted candidate iteration, including failed evaluations.
