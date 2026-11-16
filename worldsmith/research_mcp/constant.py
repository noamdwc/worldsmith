"""AutoResearch suite definitions, deadlines, and launch settings."""

import os
from pathlib import Path

from worldsmith.benchmark.evaluator import worker_count
from worldsmith.benchmark.task_catalog import DEV_TASK_GROUPS
from worldsmith.research_lifecycle import ROOT, WORKSPACE as DEFAULT_WORKSPACE

EVALUATION_TIMEOUT_SECONDS = 90
SUITE_TIMEOUT_SECONDS = {"all": 360, "libero": 300, "robosuite": 120}
SUITES = {
    "all": ("worldsmith-all-v5", tuple(DEV_TASK_GROUPS)),
    "maniskill": ("worldsmith-maniskill-v3", ("maniskill",)),
    "libero": ("worldsmith-arm-position-v2", ("libero",)),
    "robosuite": ("worldsmith-arm-skills-v3", ("robosuite",)),
}
SUITES.update({
    task: (f"worldsmith-task-{task}", (group,))
    for group, tasks in DEV_TASK_GROUPS.items()
    for task in tasks
})
OWNER_RUNS = ROOT / "runs" / "owner"
WORKSPACE = Path(os.environ.get("WORLDSMITH_RESEARCH_WORKSPACE", DEFAULT_WORKSPACE)).absolute()
SUITE = os.environ.get("WORLDSMITH_RESEARCH_SUITE", "all")
if SUITE not in SUITES:
    raise ValueError(f"WORLDSMITH_RESEARCH_SUITE must be one of {', '.join(SUITES)}")
_worker_override = os.environ.get("WORLDSMITH_EVALUATION_WORKERS")
WORKER_OVERRIDE = None if _worker_override is None else worker_count(_worker_override)
