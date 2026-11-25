"""Run one SLM-controlled task against an already-running inference server."""

import argparse
import json
import traceback
from pathlib import Path

from worldsmith.benchmark.results import snapshot
from worldsmith.benchmark.task_catalog import TASK_CATALOG, TASK_GROUPS
from worldsmith.inference.client import create_client
from worldsmith.inference.config import MODEL, MODEL_REVISION
from worldsmith.harness.controller import load_tokenizer, run_slm_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_CATALOG, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "task": args.task, "seed": args.seed, "action_repeat": args.action_repeat,
        "model": MODEL, "model_revision": MODEL_REVISION,
    }
    requests = 0
    try:
        tokenizer = load_tokenizer()
        if args.task in TASK_GROUPS["pymunk"]:
            # The legacy catalog setup also loads an AutoResearch controller,
            # which the frozen owner wheel does not include or use.
            from worldsmith.benchmark.adapters.pymunk_adapter import open_pymunk

            context = open_pymunk(args.task)
        else:
            open_adapter, _ = TASK_CATALOG[args.task]()
            context = open_adapter()
        with create_client() as client, context as adapter:
            create = client.chat.completions.create

            def counted_create(**kwargs):
                nonlocal requests
                requests += 1
                return create(**kwargs)

            client.chat.completions.create = counted_create
            episode, decisions = run_slm_episode(
                adapter, client, tokenizer, args.seed, task=args.task, action_repeat=args.action_repeat,
            )
            result.update(
                status="completed", environment_steps=episode["length"], slm_decisions=decisions,
                terminated=episode["steps"][-1]["terminated"],
                truncated=episode["steps"][-1]["truncated"], **snapshot(adapter.score(episode)),
            )
    except Exception as error:
        # Preserve a failed task as an explicit result so the suite can continue.
        traceback.print_exc()
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    result["model_requests"] = requests
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
