"""Preserve the V1 research API and trace format using the shared evaluator."""

from functools import partial
from typing import TextIO

from worldsmith.api import ControllerFactory
from worldsmith.benchmark.evaluator import evaluate_episodes
from worldsmith.benchmark.pymunk_adapter import ControllerError as ControllerError
from worldsmith.benchmark.pymunk_adapter import PymunkAdapter
from worldsmith.benchmark.results import write_episode
from worldsmith.benchmark.tasks.constants import TASKS
from worldsmith.benchmark.tasks.task import Task


def evaluate(task: Task, factory: ControllerFactory, seeds: list[int], trace: TextIO) -> dict:
    adapter = PymunkAdapter(task)

    def record(episode: dict, metrics: dict) -> None:
        # V1 frames pair the action with its resulting observation and state.
        steps = episode["steps"]
        observations = [step["observation"] for step in steps[1:]]
        observations.append(episode["final_observation"])
        initial = episode["reset_info"]
        write_episode(trace, {
            "seed": episode["seed"],
            **metrics,
            "steps": episode["length"],
            "initial_observation": steps[0]["observation"],
            "initial_state": initial["state"],
            "initial_measurements": initial["measurements"],
            "frames": [
                {
                    "action": step["action"],
                    "observation": observation,
                    "state": step["info"]["state"],
                    **step["info"]["measurements"],
                }
                for step, observation in zip(steps, observations, strict=True)
            ],
        }, legacy=True)

    episodes, summary = evaluate_episodes(adapter, partial(adapter.make_controller, factory), seeds, record)
    return {"episodes": len(episodes), **summary}


def evaluate_all(
    controller_factory: ControllerFactory, seeds: list[int], trace: TextIO,
) -> dict:
    """Evaluate one shared factory on the same fixed seeds for all tasks."""
    tasks = {}
    for name, task_factory in TASKS.items():
        metrics = evaluate(task_factory(), controller_factory, seeds, trace)
        if "capture_rate" in metrics:
            metrics["success_rate"] = metrics["capture_rate"]
        tasks[name] = metrics
    return {
        "episodes": sum(metrics["episodes"] for metrics in tasks.values()),
        "aggregate_score": sum(metrics["success_rate"] for metrics in tasks.values()) / len(tasks),
        "tasks": tasks,
    }
