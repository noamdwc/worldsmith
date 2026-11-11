"""Run controllers against task-defined scenes and write episode traces."""

import json
from dataclasses import asdict
from typing import TextIO

from worldsmith.api import ControllerFactory
from worldsmith.benchmark.simulation import InvalidAction
from worldsmith.benchmark.tasks.constants import TASKS
from worldsmith.benchmark.tasks.task import Task


class ControllerError(RuntimeError):
    """An error at the controller boundary that can be reported to the researcher."""


def evaluate(task: Task, factory: ControllerFactory, seeds: list[int], trace: TextIO) -> dict:
    episodes = []
    for seed in seeds:
        world = task.start(seed)
        try:
            controller = factory()
        except Exception as exc:
            raise ControllerError(f"Controller setup: {type(exc).__name__}: {exc}") from exc
        observation = world.initial_observation
        initial_observation = asdict(observation)
        initial_state = world.ground_truth()
        measurements = [task.measure(world)]
        frames = []
        success = False

        for step in range(task.step_limit):
            try:
                action = controller(observation)
            except Exception as exc:
                raise ControllerError(
                    f"{observation.goal}, seed {seed}, step {step + 1}: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                observation = world.step(action)
            except InvalidAction as exc:
                raise ControllerError(
                    f"{observation.goal}, seed {seed}, step {step + 1}: {exc}"
                ) from exc
            measurement = task.measure(world)
            measurements.append(measurement)
            frames.append({
                "action": {"heading": action.heading, "throttle": action.throttle},
                "observation": asdict(observation),
                "state": world.ground_truth(),
                **measurement,
            })
            success = task.succeeded(measurements)
            if success:
                break

        metrics = task.episode_metrics(measurements, success)
        episodes.append(metrics)
        trace.write(json.dumps({
            "seed": seed,
            **metrics,
            "steps": len(frames),
            "initial_observation": initial_observation,
            "initial_state": initial_state,
            "initial_measurements": measurements[0],
            "frames": frames,
        }) + "\n")

    return {"episodes": len(episodes), **task.summarize(episodes)}


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
