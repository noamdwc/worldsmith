"""Run controllers against task-defined scenes and write episode traces."""

import json
from dataclasses import asdict
from typing import TextIO

from worldsmith.api import INTERCEPTION_GOAL, ControllerFactory
from worldsmith.benchmark.tasks import TASKS
from worldsmith.benchmark.tasks.task import Task


def evaluate(task: Task, factory: ControllerFactory, seeds: list[int], trace: TextIO) -> dict:
    episodes = []
    for seed in seeds:
        world = task.start(seed)
        controller = factory()
        observation = world.initial_observation
        initial_observation = asdict(observation)
        measurements = [task.measure(world)]
        frames = []
        success = False

        for _ in range(task.step_limit):
            action = controller(observation)
            observation = world.step(action)
            measurement = task.measure(world)
            measurements.append(measurement)
            frames.append({
                "action": asdict(action),
                "observation": asdict(observation),
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
            "frames": frames,
        }) + "\n")

    return {"episodes": len(episodes), **task.summarize(episodes)}


def evaluate_all(
    factory: ControllerFactory, seeds: list[int], trace: TextIO,
) -> dict:
    """Evaluate one shared factory on the same fixed seeds for all three tasks."""
    tasks = {}
    for name, task_factory in TASKS.items():
        metrics = evaluate(task_factory(), factory, seeds, trace)
        if name == INTERCEPTION_GOAL:
            metrics["success_rate"] = metrics["capture_rate"]
        tasks[name] = metrics
    return {
        "episodes": sum(metrics["episodes"] for metrics in tasks.values()),
        "aggregate_score": sum(metrics["success_rate"] for metrics in tasks.values()) / len(tasks),
        "tasks": tasks,
    }
