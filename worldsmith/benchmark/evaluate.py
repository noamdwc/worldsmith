"""Run controllers against task-defined scenes and write episode traces."""

import json
from dataclasses import asdict
from typing import TextIO

from worldsmith.api import ControllerFactory
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
