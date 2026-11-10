"""Score controllers against fixed interception scenarios and write episode traces."""

import json
from dataclasses import asdict
from statistics import mean
from typing import TextIO

from worldsmith.api import ControllerFactory
from worldsmith.benchmark.interception import CAPTURE_RADIUS, STEP_LIMIT, Interception

BENCHMARK = "interception-v1"
SPLITS = {"dev": range(32), "heldout": range(1000, 1032)}


def evaluate(factory: ControllerFactory, seeds: list[int], trace: TextIO) -> dict:
    captures = 0
    closest_distances = []

    for seed in seeds:
        world = Interception(seed)
        controller = factory()
        observation = world.initial_observation
        initial_observation = asdict(observation)
        closest_distance = world.distance
        frames = []

        for _ in range(STEP_LIMIT):
            action = controller(observation)
            observation = world.step(action)
            distance = world.distance
            closest_distance = min(closest_distance, distance)
            frames.append({
                "action": asdict(action),
                "observation": asdict(observation),
                "distance": distance,
            })
            if distance <= CAPTURE_RADIUS:
                break

        captured = closest_distance <= CAPTURE_RADIUS
        captures += captured
        closest_distances.append(closest_distance)
        trace.write(json.dumps({
            "seed": seed,
            "captured": captured,
            "steps": len(frames),
            "closest_distance": closest_distance,
            "initial_observation": initial_observation,
            "frames": frames,
        }) + "\n")

    return {
        "episodes": len(seeds),
        "captures": captures,
        "capture_rate": captures / len(seeds),
        "mean_closest_distance": mean(closest_distances),
    }
