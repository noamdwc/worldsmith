"""Contract for benchmark tasks."""

from typing import Protocol

from worldsmith.benchmark.simulation import Simulation

type Measurements = list[dict[str, float]]


class Task(Protocol):
    step_limit: int

    def start(self, seed: int) -> Simulation: ...
    def measure(self, world: Simulation) -> dict[str, float]: ...
    def succeeded(self, measurements: Measurements) -> bool: ...
    def episode_metrics(self, measurements: Measurements, success: bool) -> dict: ...
    def summarize(self, episodes: list[dict]) -> dict: ...
