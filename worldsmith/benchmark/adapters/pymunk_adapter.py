"""Adapt V1 Pymunk scenes and task scoring to the common evaluation protocol."""

from contextlib import nullcontext

from worldsmith.api import Action
from worldsmith.benchmark.results import snapshot
from worldsmith.benchmark.runner import InvalidAction
from worldsmith.benchmark.tasks.constants import BENCHMARKS, TASKS
from worldsmith.benchmark.tasks.task import Task


class PymunkAdapter:
    def __init__(self, task: Task) -> None:
        self.task = task

    def reset(self, seed: int):
        self.world = self.task.start(seed)
        observation = self.world.initial_observation
        self.measurements = [self.task.measure(self.world)]
        self.goal = observation.goal
        return observation, self._info()

    def encode_action(self, action: Action) -> dict:
        if not isinstance(action, Action):
            raise InvalidAction("Controller must return an Action")
        return snapshot({"heading": action.heading, "throttle": action.throttle})

    def step(self, action: Action):
        observation = self.world.step(action)
        self.measurements.append(self.task.measure(self.world))
        success = self.task.succeeded(self.measurements)
        truncated = len(self.measurements) - 1 == self.task.step_limit and not success
        return observation, None, success, truncated, self._info()

    def _info(self) -> dict:
        return {"state": self.world.ground_truth(), "measurements": self.measurements[-1]}

    def score(self, episode: dict) -> dict:
        return self.task.episode_metrics(self.measurements, episode["steps"][-1]["terminated"])

    def summarize(self, episodes: list[dict]) -> dict:
        return self.task.summarize(episodes)

    def metadata(self) -> dict:
        return {
            "task_id": BENCHMARKS[self.goal],
            "objective": self.task.objective,
            "max_episode_steps": self.task.step_limit,
        }


def open_pymunk(task_name: str):
    return nullcontext(PymunkAdapter(TASKS[task_name]()))
