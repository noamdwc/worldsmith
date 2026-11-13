"""Adapt V1 Pymunk scenes and task scoring to the common evaluation protocol."""

from contextlib import nullcontext
from typing import TextIO

from worldsmith.api import INTERCEPTION_GOAL, Action
from worldsmith.benchmark.results import TaskEvaluation, write_episode
from worldsmith.benchmark.runner import InvalidAction
from worldsmith.benchmark.tasks.constants import BENCHMARKS, TASKS
from worldsmith.benchmark.tasks.interception import Chase
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
        return {"heading": action.heading, "throttle": action.throttle}

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


def write_research_episode(trace: TextIO, episode: dict, metrics: dict) -> None:
    # Research frames pair the action with its resulting observation and state.
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


def summarize_research(results: dict[str, TaskEvaluation], *, combined: bool) -> dict:
    """Report native metrics or score the combined benchmark across all tasks."""
    tasks = {
        name: {"episodes": len(result.episodes), **result.metrics}
        for name, result in results.items()
    }
    if not combined:
        (metrics,) = tasks.values()
        return metrics

    for metrics in tasks.values():
        if "capture_rate" in metrics:
            metrics["success_rate"] = metrics["capture_rate"]
    return {
        "episodes": sum(metrics["episodes"] for metrics in tasks.values()),
        "aggregate_score": sum(metrics["success_rate"] for metrics in tasks.values()) / len(tasks),
        "tasks": tasks,
    }


def evaluation_setup(task_name: str):
    def open_adapter():
        return nullcontext(PymunkAdapter(TASKS[task_name]()))

    if task_name == INTERCEPTION_GOAL:
        return open_adapter, Chase

    from autoresearch.controller import make_controller

    return open_adapter, make_controller
