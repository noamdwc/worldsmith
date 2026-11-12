"""Adapt V1 Pymunk scenes and task scoring to the common evaluation protocol."""

from contextlib import nullcontext

from worldsmith.api import INTERCEPTION_GOAL, Action, Controller, ControllerFactory
from worldsmith.benchmark.simulation import InvalidAction
from worldsmith.benchmark.tasks.constants import BENCHMARKS, TASKS
from worldsmith.benchmark.tasks.task import Task


class ControllerError(RuntimeError):
    """An error at the controller boundary that can be reported to the researcher."""


class PymunkAdapter:
    def __init__(self, task: Task) -> None:
        self.task = task

    def reset(self, seed: int):
        self.seed = seed
        self.world = self.task.start(seed)
        observation = self.world.initial_observation
        self.measurements = [self.task.measure(self.world)]
        self.goal = observation.goal
        return observation, self._info()

    def make_controller(self, factory: ControllerFactory) -> Controller:
        try:
            controller = factory()
        except Exception as exc:
            raise ControllerError(f"Controller setup: {type(exc).__name__}: {exc}") from exc

        def act(observation):
            try:
                return controller(observation)
            except Exception as exc:
                raise ControllerError(
                    f"{self.goal}, seed {self.seed}, step {len(self.measurements)}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        return act

    def encode_action(self, action: Action) -> dict:
        return {"heading": action.heading, "throttle": action.throttle}

    def step(self, action: Action):
        try:
            observation = self.world.step(action)
        except InvalidAction as exc:
            raise ControllerError(
                f"{self.goal}, seed {self.seed}, step {len(self.measurements)}: {exc}"
            ) from exc
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


def summarize_combined(tasks: dict[str, dict]) -> dict:
    """Score the V1 benchmark as the equal-weight mean of task success rates."""
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
        from worldsmith.benchmark.tasks.interception import Chase

        return open_adapter, Chase

    from autoresearch.controller import make_controller

    return open_adapter, make_controller
