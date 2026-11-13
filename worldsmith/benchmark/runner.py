"""Execute one episode against a simulator adapter."""

from collections.abc import Callable
from typing import Any, Protocol

from worldsmith.benchmark.results import snapshot

Controller = Callable[[Any], Any]
ControllerFactory = Callable[[], Controller]


class InvalidAction(ValueError):
    """A controller command violates the simulator's public action contract."""


class ControllerError(RuntimeError):
    """Failure at the controller boundary, with episode context."""

    def __init__(self, message: str, *, public_message: str = "Controller failed"):
        super().__init__(message)
        self.public_message = public_message


class Simulator(Protocol):
    """Supply native observations and actions and detached action records."""

    def reset(self, seed: int) -> tuple[Any, dict]: ...
    def encode_action(self, action: Any) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float | None, bool, bool, dict]: ...


def run_episode(
    simulator: Simulator, controller_factory: ControllerFactory, seed: int, *, task: str = "task",
) -> dict:
    observation, reset_info = simulator.reset(seed)
    reset_info = snapshot(reset_info)
    try:
        controller = controller_factory()
    except Exception as exc:
        raise ControllerError(
            f"{task}, seed {seed}, controller setup: {type(exc).__name__}: {exc}",
            public_message=f"{task}, seed {seed}: make_controller raised an exception",
        ) from exc
    steps = []
    while True:
        controller_observation = snapshot(observation)
        location = f"{task}, seed {seed}, step {len(steps) + 1}"
        try:
            action = controller(observation)
        except Exception as exc:
            raise ControllerError(
                f"{location}: {type(exc).__name__}: {exc}",
                public_message=f"{location}: controller raised an exception",
            ) from exc
        try:
            controller_action = simulator.encode_action(action)
            observation, reward, terminated, truncated, info = simulator.step(action)
        except InvalidAction as exc:
            raise ControllerError(
                f"{location}: {exc}", public_message=f"{location}: controller returned an invalid action",
            ) from exc
        steps.append({
            "observation": controller_observation,
            "action": controller_action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": snapshot(info),
        })
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "length": len(steps),
        "reset_info": reset_info,
        "steps": steps,
        "final_observation": snapshot(observation),
    }
