"""Execute one episode against a simulator adapter."""

from collections.abc import Callable
from typing import Any, Protocol

from worldsmith.benchmark.results import snapshot

Controller = Callable[[Any], Any]
ControllerFactory = Callable[[], Controller]


class Simulator(Protocol):
    """Supply native observations and actions and detached action records."""

    def reset(self, seed: int) -> tuple[Any, dict]: ...
    def encode_action(self, action: Any) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float | None, bool, bool, dict]: ...


def run_episode(simulator: Simulator, controller_factory: ControllerFactory, seed: int) -> dict:
    observation, reset_info = simulator.reset(seed)
    reset_info = snapshot(reset_info)
    controller = controller_factory()
    steps = []
    while True:
        controller_observation = snapshot(observation)
        action = controller(observation)
        controller_action = simulator.encode_action(action)
        observation, reward, terminated, truncated, info = simulator.step(action)
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
