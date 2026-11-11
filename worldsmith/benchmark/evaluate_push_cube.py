"""Run seeded PushCube-v1 episodes and save their controller traces."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import gymnasium as gym
import numpy as np
import torch

from worldsmith.benchmark.tasks.push_cube import (
    ACTION_MODE,
    OBS_MODE,
    ROBOT,
    PushController,
    make_env,
)


def json_state(value):
    """Snapshot ManiSkill's nested state without retaining simulator buffers."""
    if isinstance(value, dict):
        return {key: json_state(item) for key, item in value.items()}
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        return value.tolist()
    return value


def run_episode(env, seed: int) -> dict:
    controller = PushController()
    observation, reset_info = env.reset(seed=seed)
    reset_info = json_state(reset_info)
    steps = []
    while True:
        controller_observation = json_state(observation)
        action = controller.act(observation)
        if not env.action_space.contains(action):
            raise ValueError(f"Controller action is outside {env.action_space}: {action}")
        controller_action = action.tolist()

        observation, reward, terminated, truncated, info = env.step(action)
        terminated = bool(terminated)
        truncated = bool(truncated)
        steps.append({
            "observation": controller_observation,
            "action": controller_action,
            "reward": reward.item(),
            "terminated": terminated,
            "truncated": truncated,
            "info": json_state(info),
        })
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "length": len(steps),
        "success": bool(info["success"]),
        "reset_info": reset_info,
        "steps": steps,
        "final_observation": json_state(observation),
    }


def evaluate(episodes: int, seed: int, output: Path) -> dict:
    # ManiSkill seeds NumPy's RandomState at reset.
    if episodes < 1 or seed < 0 or seed + episodes > 2**32:
        raise ValueError("Use a positive episode count and seeds in [0, 2**32).")

    # GLFW can change the process working directory during macOS initialization.
    output = output.resolve()
    output.mkdir(parents=True)
    env = make_env()
    try:
        task = gym.spec(env.spec.id)
        backend = env.unwrapped.backend
        summary = {
            "task_id": task.id,
            "task_version": task.version,
            "max_episode_steps": task.max_episode_steps,
            "obs_mode": OBS_MODE,
            "action_mode": ACTION_MODE,
            "sim_backend": backend.sim_backend,
            "render_backend": backend.render_backend,
            "render_mode": env.render_mode,
            "robot": ROBOT,
            "controller": f"{PushController.__module__}.{PushController.__name__}",
            "versions": {
                name: version(name)
                for name in ("mani-skill-nightly", "sapien", "pin", "gymnasium", "numpy", "torch")
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seeds": list(range(seed, seed + episodes)),
            "episodes": [],
        }
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as traces:
            for episode_seed in summary["seeds"]:
                episode = run_episode(env, episode_seed)
                traces.write(json.dumps(episode, allow_nan=False, separators=(",", ":")) + "\n")
                summary["episodes"].append({
                    key: episode[key] for key in ("seed", "length", "success")
                })
    finally:
        env.close()

    summary["success_rate"] = sum(row["success"] for row in summary["episodes"]) / episodes
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="First seed in a consecutive sequence")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.episodes, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
