"""Publish the development feedback exposed to AutoResearch."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_evaluation import CONTROLLER, DEV_SEEDS, SUITES

MANISKILL_CONFIG = (
    "task_id", "task_version", "max_episode_steps", "obs_mode", "action_mode",
    "action_shape", "reward_mode", "sim_backend", "render_backend", "render_mode",
    "robot", "reconfiguration_freq", "versions",
)
TASK_METRICS = {
    "interception": ("captures", "capture_rate", "mean_closest_distance"),
    "bouncing_interception": ("captures", "capture_rate", "mean_closest_distance"),
    "navigation": ("collision_rate", "mean_final_destination_distance", "mean_minimum_obstacle_clearance"),
    "rendezvous": ("mean_longest_tracking_seconds", "mean_distance", "mean_final_distance"),
    "object_pushing": (
        "mean_final_object_distance", "mean_minimum_object_distance", "mean_final_object_speed",
        "mean_contact_seconds",
    ),
}
EPISODE_OUTCOMES = {
    "interception": ("captured",),
    "bouncing_interception": ("captured",),
    "navigation": ("success", "collided"),
    "rendezvous": ("success",),
    "object_pushing": ("success",),
}


@dataclass(frozen=True, slots=True)
class ResearchFeedback:
    summary: dict
    episodes: list[dict]
    episode_indices: dict[tuple[int, str], int]

    def episode(self, seed: int, task: str) -> dict:
        index = self.episode_indices.get((seed, task))
        if index is None:
            raise ValueError(f"Seed {seed} was not in the latest development run for task {task!r}.")
        return self.episodes[index]


def public_observation(observation: dict) -> dict:
    return {
        "time": observation["time"],
        "robot_id": observation["robot_id"],
        "entities": {
            name: {"position": measurement["position"], "covariance": measurement["covariance"]}
            for name, measurement in observation["entities"].items()
        },
        "robot_max_speed": observation["robot_max_speed"],
        "goal": observation["goal"],
    }


def public_episode(task: str, group: str, episode: dict) -> dict:
    steps = episode["steps"]
    public = {"task": task, "seed": episode["seed"], "steps": episode["length"]}
    if group == "pymunk":
        # Preserve the existing pairing of action with its resulting observation.
        observations = [step["observation"] for step in steps[1:]] + [episode["final_observation"]]
        return {
            **public,
            **{field: episode[field] for field in EPISODE_OUTCOMES[task]},
            "initial_observation": public_observation(steps[0]["observation"]),
            "frames": [
                {
                    "observation": public_observation(observation),
                    "action": {key: step["action"][key] for key in ("heading", "throttle")},
                }
                for step, observation in zip(steps, observations, strict=True)
            ],
        }

    # The entire native state_dict is a controller input. Reset/step info, rewards,
    # and other simulator diagnostics remain in the owner artifacts.
    return {
        **public,
        "success": episode["success"],
        "terminated": steps[-1]["terminated"],
        "truncated": steps[-1]["truncated"],
        "initial_observation": steps[0]["observation"],
        "frames": [
            {"observation": step["observation"], "action": step["action"]}
            for step in steps
        ],
        "final_observation": episode["final_observation"],
    }


def publish_feedback(
    owner_results: Path,
    results: Path,
    *,
    suite: str,
    controller_sha256: str,
    started: float,
    retained: dict | None,
) -> ResearchFeedback:
    tasks = {}
    episodes = []
    episode_indices = {}
    seeds = list(DEV_SEEDS)
    benchmark, groups = SUITES[suite]
    for group in groups:
        for task in TASK_GROUPS[group]:
            task_results = owner_results / group / task
            native = json.loads((task_results / "summary.json").read_text(encoding="utf-8"))
            with (task_results / "episodes.jsonl").open(encoding="utf-8") as trace:
                rows = [json.loads(line) for line in trace]
            if (
                native["seeds"] != seeds
                or [row["seed"] for row in native["episodes"]] != seeds
                or [row["seed"] for row in rows] != seeds
            ):
                raise ValueError(f"Incomplete development split for {task}")

            metric_fields = TASK_METRICS[task] if group == "pymunk" else ()
            config_fields = (
                ("task_id", "objective", "max_episode_steps") if group == "pymunk" else MANISKILL_CONFIG
            )
            rate_field = "capture_rate" if "capture_rate" in metric_fields else "success_rate"
            success_rate = native[rate_field]
            tasks[task] = {
                "episodes": len(seeds),
                "seeds": seeds,
                "success_rate": success_rate,
                **{field: native[field] for field in metric_fields},
                "success_rate_change": (
                    None if retained is None else success_rate - retained["tasks"][task]["success_rate"]
                ),
                "configuration": {
                    "runtime": group,
                    "python": native["python"],
                    **{field: native[field] for field in config_fields},
                },
            }
            for row in rows:
                episode_indices[row["seed"], task] = len(episodes)
                episodes.append(public_episode(task, group, row))

    aggregate = sum(task["success_rate"] for task in tasks.values()) / len(tasks)
    summary = {
        "benchmark": benchmark,
        "split": "dev",
        "controller": CONTROLLER,
        "controller_snapshot_sha256": controller_sha256,
        "seeds": seeds,
        "episodes": len(episodes),
        "aggregate_score": aggregate,
        "retained_aggregate_score": None if retained is None else retained["aggregate_score"],
        "retained_controller_snapshot_sha256": None if retained is None else retained["controller_snapshot_sha256"],
        "tasks": tasks,
    }
    staging = results.with_suffix(".tmp")
    staging.mkdir()
    with (staging / "episodes.jsonl").open("w", encoding="utf-8") as trace:
        for episode in episodes:
            trace.write(json.dumps(episode) + "\n")
    summary["runtime_seconds"] = monotonic() - started
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    staging.rename(results)
    return ResearchFeedback(summary, episodes, episode_indices)
