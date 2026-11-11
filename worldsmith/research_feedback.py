"""Publish the development feedback exposed to AutoResearch."""

import json
from dataclasses import dataclass
from pathlib import Path

from worldsmith.benchmark.tasks.constants import COMBINED_SPLITS

SUMMARY_FIELDS = ("benchmark", "goal", "controller", "split", "seeds", "episodes", "aggregate_score")
TASK_METRICS = {
    "interception": ("captures", "capture_rate", "mean_closest_distance"),
    "bouncing_interception": ("captures", "capture_rate", "mean_closest_distance"),
    "navigation": ("collision_rate", "mean_final_destination_distance", "mean_minimum_obstacle_clearance"),
    "rendezvous": ("mean_longest_tracking_seconds", "mean_distance", "mean_final_distance"),
}
EPISODE_OUTCOMES = {
    "interception": ("captured",),
    "bouncing_interception": ("captured",),
    "navigation": ("success", "collided"),
    "rendezvous": ("success",),
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


def publish_feedback(owner_results: Path, results: Path) -> ResearchFeedback:
    owner_summary = json.loads((owner_results / "summary.json").read_text(encoding="utf-8"))
    if owner_summary["split"] != "dev" or owner_summary["seeds"] != list(COMBINED_SPLITS["dev"]):
        raise ValueError("Research feedback requires the combined development split.")
    summary = {field: owner_summary[field] for field in SUMMARY_FIELDS}
    summary["tasks"] = {
        task: {field: metrics[field] for field in ("episodes", "success_rate", *TASK_METRICS[task])}
        for task, metrics in owner_summary["tasks"].items()
    }

    episodes = []
    episode_indices = {}
    with (owner_results / "episodes.jsonl").open(encoding="utf-8") as trace:
        for line in trace:
            episode = json.loads(line)
            seed = episode["seed"]
            task = episode["initial_observation"]["goal"]
            episode_indices[seed, task] = len(episodes)
            episodes.append({
                "task": task,
                "seed": seed,
                "steps": episode["steps"],
                **{field: episode[field] for field in EPISODE_OUTCOMES[task]},
                "initial_observation": public_observation(episode["initial_observation"]),
                "frames": [
                    {
                        "observation": public_observation(frame["observation"]),
                        "action": {
                            "heading": frame["action"]["heading"],
                            "throttle": frame["action"]["throttle"],
                        },
                    }
                    for frame in episode["frames"]
                ],
            })

    results.mkdir()
    (results / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with (results / "episodes.jsonl").open("w", encoding="utf-8") as trace:
        for episode in episodes:
            trace.write(json.dumps(episode) + "\n")
    return ResearchFeedback(summary, episodes, episode_indices)
