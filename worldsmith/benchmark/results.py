"""Snapshots and JSON artifacts shared by benchmark evaluations."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import NamedTuple, TextIO


class TaskEvaluation(NamedTuple):
    episodes: list[dict]
    metrics: dict


def snapshot(value):
    """Detach controller inputs and simulator outputs from mutable buffers."""
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_episode(trace: TextIO, episode: dict, *, legacy: bool = False) -> None:
    options = {} if legacy else {"allow_nan": False, "separators": (",", ":")}
    trace.write(json.dumps(episode, **options) + "\n")


def write_scored_episode(trace: TextIO, episode: dict, metrics: dict) -> None:
    write_episode(trace, {**episode, **metrics})


def write_summary(output: Path, summary: dict) -> None:
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
