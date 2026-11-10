import json
import subprocess

import pytest

from worldsmith import research_mcp


def test_evaluate_controller_saves_metrics_and_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(research_mcp, "WORKSPACE", tmp_path)
    monkeypatch.setattr(research_mcp, "latest_run", None)

    with pytest.raises(ValueError, match="Run evaluate_controller"):
        research_mcp.get_episode_trace(12)

    metrics = research_mcp.evaluate_controller()
    run = tmp_path / metrics["run_dir"]
    summary = json.loads((run / "summary.json").read_text())
    assert metrics["episodes"] == summary["episodes"] == 32
    assert metrics["captures"] == summary["captures"]
    assert metrics["capture_rate"] == summary["capture_rate"]

    episode = research_mcp.get_episode_trace(12)
    saved_episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert episode == next(item for item in saved_episodes if item["seed"] == 12)
    with pytest.raises(ValueError, match="Seed 1000 was not in the latest development run"):
        research_mcp.get_episode_trace(1000)


def test_evaluation_timeout_keeps_previous_run(tmp_path, monkeypatch):
    previous_run = tmp_path / "previous"
    monkeypatch.setattr(research_mcp, "WORKSPACE", tmp_path)
    monkeypatch.setattr(research_mcp, "latest_run", previous_run)

    def time_out(command, *, timeout, **kwargs):
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(research_mcp.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="Evaluation timed out after 30 seconds"):
        research_mcp.evaluate_controller()
    assert research_mcp.latest_run == previous_run


def test_evaluation_reports_subprocess_error(monkeypatch):
    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stderr="Traceback\nValueError: invalid action\n")

    monkeypatch.setattr(research_mcp.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="^Evaluation failed: ValueError: invalid action$"):
        research_mcp.evaluate_controller()
