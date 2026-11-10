import json
import shutil
import subprocess

import pytest

from worldsmith import research_lifecycle, research_mcp


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "autoresearch"
    workspace.mkdir()
    shutil.copyfile(research_lifecycle.BASELINE / "controller.py", workspace / "controller.py")
    monkeypatch.setattr(research_mcp, "WORKSPACE", workspace)
    monkeypatch.setattr(research_mcp, "latest_run", None)
    return workspace


def test_evaluate_controller_saves_metrics_and_trace(workspace):
    (workspace / "skill.py").write_text("def throttle(): return 0.0\n")
    (workspace / "controller.py").write_text(
        "from autoresearch.skill import throttle\n"
        "from worldsmith.api import Action\n"
        "def make_controller(): return lambda observation: Action(0, throttle())\n"
    )
    with pytest.raises(ValueError, match="Run evaluate_controller"):
        research_mcp.get_episode_trace(0)

    metrics = research_mcp.evaluate_controller()
    run = workspace / metrics["run_dir"]
    summary = json.loads((run / "summary.json").read_text())
    assert metrics["episodes"] == summary["episodes"] == 24
    assert metrics["aggregate_score"] == summary["aggregate_score"] == 0
    assert metrics["tasks"] == summary["tasks"]
    assert set(metrics["tasks"]) == {"interception", "navigation", "rendezvous"}
    assert metrics["tasks"]["interception"]["captures"] == 0
    assert (run.parent / "code/autoresearch/skill.py").read_bytes() == (workspace / "skill.py").read_bytes()
    record = json.loads((run.parent / "evaluation.json").read_text())
    assert record["status"] == "succeeded"
    assert record["run_id"] == metrics["run_id"]

    episode = research_mcp.get_episode_trace(0)
    saved_episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert episode == saved_episodes[0]
    for task in metrics["tasks"]:
        episode = research_mcp.get_episode_trace(0, task)
        assert episode == next(
            item for item in saved_episodes
            if item["seed"] == 0 and item["initial_observation"]["goal"] == task
        )
    with pytest.raises(ValueError, match="task 'unknown'"):
        research_mcp.get_episode_trace(0, "unknown")
    with pytest.raises(ValueError, match="Seed 1000 was not in the latest development run"):
        research_mcp.get_episode_trace(1000)


def test_evaluation_timeout_keeps_previous_run(workspace, monkeypatch):
    previous_run = workspace / "previous"
    monkeypatch.setattr(research_mcp, "latest_run", previous_run)

    def time_out(command, *, timeout, **kwargs):
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(research_mcp.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="Evaluation timed out after 30 seconds"):
        research_mcp.evaluate_controller()
    assert research_mcp.latest_run == previous_run
    [run] = (workspace / "runs").iterdir()
    assert json.loads((run / "evaluation.json").read_text())["status"] == "timed_out"
    assert (run / "code/autoresearch/controller.py").is_file()


def test_evaluation_reports_subprocess_error(workspace):
    controller = workspace / "controller.py"
    controller.write_text("raise ValueError('invalid action')\n")

    with pytest.raises(RuntimeError, match="Evaluation failed: ValueError: invalid action"):
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert json.loads((run / "evaluation.json").read_text())["status"] == "failed"
    assert (run / "code/autoresearch/controller.py").read_bytes() == controller.read_bytes()


def test_interrupted_evaluation_keeps_snapshot(workspace, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert json.loads((run / "evaluation.json").read_text())["status"] == "running"
    assert (run / "code/autoresearch/controller.py").is_file()
