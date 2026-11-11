import json
import shutil
import subprocess

import pytest

from worldsmith import research_lifecycle, research_mcp
from worldsmith.research_feedback import publish_feedback


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "autoresearch"
    workspace.mkdir()
    shutil.copyfile(research_lifecycle.BASELINE / "controller.py", workspace / "controller.py")
    monkeypatch.setattr(research_mcp, "WORKSPACE", workspace)
    monkeypatch.setattr(research_mcp, "OWNER_RUNS", tmp_path / "owner")
    monkeypatch.setattr(research_mcp, "latest_feedback", None)
    return workspace


def test_evaluate_controller_saves_metrics_and_trace(workspace):
    (workspace / "skill.py").write_text("def throttle(): return 0.0\n")
    (workspace / "controller.py").write_text(
        "import json, sys\n"
        "from dataclasses import asdict\n"
        "from autoresearch.skill import throttle\n"
        "from worldsmith.api import Action\n"
        "def control(observation):\n"
        "    action = Action(0, throttle())\n"
        "    print(json.dumps([asdict(observation), asdict(action)]), file=sys.stderr)\n"
        "    return action\n"
        "def make_controller(): return control\n"
    )
    with pytest.raises(ValueError, match="Run evaluate_controller"):
        research_mcp.get_episode_trace(0)

    metrics = research_mcp.evaluate_controller()
    run = workspace / metrics["run_dir"]
    owner = research_mcp.OWNER_RUNS / metrics["run_id"]
    summary = json.loads((run / "summary.json").read_text())
    owner_summary = json.loads((owner / "results/summary.json").read_text())
    assert summary == owner_summary
    assert summary["split"] == "dev" and summary["seeds"] == list(range(8))
    assert metrics["episodes"] == summary["episodes"] == 32
    assert metrics["aggregate_score"] == summary["aggregate_score"] == 0
    assert metrics["tasks"] == summary["tasks"]
    assert set(metrics["tasks"]) == {"interception", "navigation", "rendezvous", "bouncing_interception"}
    assert metrics["tasks"]["interception"]["captures"] == 0
    assert (run.parent / "code/autoresearch/skill.py").read_bytes() == (workspace / "skill.py").read_bytes()
    controller_snapshot = "code/autoresearch/controller.py"
    assert (owner / controller_snapshot).read_bytes() == (run.parent / controller_snapshot).read_bytes()
    record = json.loads((run.parent / "evaluation.json").read_text())
    assert record == json.loads((owner / "evaluation.json").read_text())
    assert record["status"] == "succeeded"
    assert record["run_id"] == metrics["run_id"]

    episode = research_mcp.get_episode_trace(0)
    public_trace = run / "episodes.jsonl"
    saved_episodes = [json.loads(line) for line in public_trace.read_text().splitlines()]
    owner_trace = owner / "results/episodes.jsonl"
    private_episodes = [json.loads(line) for line in owner_trace.read_text().splitlines()]
    received = iter(json.loads(line) for line in (owner / "stderr.txt").read_text().splitlines())
    outcomes = {
        "interception": ("captured",),
        "bouncing_interception": ("captured",),
        "navigation": ("success", "collided"),
        "rendezvous": ("success",),
    }
    assert len(saved_episodes) == len(private_episodes) == 32
    for public, private in zip(saved_episodes, private_episodes, strict=True):
        task = private["initial_observation"]["goal"]
        assert public["task"] == task and public["seed"] == private["seed"]
        assert set(public) == {"task", "seed", "steps", "initial_observation", "frames", *outcomes[task]}
        assert all(public[key] == private[key] for key in outcomes[task])
        assert public["steps"] == private["steps"] == len(public["frames"])
        assert public["initial_observation"] == private["initial_observation"]
        assert private["initial_state"]["robot"] == {"position": [0, 0], "velocity": [0, 0]}
        assert public["initial_observation"]["entities"]["robot"]["position"] != [0, 0]
        observation = public["initial_observation"]
        if task == "bouncing_interception":
            assert set(observation["entities"]) == {
                "robot", "target", "wall_left", "wall_right", "wall_bottom", "wall_top",
            }
            target = private["initial_state"]["target"]
            assert target["body_type"] == "dynamic"
            assert target["shapes"][0]["type"] == "Circle"
            assert set(observation["entities"]["target"]) == {"position", "covariance"}
            assert observation["entities"]["target"]["position"] != target["position"]
            wall = private["initial_state"]["wall_left"]
            assert wall["body_type"] == "static"
            assert "elasticity" in wall["shapes"][0]
            assert observation["entities"]["wall_left"] == {
                "position": wall["position"], "covariance": [[0, 0], [0, 0]],
            }
        for frame, private_frame in zip(public["frames"], private["frames"], strict=True):
            assert set(frame) == {"observation", "action"}
            assert next(received) == [observation, frame["action"]]
            assert frame == {key: private_frame[key] for key in frame}
            assert set(private_frame["state"]) == set(observation["entities"])
            assert all(
                set(entity) == {"position", "covariance"}
                for entity in frame["observation"]["entities"].values()
            )
            assert set(private_frame["state"]["robot"]) == {"position", "velocity"}
            assert {"distance", "obstacle_clearance"} & private_frame.keys()
            observation = frame["observation"]
    assert next(received, None) is None
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

    # MCP serves the published feedback even if an editable run file is replaced.
    public_trace.write_bytes(owner_trace.read_bytes())
    assert research_mcp.get_episode_trace(0) == saved_episodes[0]


def test_feedback_excludes_extra_observation_and_action_fields(workspace, tmp_path):
    metrics = research_mcp.evaluate_controller()
    expected = research_mcp.get_episode_trace(0)
    owner_results = research_mcp.OWNER_RUNS / metrics["run_id"] / "results"
    owner_trace = owner_results / "episodes.jsonl"
    lines = owner_trace.read_text().splitlines(keepends=True)
    episode = json.loads(lines[0])
    episode["initial_observation"]["true_position"] = [0, 0]
    frame = episode["frames"][0]
    frame["observation"]["scoring_distance"] = 4.0
    frame["observation"]["entities"]["target"]["true_velocity"] = [1, 0]
    frame["action"]["diagnostic"] = "owner only"
    lines[0] = json.dumps(episode) + "\n"
    owner_trace.write_text("".join(lines))

    results = tmp_path / "published"
    feedback = publish_feedback(owner_results, results)
    assert feedback.episode(0, "interception") == expected
    with (results / "episodes.jsonl").open() as trace:
        assert json.loads(next(trace)) == expected


def test_feedback_refuses_heldout_results(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"split": "heldout", "seeds": list(range(1000, 1008))}))
    results = tmp_path / "public"
    with pytest.raises(ValueError, match="development split"):
        publish_feedback(tmp_path, results)
    assert not results.exists()


def test_researcher_cannot_select_heldout_split(workspace):
    with pytest.raises(TypeError, match="unexpected keyword argument 'split'"):
        research_mcp.evaluate_controller(split="heldout")
    assert not (workspace / "runs").exists()


def test_archive_and_reset_preserve_both_result_views(workspace, tmp_path):
    metrics = research_mcp.evaluate_controller()
    public_trace = workspace / metrics["run_dir"] / "episodes.jsonl"
    owner_trace = research_mcp.OWNER_RUNS / metrics["run_id"] / "results/episodes.jsonl"
    public_bytes, owner_bytes = public_trace.read_bytes(), owner_trace.read_bytes()
    history = tmp_path / "research-history"
    archive_id = research_lifecycle.archive_session(workspace, history)
    research_lifecycle.reset_workspace(workspace, history, archive_id)
    archived = history / archive_id / "workspace"
    assert not (workspace / "runs").exists()
    assert (archived / public_trace.relative_to(workspace)).read_bytes() == public_bytes
    assert owner_trace.read_bytes() == owner_bytes
    assert not (archived / "runs/owner").exists()


def test_evaluation_timeout_keeps_previous_run(workspace, monkeypatch):
    previous_feedback = research_mcp.ResearchFeedback({}, [], {})
    monkeypatch.setattr(research_mcp, "latest_feedback", previous_feedback)

    def time_out(command, *, timeout, **kwargs):
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(research_mcp.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="Evaluation timed out after 30 seconds"):
        research_mcp.evaluate_controller()
    assert research_mcp.latest_feedback is previous_feedback
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


def test_evaluator_stderr_stays_private(workspace, monkeypatch):
    private_error = "ValueError: privileged scenario diagnostic"
    monkeypatch.setattr(
        research_mcp.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", private_error),
    )
    with pytest.raises(RuntimeError, match="evaluator exited with status 1") as failure:
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert private_error not in str(failure.value)
    assert private_error not in (run / "evaluation.json").read_text()
    assert (research_mcp.OWNER_RUNS / run.name / "stderr.txt").read_text() == private_error


def test_interrupted_evaluation_keeps_snapshot(workspace, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert json.loads((run / "evaluation.json").read_text())["status"] == "running"
    assert (run / "code/autoresearch/controller.py").is_file()
