import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic

import pytest

from worldsmith import research_lifecycle, research_mcp
from worldsmith.benchmark.task_catalog import TASK_GROUPS
from worldsmith.research_evaluation import DEV_SEEDS
from worldsmith.research_feedback import publish_feedback

RUN = subprocess.run


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "autoresearch"
    workspace.mkdir()
    shutil.copyfile(research_lifecycle.BASELINE / "controller.py", workspace / "controller.py")
    monkeypatch.setattr(research_mcp, "WORKSPACE", workspace)
    monkeypatch.setattr(research_mcp, "SUITE", "all")
    monkeypatch.setattr(research_mcp, "OWNER_RUNS", tmp_path / "owner")
    monkeypatch.setattr(research_mcp, "latest_feedback", None)
    monkeypatch.setattr(research_mcp, "retained_summary", None)
    for group in ("maniskill", "libero", "robosuite"):
        interpreter = tmp_path / f"{group}-python"
        interpreter.symlink_to(sys.executable)
        monkeypatch.setenv(f"WORLDSMITH_{group.upper()}_PYTHON", str(interpreter))
    return workspace


@pytest.fixture
def native_workers(workspace, monkeypatch):
    # Standard tests need only Pymunk. The opt-in integration tests run the native simulators.
    class Worker:
        def __init__(self):
            self.calls = []
            self.successes = 8
            self.after_pymunk = None

        def run(self, command, *, cwd, **kwargs):
            if "worldsmith.research_evaluation" not in command:
                return RUN(command, cwd=cwd, **kwargs)
            group, directory = command[-2:]
            output = Path(directory)
            self.calls.append((group, command[0], cwd, (cwd / "autoresearch/controller.py").read_bytes()))
            if group == "pymunk":
                result = RUN(command, cwd=cwd, **kwargs)
                if self.after_pymunk:
                    self.after_pymunk()
                return result

            successes = self.successes if group == "maniskill" else 0
            for task in TASK_GROUPS[group]:
                task_output = output / task
                task_output.mkdir(parents=True)
                episodes = []
                for seed in DEV_SEEDS:
                    observation = {"agent": {"qpos": [[0.0]]}, "extra": {"tcp_pose": [[1.0]]}}
                    episodes.append({
                        "seed": seed,
                        "length": 1,
                        "success": seed < successes,
                        "reset_info": {"private_reset": True},
                        "steps": [{
                            "observation": observation,
                            "action": [0.0],
                            "terminated": seed < successes,
                            "truncated": seed >= successes,
                            "reward": 1.0,
                            "info": {"private_step": True},
                        }],
                        "final_observation": observation,
                    })
                (task_output / "episodes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in episodes),
                )
                (task_output / "summary.json").write_text(json.dumps({
                    "task_id": "PushCube-v1",
                    "task_version": 1,
                    "max_episode_steps": 50,
                    "action_shape": [4],
                    "sim_backend": "physx_cpu",
                    "render_backend": "sapien_cpu",
                    "versions": {"mani-skill-nightly": "private version"},
                    "python": "3.11",
                    "obs_mode": "state_dict",
                    "seeds": list(DEV_SEEDS),
                    "episodes": [{key: row[key] for key in ("seed", "length", "success")} for row in episodes],
                    "success_rate": successes / len(DEV_SEEDS),
                    "private_metric": 123,
                }))
            return subprocess.CompletedProcess(command, 0)

    worker = Worker()
    monkeypatch.setattr(research_mcp.subprocess, "run", worker.run)
    return worker


def test_full_suite_snapshot_scoring_and_public_feedback(workspace, native_workers):
    source = '''from worldsmith.api import Action

def make_controller(*, task):
    def control(observation):
        assert observation.goal == task
        return Action(0, 0)
    return control
'''
    controller = workspace / "controller.py"
    controller.write_text(source)
    native_workers.after_pymunk = lambda: controller.write_text("raise RuntimeError('live edit')\n")
    with pytest.raises(ValueError, match="Run evaluate_controller"):
        research_mcp.get_episode_trace(0)

    metrics = research_mcp.evaluate_controller()
    run = workspace / metrics["run_dir"]
    owner = research_mcp.OWNER_RUNS / metrics["run_id"]
    assert native_workers.calls == [
        ("pymunk", sys.executable, owner / "code", source.encode()),
        ("maniskill", str(workspace.parent / "maniskill-python"), owner / "code", source.encode()),
        ("libero", str(workspace.parent / "libero-python"), owner / "code", source.encode()),
        ("robosuite", str(workspace.parent / "robosuite-python"), owner / "code", source.encode()),
    ]
    assert (run.parent / "code/autoresearch/controller.py").read_text() == source
    summary = json.loads((run / "summary.json").read_text())
    assert summary["benchmark"] == "worldsmith-all-v3"
    assert summary["split"] == "dev" and summary["seeds"] == list(range(8))
    assert summary["episodes"] == 136
    assert summary["runtime_seconds"] > 0
    assert set(summary["tasks"]) == {task for group in TASK_GROUPS.values() for task in group}
    assert all(row["episodes"] == 8 for row in summary["tasks"].values())
    assert summary["aggregate_score"] == pytest.approx(6 / 17)
    assert summary["tasks"]["interception"]["captures"] == 0
    assert summary["tasks"]["open_middle_drawer"]["success_rate"] == 0
    assert not research_mcp.get_episode_trace(0, "open_middle_drawer")["success"]
    assert summary["tasks"]["push_cube"]["configuration"] == {
        "max_episode_steps": 50, "action_shape": [4],
    }
    assert set(summary["tasks"]["interception"]["configuration"]) == {"max_episode_steps", "objective"}
    for term in ("pymunk", "maniskill", "robosuite", "sapien", "physx", "state_dict", "PushCube-v1", "private version"):
        assert term not in json.dumps(metrics)
    native = json.loads((owner / "results/maniskill/push_cube/summary.json").read_text())
    assert native["sim_backend"] == "physx_cpu" and native["versions"]["mani-skill-nightly"] == "private version"
    assert "private_metric" not in summary["tasks"]["push_cube"]
    assert all(row["success_rate_change"] is None for row in summary["tasks"].values())
    record = json.loads((run.parent / "evaluation.json").read_text())
    assert record == json.loads((owner / "evaluation.json").read_text())
    assert record["status"] == "succeeded"

    saved = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert len(saved) == 136
    for public in saved:
        assert research_mcp.get_episode_trace(public["seed"], public["task"]) == public
        assert all(set(frame) == {"observation", "action"} for frame in public["frames"])
        assert "reset_info" not in public and "private_step" not in json.dumps(public)
    private_trace = owner / "results/maniskill/push_cube/episodes.jsonl"
    private = json.loads(private_trace.read_text().splitlines()[0])
    push = research_mcp.get_episode_trace(0, "push_cube")
    assert push["success"] and push["terminated"] and not push["truncated"]
    assert push["frames"][0]["observation"] == private["steps"][0]["observation"]
    assert push["frames"][0]["action"] == private["steps"][0]["action"]
    intercept = research_mcp.get_episode_trace(0)
    path = owner / "results/pymunk/interception/episodes.jsonl"
    private = json.loads(path.read_text().splitlines()[0])
    assert intercept["initial_observation"] == private["steps"][0]["observation"]
    assert intercept["frames"][0]["observation"] == private["steps"][0]["observation"]
    assert intercept["frames"][0]["action"] == private["steps"][0]["action"]
    assert intercept["final_observation"] == private["final_observation"]
    assert set(intercept) == set(push) | {"captured"}
    assert intercept["success"] == intercept["captured"]
    assert "state" in private["reset_info"] and "state" not in intercept
    with pytest.raises(ValueError, match="task 'unknown'"):
        research_mcp.get_episode_trace(0, "unknown")
    with pytest.raises(ValueError, match="Seed 1000 was not in the latest development run"):
        research_mcp.get_episode_trace(1000, "push_cube")

    # MCP serves the published feedback even if an editable run file is replaced.
    (run / "episodes.jsonl").write_bytes(private_trace.read_bytes())
    assert research_mcp.get_episode_trace(0, "push_cube") == push


def test_maniskill_suite_runs_only_maniskill_and_defaults_to_push_cube(workspace, native_workers, monkeypatch):
    monkeypatch.setattr(research_mcp, "SUITE", "maniskill")
    native_workers.successes = 6

    metrics = research_mcp.evaluate_controller()
    assert [call[0] for call in native_workers.calls] == ["maniskill"]
    assert metrics["benchmark"] == "worldsmith-manipulation-v1"
    assert metrics["episodes"] == 48
    assert set(metrics["tasks"]) == set(TASK_GROUPS["maniskill"])
    assert metrics["aggregate_score"] == 0.75
    assert research_mcp.get_episode_trace(0) == research_mcp.get_episode_trace(0, "push_cube")
    with pytest.raises(ValueError, match="task 'interception'"):
        research_mcp.get_episode_trace(0, "interception")


@pytest.mark.parametrize(("suite", "benchmark", "first_task"), [
    ("libero", "worldsmith-arm-position-v1", "open_middle_drawer"),
    ("robosuite", "worldsmith-arm-skills-v1", "open_latched_door"),
])
def test_arm_suite_dispatch_and_public_feedback(workspace, native_workers, monkeypatch, suite, benchmark, first_task):
    monkeypatch.setattr(research_mcp, "SUITE", suite)
    metrics = research_mcp.evaluate_controller()
    assert [call[0] for call in native_workers.calls] == [suite]
    assert metrics["benchmark"] == benchmark
    assert metrics["episodes"] == 24
    assert set(metrics["tasks"]) == set(TASK_GROUPS[suite])
    assert suite not in json.dumps(metrics)
    episode = research_mcp.get_episode_trace(0)
    assert episode == research_mcp.get_episode_trace(0, first_task)
    assert "private_step" not in json.dumps(episode)
    assert "reset_info" not in episode


def test_per_task_changes_use_retained_run(workspace, native_workers):
    baseline = research_mcp.evaluate_controller()
    native_workers.successes = 6
    weaker = research_mcp.evaluate_controller()
    tie = research_mcp.evaluate_controller()
    assert weaker["aggregate_score"] < baseline["aggregate_score"]
    assert weaker["retained_aggregate_score"] == tie["retained_aggregate_score"] == baseline["aggregate_score"]
    assert tie["tasks"]["push_cube"]["success_rate_change"] == -0.25
    assert tie["tasks"]["interception"]["success_rate_change"] == 0


def test_feedback_refuses_incomplete_or_heldout_results(workspace, native_workers, tmp_path):
    metrics = research_mcp.evaluate_controller()
    owner_results = research_mcp.OWNER_RUNS / metrics["run_id"] / "results"
    last_trace = owner_results / "maniskill/push_t/episodes.jsonl"
    rows = last_trace.read_text().splitlines()
    last_trace.write_text("\n".join(rows[:-1]) + "\n")

    results = tmp_path / "public"
    with pytest.raises(ValueError, match="Incomplete development split for push_t"):
        publish_feedback(owner_results, results, suite="all", controller_sha256="", started=monotonic(), retained=None)
    first_summary = owner_results / "pymunk/interception/summary.json"
    summary = json.loads(first_summary.read_text())
    summary["seeds"] = list(range(1000, 1008))
    first_summary.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Incomplete development split for interception"):
        publish_feedback(owner_results, results, suite="all", controller_sha256="", started=monotonic(), retained=None)
    assert not results.exists()


def test_researcher_cannot_select_heldout_split(workspace):
    with pytest.raises(TypeError, match="unexpected keyword argument 'split'"):
        research_mcp.evaluate_controller(split="heldout")
    assert not (workspace / "runs").exists()


def test_archive_and_reset_preserve_both_result_views(workspace, native_workers, tmp_path):
    metrics = research_mcp.evaluate_controller()
    public_trace = workspace / metrics["run_dir"] / "episodes.jsonl"
    owner_trace = research_mcp.OWNER_RUNS / metrics["run_id"] / "results/maniskill/push_cube/episodes.jsonl"
    public_bytes, owner_bytes = public_trace.read_bytes(), owner_trace.read_bytes()
    history = tmp_path / "research-history"
    archive_id = research_lifecycle.archive_session(workspace, history)
    research_lifecycle.reset_workspace(workspace, history, archive_id)
    archived = history / archive_id / "workspace"
    assert not (workspace / "runs").exists()
    assert (archived / public_trace.relative_to(workspace)).read_bytes() == public_bytes
    assert owner_trace.read_bytes() == owner_bytes
    assert not (archived / "runs/owner").exists()


@pytest.mark.parametrize("timed_out", [True, False])
def test_failed_task_keeps_previous_feedback_and_diagnostics_private(workspace, monkeypatch, timed_out):
    previous_feedback = research_mcp.ResearchFeedback({}, [], {})
    monkeypatch.setattr(research_mcp, "latest_feedback", previous_feedback)
    private_error = "ValueError: privileged scenario diagnostic"

    def fail(command, *, timeout, stdout, **kwargs):
        assert 0 < timeout <= research_mcp.SUITE_TIMEOUT_SECONDS["all"]
        if command[-2] == "pymunk":
            return subprocess.CompletedProcess(command, 0)
        output = Path(command[-1])
        output.mkdir(parents=True)
        (output / "task.txt").write_text("push_cube")
        stdout.write(private_error)
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(research_mcp.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="while evaluating push_cube") as failure:
        research_mcp.evaluate_controller()
    assert research_mcp.latest_feedback is previous_feedback
    [run] = (workspace / "runs").iterdir()
    status = json.loads((run / "evaluation.json").read_text())
    assert status["status"] == ("timed_out" if timed_out else "failed")
    assert private_error not in str(failure.value)
    assert private_error not in status["error"]
    assert "maniskill" not in status["error"]
    assert (research_mcp.OWNER_RUNS / run.name / "maniskill-stderr.txt").read_text() == private_error
    assert (run / "code/autoresearch/controller.py").is_file()
    assert not (run / "results").exists()


@pytest.mark.parametrize(("source", "diagnostic", "public_message"), [
    (
        "raise ValueError('private pymunk import failure')\n",
        "Controller import: ValueError: private pymunk import failure",
        "Could not load controller.py or its make_controller factory",
    ),
    (
        "def make_controller(): raise ValueError('private pymunk setup failure')\n",
        "interception, seed 0, controller setup: ValueError: private pymunk setup failure",
        "interception, seed 0: make_controller raised an exception",
    ),
    (
        "def control(observation): raise RuntimeError('private pymunk action failure')\n"
        "def make_controller(): return control\n",
        "interception, seed 0, step 1: RuntimeError: private pymunk action failure",
        "interception, seed 0, step 1: controller raised an exception",
    ),
    (
        "from worldsmith.api import Action\n"
        "def make_controller(): return lambda observation: Action(0, 2)\n",
        "interception, seed 0, step 1: Action requires a finite heading and throttle in [0, 1]",
        "interception, seed 0, step 1: controller returned an invalid action",
    ),
], ids=["import", "setup", "action", "invalid-throttle"])
def test_controller_errors_without_publishing_tracebacks(workspace, source, diagnostic, public_message):
    controller = workspace / "controller.py"
    controller.write_text(source)
    with pytest.raises(RuntimeError) as failure:
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert str(failure.value) == f"Evaluation failed: {public_message}. Run: {run.name}"
    status = json.loads((run / "evaluation.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == f"Evaluation failed: {public_message}"
    assert (run / "code/autoresearch/controller.py").read_bytes() == controller.read_bytes()
    assert not (run / "results").exists()
    owner = research_mcp.OWNER_RUNS / run.name
    assert (owner / "results/pymunk/controller_error.txt").read_text() == diagnostic + "\n"
    assert "Traceback" in (owner / "pymunk-stderr.txt").read_text()


def test_interrupted_evaluation_keeps_snapshot(workspace, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research_mcp.evaluate_controller()
    [run] = (workspace / "runs").iterdir()
    assert json.loads((run / "evaluation.json").read_text())["status"] == "running"
    assert (run / "code/autoresearch/controller.py").is_file()
