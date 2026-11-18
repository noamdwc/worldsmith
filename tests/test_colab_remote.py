"""Remote evaluation, artifact recovery, and existing MCP contracts."""

import ast
import json
import subprocess
import tarfile
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import pytest

from test_colab_environments import PREPARED_OBJECT
from worldsmith import research_lifecycle
from worldsmith.colab import backend, remote_job, session, storage
from worldsmith.research_mcp import research_mcp

@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Fake the CLI transport and R2, with real job unpacking and evaluation."""
    staging = tmp_path / "vm"
    staging.mkdir()
    calls = []
    objects = {}
    requests = []

    class Storage:
        def check(self):
            pass

        def exists(self, key):
            return key in objects

        def head(self, key):
            return PREPARED_OBJECT

        def put(self, key, source, *, cache=False):
            assert key not in objects
            objects[key] = source.read_bytes()

        def signed_url(self, key, *, expires, upload=False):
            assert 0 < expires <= 3600
            return f"https://example.test/{key}?private-signature"

        def download(self, key, destination):
            if key not in objects:
                raise storage.ObjectMissing("R2 has no artifact bundle for this job")
            destination.write_bytes(objects[key])

    def upload(request, *, timeout):
        key = request.full_url.split("example.test/")[1].split("?")[0]
        assert request.get_method() == "PUT"
        assert request.get_header("If-none-match") == "*"
        assert key not in objects
        objects[key] = request.data.read()
        return nullcontext()

    def _command(self, action, *args, **kwargs):
        calls.append((self.name, action, args))
        if action == "new":
            assert args == ("--gpu", "L4")
        elif action == "upload":
            local, remote = args
            (staging / Path(remote).name).write_bytes(Path(local).read_bytes())
        elif action == "exec":
            code = kwargs["code"]
            if "setup-colab.py" in code:
                with tarfile.open(staging / "bootstrap.tar.gz") as archive:
                    assert "scripts/setup-colab.py" in archive.getnames()
                    assert "pyproject.toml" in archive.getnames()
                    assert "simulation_environments/maniskill/uv.lock" in archive.getnames()
                    assert "worldsmith/colab/remote_job.py" not in archive.getnames()
                    assert "autoresearch/controller.py" not in archive.getnames()
                return "WORLDSMITH_COLAB_READY\n"
            assert "private-signature" not in code
            call = next(node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen")
            job_id, budget = ast.literal_eval(call.args[0])[-2:]
            dispatch = json.loads((staging / f"{job_id}.json").read_text())
            requests.append(dispatch["request"])
            status = remote_job.run_job(job_id, float(budget), staging)
            return f"WORLDSMITH_JOB {job_id} {status}\n"
        return ""

    monkeypatch.setattr(session, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(session.ColabSession, "_command", _command)
    monkeypatch.setattr(session, "R2", Storage)
    run = session.subprocess.run

    def prepare_wheel(command, **kwargs):
        if len(command) > 1 and command[1].endswith("prepare-colab-wheel.sh"):
            return subprocess.CompletedProcess(command, 0, "test.whl\nhttps://example.test/wheel\n")
        return run(command, **kwargs)

    monkeypatch.setattr(session.subprocess, "run", prepare_wheel)
    monkeypatch.setattr(backend, "R2", Storage)
    monkeypatch.setattr(backend, "RECOVERY_WINDOW_SECONDS", 0)
    monkeypatch.setenv("PYTHONPATH", str(research_lifecycle.ROOT))
    monkeypatch.setattr(remote_job, "urlopen", upload)
    return calls, objects, requests


@pytest.fixture
def researcher(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "controller.py"
    candidate.write_bytes((research_lifecycle.BASELINE / "controller.py").read_bytes())
    monkeypatch.setattr(research_mcp, "WORKSPACE", workspace)
    monkeypatch.setattr(research_mcp, "OWNER_RUNS", tmp_path / "owner")
    monkeypatch.setattr(research_mcp, "SUITE", "navigation")
    monkeypatch.setattr(research_mcp, "latest_feedback", None)
    monkeypatch.setattr(research_mcp, "retained_summary", None)
    monkeypatch.setenv("WORLDSMITH_EVALUATION_BACKEND", "colab")
    monkeypatch.delenv("WORLDSMITH_COLAB_SESSION", raising=False)
    return candidate


@pytest.mark.parametrize("override, workers", [(None, 4), (2, 2)])
def test_mcp_reuses_session_and_isolates_candidate_processes(fake_cli, researcher, monkeypatch, override, workers):
    calls, objects, requests = fake_cli
    runtime = session.start_session()
    monkeypatch.setenv("WORLDSMITH_COLAB_SESSION", runtime.name)
    monkeypatch.setattr(backend, "WORKER_OVERRIDE", override)
    # The two evaluations import the same module name. An interpreter reused
    # across jobs would either cache the old source or retain this global.
    prefix = (
        "import builtins\n"
        "assert not hasattr(builtins, 'worldsmith_colab_test')\n"
        "builtins.worldsmith_colab_test = True\n"
    )
    source = researcher.read_text()
    researcher.write_text(prefix + source)
    first = research_mcp.evaluate_controller()
    first_trace = research_mcp.get_episode_trace(0)
    researcher.write_text(prefix + source + "\n# second candidate\n")
    second = research_mcp.evaluate_controller()
    runtime.stop()

    assert first["controller_snapshot_sha256"] != second["controller_snapshot_sha256"]
    assert second["retained_controller_snapshot_sha256"] == first["controller_snapshot_sha256"]
    assert set(first["tasks"]) == {"navigation"}
    assert first["aggregate_score"] == second["aggregate_score"]
    assert research_mcp.get_episode_trace(0) == first_trace
    assert first["seeds"] == second["seeds"] == list(range(8))
    assert first["run_dir"] == f"runs/{first['run_id']}/results"
    for result, request in zip((first, second), requests, strict=True):
        run_id = result["run_id"]
        job = f"{backend.PREFIX}/{run_id}"
        assert request["groups"] == {"pymunk": {"tasks": ["navigation"], "workers": workers}}
        assert request == json.loads(objects[f"{job}/input/request.json"])
        assert request["job_id"] == run_id
        with tarfile.open(fileobj=BytesIO(objects[f"{job}/input/candidate.tar.gz"])) as archive:
            stored_source = archive.extractfile("autoresearch/controller.py").read().decode()
        assert stored_source == (prefix + source if result is first else researcher.read_text())
        assert json.loads(objects[f"{job}/output/completed.json"])["job_id"] == run_id
        assert (research_mcp.OWNER_RUNS / run_id / "pymunk-stderr.txt").is_file()

    assert [action for _, action, _ in calls if action in {"new", "exec", "stop"}] == [
        "new", "exec", "exec", "exec", "stop",
    ]
    assert {name for name, _, _ in calls} == {runtime.name}
    assert session.tracked_session() is None
    assert not any(b"private-signature" in content for content in objects.values())


def test_mcp_requires_existing_session(fake_cli, researcher):
    with pytest.raises(RuntimeError, match="Colab session is unavailable"):
        research_mcp.evaluate_controller()
    assert fake_cli == ([], {}, [])


@pytest.mark.parametrize("fault, message", [
    ("controller", "Could not load controller"),
    ("simulation", "Simulation execution failed"),
    ("runtime", "Colab session disappeared"),
    ("command", "Colab remote evaluation command did not complete"),
    ("transfer", "R2 remote artifact upload failed"),
    ("download", "R2 download failed"),
    ("corrupt", "Colab returned invalid evaluation artifacts"),
    ("overlap", "Another Colab evaluation is in progress"),
    ("timeout", "Evaluation timed out"),
])
def test_failures_are_distinct_and_cannot_complete(fake_cli, researcher, monkeypatch, fault, message):
    _, objects, _ = fake_cli
    session.start_session()
    if fault == "controller":
        researcher.write_text("raise ValueError('private controller diagnostic')\n")
    elif fault == "simulation":
        monkeypatch.setattr(remote_job, "run_group_process", lambda *a, **kw: subprocess.CompletedProcess([], 3))
    elif fault == "timeout":
        def expire(command, *, timeout, **kwargs):
            raise subprocess.TimeoutExpired(command, timeout)
        monkeypatch.setattr(remote_job, "run_group_process", expire)
    elif fault in {"runtime", "command"}:
        monkeypatch.setattr(session.ColabSession, "execute", lambda *a, **kw: "No completion marker")
        if fault == "runtime":
            monkeypatch.setattr(session.ColabSession, "is_assigned", lambda self: False)
    elif fault == "transfer":
        def abort(*a, **kw):
            raise OSError("private-signature")
        monkeypatch.setattr(remote_job, "urlopen", abort)
    elif fault == "download":
        storage = backend.R2
        def abort(self, key, destination):
            raise backend.StorageError("R2 download failed")
        monkeypatch.setattr(storage, "download", abort)
    elif fault == "corrupt":
        monkeypatch.setattr(backend.R2, "download", lambda self, key, destination: destination.write_bytes(b"partial"))
    lock = session.session_lock("evaluation") if fault == "overlap" else nullcontext()
    with lock, pytest.raises(RuntimeError, match=message) as error:
        research_mcp.evaluate_controller()
    assert "private" not in str(error.value)
    assert research_mcp.latest_feedback is None
    assert all(not key.endswith("completed.json") for key in objects)
    [run] = (researcher.parent / "runs").iterdir()
    status = json.loads((run / "evaluation.json").read_text())
    assert status["status"] == ("timed_out" if fault == "timeout" else "failed")
    assert not (run / "results").exists()


def test_interrupted_evaluation_keeps_r2_inputs_without_a_completion_marker(fake_cli, researcher, monkeypatch):
    _, objects, _ = fake_cli
    session.start_session()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(session.ColabSession, "execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research_mcp.evaluate_controller()
    assert any(key.endswith("input/request.json") for key in objects)
    assert all("/output/" not in key for key in objects)
    assert research_mcp.latest_feedback is None


@pytest.mark.parametrize("reply", ["failed", "timed out", "lost", "controller error"])
def test_recovers_result_or_controller_diagnostics_after_lost_reply(fake_cli, researcher, monkeypatch, reply):
    calls, objects, requests = fake_cli
    session.start_session()
    execute = session.ColabSession.execute
    if reply == "controller error":
        researcher.write_text("raise RuntimeError('private candidate import failure')\n")

    def lose_reply(self, code, *, timeout):
        execute(self, code, timeout=timeout)
        if reply == "lost":
            return ""
        raise session.ColabError(f"Colab exec {reply}")

    monkeypatch.setattr(session.ColabSession, "execute", lose_reply)
    if reply == "controller error":
        with pytest.raises(RuntimeError, match="Could not load controller"):
            research_mcp.evaluate_controller()
        assert all(not key.endswith("completed.json") for key in objects)
    else:
        result = research_mcp.evaluate_controller()
        completed = f"{backend.PREFIX}/{result['run_id']}/output/completed.json"
        assert json.loads(objects[completed])["seeds"] == list(range(8))
    assert len(requests) == 1
    assert [action for _, action, _ in calls].count("exec") == 2  # Setup and one job.


@pytest.mark.parametrize("fault", [
    "job_id", "seeds", "groups", "missing trace", "trace seeds", "trace body", "truncated gzip",
])
def test_recovery_rejects_mismatched_or_incomplete_artifacts(fake_cli, researcher, tmp_path, monkeypatch, fault):
    _, objects, requests = fake_cli
    session.start_session()
    execute = session.ColabSession.execute

    def damage_output(self, code, *, timeout):
        execute(self, code, timeout=timeout)
        key = next(key for key in objects if key.endswith("artifacts.tar.gz"))
        if fault == "truncated gzip":
            objects[key] = objects[key][:-4]
            return ""
        artifacts = tmp_path / "damaged"
        with tarfile.open(fileobj=BytesIO(objects[key])) as archive:
            archive.extractall(artifacts, filter="data")
        manifest_file = artifacts / "manifest.json"
        trace = artifacts / "results/pymunk/navigation/episodes.jsonl"
        if fault == "missing trace":
            trace.unlink()
        elif fault == "trace seeds":
            first, *rest = trace.read_text().splitlines()
            trace.write_text("\n".join([json.dumps({**json.loads(first), "seed": 8}), *rest]) + "\n")
        elif fault == "trace body":
            first, *rest = trace.read_text().splitlines()
            trace.write_text("\n".join([json.dumps({**json.loads(first), "steps": []}), *rest]) + "\n")
        else:
            manifest = json.loads(manifest_file.read_text())
            del manifest[fault]
            if fault == "job_id":
                manifest["job_id"] = "different-job"
            manifest_file.write_text(json.dumps(manifest))
        modified = BytesIO()
        with tarfile.open(fileobj=modified, mode="w:gz") as archive:
            archive.add(artifacts, arcname=".")
        objects[key] = modified.getvalue()
        raise session.ColabError("Colab lost its response")

    monkeypatch.setattr(session.ColabSession, "execute", damage_output)
    with pytest.raises(RuntimeError, match="invalid evaluation artifacts"):
        research_mcp.evaluate_controller()
    assert len(requests) == 1
    assert all(not key.endswith("completed.json") for key in objects)


@pytest.mark.parametrize("assigned, notice, message", [
    (True, "Colab exec timed out", "Colab exec timed out[.] R2 has no artifact bundle"),
    (False, "Colab exec lost the connection", "Colab exec lost the connection.*Colab session disappeared.*R2 has no artifact bundle"),
])
def test_missing_artifacts_fail_without_resubmission_or_state_change(
    fake_cli, researcher, monkeypatch, assigned, notice, message,
):
    _, objects, requests = fake_cli
    session.start_session()
    baseline = research_mcp.evaluate_controller()
    retained = research_mcp.retained_summary
    trace = research_mcp.get_episode_trace(0)
    outputs = {key for key in objects if "/output/" in key}
    researcher.write_text(researcher.read_text() + "\n# Candidate under test\n")

    def fail(*a, **kw):
        requests.append("submitted")
        raise session.ColabError(notice)

    monkeypatch.setattr(session.ColabSession, "execute", fail)
    monkeypatch.setattr(session.ColabSession, "is_assigned", lambda self: assigned)
    with pytest.raises(RuntimeError, match=message):
        research_mcp.evaluate_controller()
    assert requests[1:] == ["submitted"]
    assert {key for key in objects if "/output/" in key} == outputs
    assert research_mcp.retained_summary is retained
    assert research_mcp.get_episode_trace(0) == trace
    [failed_run] = [path for path in (researcher.parent / "runs").iterdir() if path.name != baseline["run_id"]]
    assert json.loads((failed_run / "evaluation.json").read_text())["status"] == "failed"
    assert not (failed_run / "results").exists()


def test_remote_job_id_cannot_be_started_twice(fake_cli, researcher, tmp_path, monkeypatch):
    _, objects, _ = fake_cli
    session.start_session()
    result = research_mcp.evaluate_controller()
    job_id = result["run_id"]
    staging = tmp_path / "vm"
    (staging / f"{job_id}.json").write_text("{}")
    (staging / f"{job_id}.tar.gz").write_bytes(objects[f"{backend.PREFIX}/{job_id}/input/candidate.tar.gz"])

    monkeypatch.setattr(remote_job, "evaluate", lambda *a: pytest.fail("duplicate evaluation executed"))
    assert remote_job.run_job(job_id, 10, staging) == remote_job.JOB_ALREADY_STARTED
