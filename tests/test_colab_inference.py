"""The optional SLM smoke reuses the assigned session and owner package."""

from contextlib import contextmanager
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsmith.colab import session
from worldsmith.inference import colab_smoke


@pytest.fixture
def model_plan(monkeypatch):
    uploads = []

    def upload(self, action, local, remote):
        assert action == "upload"
        assert remote.endswith("/model-cache.json")
        uploads.append(json.loads(Path(local).read_text()))

    monkeypatch.setattr(colab_smoke, "MODEL_FILES", ("model.safetensors",))
    monkeypatch.setattr(colab_smoke, "R2", lambda: SimpleNamespace(
        head=lambda key: {"ETag": '"cached-etag"'},
        signed_url=lambda *args, **kwargs: "https://example.test/model?private-signature",
    ))
    monkeypatch.setattr(session.ColabSession, "_command", upload)
    return uploads


@pytest.mark.parametrize("cached", [False, True])
def test_model_cache_uses_r2_when_available_and_publishes_missing_files(tmp_path, monkeypatch, cached):
    import sys
    from hashlib import md5

    body = b"model weights"
    plan = tmp_path / "model-cache.json"
    entry = {"get": "r2-url", "etag": md5(body).hexdigest()} if cached else {"put": "r2-url", "headers": {}}
    plan.write_text(json.dumps({"model.safetensors": entry}))
    directory = tmp_path / "hf-cache" / ("models--" + colab_smoke.MODEL.replace("/", "--")) / "snapshots" / colab_smoke.MODEL_REVISION
    calls = []

    def snapshot(model, **kwargs):
        calls.append("hub")
        assert kwargs["revision"] == colab_smoke.MODEL_REVISION
        assert kwargs["allow_patterns"] == ["model.safetensors"]
        (directory / "model.safetensors").write_bytes(body)

    def download(url, destination):
        calls.append("r2")
        destination.write_bytes(body)

    def publish(cache, destination):
        calls.append("publish")
        assert destination.read_bytes() == body

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setattr(sys, "argv", ["model_cache", str(plan)])
    monkeypatch.setattr(runpy, "run_path", lambda path: {
        "download": download, "publish": publish,
        "cache_digest": lambda path: md5(path.read_bytes()),
    })
    runpy.run_module("worldsmith.inference.model_cache", run_name="__main__")
    assert calls == (["r2"] if cached else ["hub", "publish"])
    assert (directory / "model.safetensors").read_bytes() == body
    assert not plan.exists()


@pytest.mark.parametrize("gpu", ["L4", "T4", "A100"])
def test_bootstrap_retains_owner_wheel_for_inference(tmp_path, monkeypatch, gpu):
    main = runpy.run_path(str(session.ROOT / "scripts/setup-colab.py"))["main"]
    namespace = main.__globals__
    (tmp_path / "setup.json").write_text(json.dumps({
        "mode": "research", "gpu": gpu, "wheelhouses": {"pymunk": {"python": "3.12.13"}},
        "assets": {}, "wheel": {"filename": "worldsmith-0.1.0-py3-none-any.whl", "url": "owner-wheel"},
    }))
    installs = []

    def stage(name, *command):
        if name == "pymunk Worldsmith installation":
            installs.append(Path(command[-1]))
        elif name == "pymunk smoke episode":
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "summary.json").write_text("{}")

    monkeypatch.setitem(namespace, "ROOT", tmp_path)
    monkeypatch.setitem(namespace, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setitem(namespace, "platform", SimpleNamespace(machine=lambda: "x86_64"))
    reported_gpu = "NVIDIA A100-SXM4-40GB" if gpu == "A100" else f"Tesla {gpu}"
    monkeypatch.setitem(namespace, "subprocess", SimpleNamespace(check_output=lambda *a, **kw: reported_gpu))
    monkeypatch.setitem(namespace, "install_tools", lambda *a, **kw: {"3.12.13": ("uv", "python")})
    monkeypatch.setitem(namespace, "install_environment", lambda *a, **kw: "pymunk-python")
    monkeypatch.setitem(namespace, "install_assets", lambda *a, **kw: None)
    monkeypatch.setitem(namespace, "download", lambda url, path: path.write_bytes(b"owner wheel"))
    monkeypatch.setitem(namespace, "stage", stage)
    main()
    [wheel] = installs
    assert wheel.parent == tmp_path / "packages"
    assert wheel.read_bytes() == b"owner wheel"


@pytest.mark.parametrize("success", [True, False])
def test_smoke_dispatches_to_assigned_session(monkeypatch, capsys, success, model_plan):
    locks = []

    @contextmanager
    def lock(name):
        locks.append(name)
        yield

    def execute(self, code, *, timeout):
        assert self.name == "existing-session"
        assert timeout == 3600
        assert "private-signature" not in code
        return "chat and tool passed\nWORLDSMITH_SLM_SMOKE_PASSED\n" if success else "remote failure"

    monkeypatch.setattr(colab_smoke, "session_lock", lock)
    monkeypatch.setattr(session, "tracked_session", lambda: {"ready": True, "name": "existing-session"})
    monkeypatch.setattr(session.ColabSession, "execute", execute)
    if success:
        colab_smoke.smoke_test()
        assert "chat and tool passed" in capsys.readouterr().out
    else:
        with pytest.raises(session.ColabError, match="slm-install.log,slm-smoke.log,vllm.log"):
            colab_smoke.smoke_test()
    assert locks == ["lifecycle", "evaluation"]
    assert len(model_plan) == 1


def test_smoke_requires_ready_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "STATE_DIR", tmp_path)
    with pytest.raises(session.ColabError, match="unavailable"):
        colab_smoke.smoke_test()


@pytest.mark.parametrize("environment", [
    "new", "matching", "outdated", "missing dependency", "missing owner wheel",
    "incompatible numpy", "different compatible numpy", "incompatible pymunk",
])
def test_smoke_installs_only_when_pinned_environment_is_not_ready(tmp_path, monkeypatch, environment, model_plan):
    import importlib.metadata
    import subprocess
    from contextlib import nullcontext

    pins = {
        "vllm": "0.10.2", "openai": "1.99.9", "transformers": "4.55.2",
        "worldsmith": "0.1.0", "pymunk": "7.3.1", "numpy": "2.2.6",
    }
    installed = pins.copy()
    if environment == "new":
        installed.clear()
    elif environment == "outdated":
        installed["vllm"] = "0.10.1"
    elif environment == "missing dependency":
        del installed["transformers"]
    elif environment == "missing owner wheel":
        del installed["worldsmith"]
    elif environment == "incompatible numpy":
        installed["numpy"] = "2.4.6"
    elif environment == "different compatible numpy":
        installed["numpy"] = "2.1.3"
    elif environment == "incompatible pymunk":
        installed["pymunk"] = "7.2.0"
    (tmp_path / "tools/3.12.13").mkdir(parents=True)
    (tmp_path / "packages").mkdir()
    (tmp_path / "packages/worldsmith.whl").touch()
    python = tmp_path / ".slm/bin/python"
    if environment != "new":
        python.parent.mkdir(parents=True)
        python.touch()
    installs = []
    smokes = []
    restores = []

    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    def run(command, **kwargs):
        if "venv" in command:
            python.parent.mkdir(parents=True)
            python.touch()
        elif "-c" in command:
            # Run the actual emitted metadata check with controlled installed versions.
            with pytest.raises(SystemExit) as result:
                exec(command[2], {})
            return subprocess.CompletedProcess(command, result.value.code)
        elif "install" in command:
            installs.append(command)
            installed.update(pins)
        elif "worldsmith.inference.model_cache" in command:
            restores.append(command)
            assert command[-1] == str(tmp_path / "model-cache.json")
        elif "worldsmith.inference.robot_smoke" in command:
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            assert kwargs["env"]["HF_HUB_CACHE"] == str(tmp_path / "hf-cache")
            smokes.append(command)
            kwargs["stdout"].write("chat and tool passed\n")
        else:
            pytest.fail(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0)

    def execute(self, code, *, timeout):
        exec(code, {})
        return "WORLDSMITH_SLM_SMOKE_PASSED\n"

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(colab_smoke, "REMOTE_PROJECT", str(tmp_path))
    monkeypatch.setattr(colab_smoke, "session_lock", lambda name: nullcontext())
    monkeypatch.setattr(session, "tracked_session", lambda: {"ready": True, "name": "test-session"})
    monkeypatch.setattr(session.ColabSession, "execute", execute)
    colab_smoke.smoke_test()
    colab_smoke.smoke_test()
    assert len(installs) == (0 if environment == "matching" else 1)
    assert len(smokes) == 2
    assert len(restores) == 2
    if installs:
        assert {
            "vllm==0.10.2", "openai==1.99.9", "transformers==4.55.2",
            "numpy==2.2.6", "pymunk==7.3.1",
        } <= set(installs[0])


@pytest.mark.parametrize("failure", [None, "smoke", "interrupt", "download"])
@pytest.mark.parametrize("gpu", ["L4", "T4", "A100"])
def test_new_session_smoke_always_releases_its_session(tmp_path, monkeypatch, failure, gpu):
    calls = []

    def prepare(selector):
        calls.append(("prepare", selector))
        return 0

    def start(selector, *, gpu):
        calls.append(("start", selector, gpu))
        return SimpleNamespace(name="smoke-session", _command=download)

    def smoke():
        calls.append(("smoke",))
        if failure == "smoke":
            raise session.ColabError("model failed")
        if failure == "interrupt":
            raise SystemExit(130)

    def download(action, remote, local):
        calls.append((action, Path(remote).name))
        assert Path(local).parent == tmp_path / "smoke-session"
        if failure == "download":
            raise session.ColabError("log unavailable")

    monkeypatch.setattr(colab_smoke, "STATE_DIR", tmp_path)
    monkeypatch.setattr(colab_smoke, "build_environments", prepare)
    monkeypatch.setattr(colab_smoke, "start_session", start)
    monkeypatch.setattr(colab_smoke, "smoke_test", smoke)
    monkeypatch.setattr(colab_smoke, "stop_tracked_session", lambda *, expected_name: calls.append(("stop", expected_name)))
    if failure == "smoke":
        with pytest.raises(session.ColabError, match="model failed"):
            colab_smoke.run_session_smoke(gpu=gpu)
    elif failure == "interrupt":
        with pytest.raises(SystemExit) as error:
            colab_smoke.run_session_smoke(gpu=gpu)
        assert error.value.code == 130
    else:
        assert colab_smoke.run_session_smoke(gpu=gpu) == 0
    assert calls == [
        ("prepare", "pymunk"), ("start", "pymunk", gpu), ("smoke",),
        ("download", "slm-install.log"), ("download", "slm-smoke.log"), ("download", "vllm.log"),
        ("stop", "smoke-session"),
    ]


def test_preparation_failure_does_not_start_session(monkeypatch):
    monkeypatch.setattr(colab_smoke, "build_environments", lambda selector: 1)
    monkeypatch.setattr(colab_smoke, "start_session", lambda selector: pytest.fail("started after preparation failure"))
    assert colab_smoke.run_session_smoke() == 1


def test_start_failure_does_not_stop_existing_session(monkeypatch):
    def start(selector, *, gpu):
        raise session.ColabError("already tracked")

    monkeypatch.setattr(colab_smoke, "build_environments", lambda selector: 0)
    monkeypatch.setattr(colab_smoke, "start_session", start)
    monkeypatch.setattr(colab_smoke, "stop_tracked_session", lambda **kw: pytest.fail("stopped another session"))
    with pytest.raises(session.ColabError, match="already tracked"):
        colab_smoke.run_session_smoke()


@pytest.mark.parametrize("gpu", ["L4", "T4", "A100"])
def test_cli_dispatches_new_session_smoke(monkeypatch, gpu):
    from worldsmith.colab import __main__ as owner

    monkeypatch.setattr(owner.sys, "argv", ["worldsmith.colab", "slm-smoke", "--new-session", "--gpu", gpu])
    monkeypatch.setattr(owner.signal, "signal", lambda *args: None)

    def run(*, gpu):
        assert gpu == owner.sys.argv[-1]
        return 1

    monkeypatch.setattr(colab_smoke, "run_session_smoke", run)
    assert owner.main() == 1
