"""The optional SLM smoke reuses the assigned session and owner package."""

from contextlib import contextmanager
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsmith.colab import session
from worldsmith.inference import colab_smoke


@pytest.mark.parametrize("gpu", ["L4", "T4"])
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
    monkeypatch.setitem(namespace, "subprocess", SimpleNamespace(check_output=lambda *a, **kw: f"Tesla {gpu}"))
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
def test_smoke_dispatches_to_assigned_session(monkeypatch, capsys, success):
    locks = []

    @contextmanager
    def lock(name):
        locks.append(name)
        yield

    def execute(self, code, *, timeout):
        assert self.name == "existing-session"
        assert timeout == 3600
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


def test_smoke_requires_ready_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "STATE_DIR", tmp_path)
    with pytest.raises(session.ColabError, match="unavailable"):
        colab_smoke.smoke_test()


@pytest.mark.parametrize("environment", ["new", "matching", "outdated", "missing dependency", "missing owner wheel"])
def test_smoke_installs_only_when_pinned_environment_is_not_ready(tmp_path, monkeypatch, environment):
    import importlib.metadata
    import subprocess
    from contextlib import nullcontext

    pins = {"vllm": "0.10.2", "openai": "1.99.9", "transformers": "4.55.2", "worldsmith": "0.1.0"}
    installed = pins.copy()
    if environment == "new":
        installed.clear()
    elif environment == "outdated":
        installed["vllm"] = "0.10.1"
    elif environment == "missing dependency":
        del installed["transformers"]
    elif environment == "missing owner wheel":
        del installed["worldsmith"]
    (tmp_path / "tools/3.12.13").mkdir(parents=True)
    (tmp_path / "packages").mkdir()
    (tmp_path / "packages/worldsmith.whl").touch()
    python = tmp_path / ".slm/bin/python"
    if environment != "new":
        python.parent.mkdir(parents=True)
        python.touch()
    installs = []
    smokes = []

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
        elif "worldsmith.inference.smoke" in command:
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
    if installs:
        assert {"vllm==0.10.2", "openai==1.99.9", "transformers==4.55.2"} <= set(installs[0])


@pytest.mark.parametrize("failure", [None, "smoke", "interrupt", "download"])
@pytest.mark.parametrize("gpu", ["L4", "T4"])
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


@pytest.mark.parametrize("gpu", ["L4", "T4"])
def test_cli_dispatches_new_session_smoke(monkeypatch, gpu):
    from worldsmith.colab import __main__ as owner

    monkeypatch.setattr(owner.sys, "argv", ["worldsmith.colab", "slm-smoke", "--new-session", "--gpu", gpu])
    monkeypatch.setattr(owner.signal, "signal", lambda *args: None)

    def run(*, gpu):
        assert gpu == owner.sys.argv[-1]
        return 1

    monkeypatch.setattr(colab_smoke, "run_session_smoke", run)
    assert owner.main() == 1
