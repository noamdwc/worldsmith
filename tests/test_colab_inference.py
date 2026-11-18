"""The optional SLM smoke reuses the assigned session and owner package."""

from contextlib import contextmanager
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsmith.colab import session
from worldsmith.inference import colab_smoke


def test_bootstrap_retains_owner_wheel_for_inference(tmp_path, monkeypatch):
    main = runpy.run_path(str(session.ROOT / "scripts/setup-colab.py"))["main"]
    namespace = main.__globals__
    (tmp_path / "setup.json").write_text(json.dumps({
        "mode": "research", "wheelhouses": {"pymunk": {"python": "3.12.13"}},
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
    monkeypatch.setitem(namespace, "subprocess", SimpleNamespace(check_output=lambda *a, **kw: "NVIDIA L4"))
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
        compile(code, "colab-slm", "exec")
        assert "'.slm/bin/python'" in code
        assert "'packages'" in code
        assert "worldsmith.inference.smoke" in code
        assert "vllm==0.10.2" in code
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


@pytest.mark.parametrize("failure", [None, "smoke", "interrupt", "download"])
def test_new_session_smoke_always_releases_its_session(tmp_path, monkeypatch, failure):
    calls = []

    def prepare(selector):
        calls.append(("prepare", selector))
        return 0

    def start(selector):
        calls.append(("start", selector))
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
            colab_smoke.run_session_smoke()
    elif failure == "interrupt":
        with pytest.raises(SystemExit) as error:
            colab_smoke.run_session_smoke()
        assert error.value.code == 130
    else:
        assert colab_smoke.run_session_smoke() == 0
    assert calls == [
        ("prepare", "pymunk"), ("start", "pymunk"), ("smoke",),
        ("download", "slm-install.log"), ("download", "slm-smoke.log"), ("download", "vllm.log"),
        ("stop", "smoke-session"),
    ]


def test_preparation_failure_does_not_start_session(monkeypatch):
    monkeypatch.setattr(colab_smoke, "build_environments", lambda selector: 1)
    monkeypatch.setattr(colab_smoke, "start_session", lambda selector: pytest.fail("started after preparation failure"))
    assert colab_smoke.run_session_smoke() == 1


def test_start_failure_does_not_stop_existing_session(monkeypatch):
    def start(selector):
        raise session.ColabError("already tracked")

    monkeypatch.setattr(colab_smoke, "build_environments", lambda selector: 0)
    monkeypatch.setattr(colab_smoke, "start_session", start)
    monkeypatch.setattr(colab_smoke, "stop_tracked_session", lambda **kw: pytest.fail("stopped another session"))
    with pytest.raises(session.ColabError, match="already tracked"):
        colab_smoke.run_session_smoke()


def test_cli_dispatches_new_session_smoke(monkeypatch):
    from worldsmith.colab import __main__ as owner

    monkeypatch.setattr(owner.sys, "argv", ["worldsmith.colab", "slm-smoke", "--new-session"])
    monkeypatch.setattr(owner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(colab_smoke, "run_session_smoke", lambda: 1)
    assert owner.main() == 1
