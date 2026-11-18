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
