"""AutoResearch startup, research bootstrap, and owner process lifecycle."""

import json
import os
import signal
import sys
import tarfile
from textwrap import dedent

import pytest

from test_colab_remote import fake_cli
from worldsmith import research_launcher as launcher, research_lifecycle as workspace
from worldsmith.colab import __main__ as owner
from worldsmith.colab import environment_cache, session, storage


def test_research_requires_prepared_caches_before_allocating(fake_cli, monkeypatch):
    calls, _, _ = fake_cli
    monkeypatch.setattr(session.R2, "head", lambda self, key: None)
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **kw: pytest.fail("built Worldsmith wheel"))
    with pytest.raises(storage.StorageError, match="Run make colab-build-envs"):
        session.start_session("pymunk")
    assert calls == []
    assert session.tracked_session() is None


@pytest.mark.parametrize("selector", ["all", "maniskill"])
def test_research_bootstrap_has_only_downloads_for_selected_simulators(fake_cli, monkeypatch, selector):
    initialize = session.initialize

    def check_bootstrap(runtime, archive, *, marker):
        with tarfile.open(archive) as bundle:
            config = json.load(bundle.extractfile("setup.json"))
        assert config["mode"] == "research"
        assert config["wheel"]
        expected_wheels = set(environment_cache.ENVIRONMENT_PYTHONS) if selector == "all" else {"maniskill"}
        assert set(config["wheelhouses"]) == expected_wheels
        assert set(config["assets"]) == expected_wheels & {"libero", "maniskill"}
        for family in ("tools", "wheelhouses", "assets"):
            assert all("get" in entry and "put" not in entry for entry in config[family].values())
        initialize(runtime, archive, marker=marker)

    monkeypatch.setattr(session, "initialize", check_bootstrap)
    session.start_session(selector).stop()


def test_failed_initialization_stops_session_without_launching_research(fake_cli, monkeypatch, capsys):
    calls, _, _ = fake_cli
    monkeypatch.delenv("WORLDSMITH_EVALUATION_BACKEND", raising=False)
    monkeypatch.setattr(session.ColabSession, "execute", lambda *a, **kw: "WORLDSMITH_SETUP GPU mismatch")
    monkeypatch.setattr(launcher, "run_workspace", lambda *a: pytest.fail("research started"))
    assert launcher.main([]) == 1
    assert "GPU mismatch" in capsys.readouterr().err
    assert [action for _, action, _ in calls] == ["new", "upload", "stop"]
    assert session.tracked_session() is None


def test_colab_start_prints_session_and_next_commands(fake_cli, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["worldsmith.colab", "start", "maniskill"])
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    assert owner.main() == 0
    state = session.tracked_session()
    assert state["ready"]
    output = capsys.readouterr().out
    assert state["name"] in output
    assert "make colab-status" in output
    assert "make colab-stop" in output


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    research = tmp_path / "autoresearch"
    research.mkdir()
    history = tmp_path / "research-history"
    monkeypatch.setattr(workspace, "WORKSPACE", research)
    monkeypatch.setattr(workspace, "HISTORY", history)
    monkeypatch.delenv("WORLDSMITH_COLAB_SESSION", raising=False)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    codex = tmp_path / "codex"
    codex.write_text(f"#!{sys.executable}\n" + dedent("""\
        import json
        import os
        import signal
        import sys
        from pathlib import Path

        assert Path("controller.py").is_file()
        assert sys.argv[1:-1] == ["exec", "-c", 'model_reasoning_effort="low"', "--sandbox", "workspace-write"]
        Path("JOURNAL.md").write_text("research notes")
        Path("launch.json").write_text(json.dumps({
            "suite": os.environ["WORLDSMITH_RESEARCH_SUITE"],
            "backend": os.environ["WORLDSMITH_EVALUATION_BACKEND"],
            "session": os.environ.get("WORLDSMITH_COLAB_SESSION"),
            "prompt": sys.argv[-1],
        }))
        if os.environ["CODEX_STATUS"] == "interrupt":
            def stop(signum, frame):
                Path("interrupted").touch()
                sys.exit(0)

            signal.signal(signal.SIGTERM, stop)
            os.kill(os.getppid(), signal.SIGTERM)
            while True:
                signal.pause()
        sys.exit(int(os.environ["CODEX_STATUS"]))
        """))
    codex.chmod(0o700)
    return research, history


@pytest.mark.parametrize("backend, codex_status, cleanup_fails, expected", [
    ("local", "0", False, 0),
    ("local", "7", False, 7),
    ("local", "interrupt", False, 143),
    ("colab", "0", False, 0),
    ("colab", "7", False, 7),
    ("colab", "interrupt", False, 143),
    ("colab", "7", True, 7),
    ("colab", "0", True, 1),
])
def test_launcher_lifecycle(fake_cli, fake_codex, monkeypatch, capsys, backend, codex_status, cleanup_fails, expected):
    calls, _, _ = fake_cli
    research, history = fake_codex
    monkeypatch.setenv("CODEX_STATUS", codex_status)
    if backend == "colab":
        monkeypatch.delenv("WORLDSMITH_EVALUATION_BACKEND", raising=False)
    else:
        monkeypatch.setenv("WORLDSMITH_EVALUATION_BACKEND", backend)
    if cleanup_fails:
        def fail_stop(self):
            raise session.ColabError("Could not stop Colab")

        monkeypatch.setattr(session.ColabSession, "stop", fail_stop)

    assert launcher.main(["maniskill"]) == expected

    [archive] = history.iterdir()
    assert (archive / ".complete").is_file()
    saved = archive / "workspace"
    assert (saved / "JOURNAL.md").read_text() == "research notes"
    launch = json.loads((saved / "launch.json").read_text())
    assert launch["suite"] == "maniskill"
    assert launch["backend"] == backend
    assert "tabletop manipulation and quadruped locomotion" in launch["prompt"]
    if codex_status == "0":
        assert (research / "JOURNAL.md").read_bytes() == (workspace.BASELINE / "JOURNAL.md").read_bytes()
        assert not (research / "launch.json").exists()
    else:
        assert (research / "JOURNAL.md").read_text() == "research notes"
    if codex_status == "interrupt":
        assert (saved / "interrupted").is_file()

    if backend == "local":
        assert calls == []
        assert launch["session"] is None
    else:
        assert {name for name, _, _ in calls} == {launch["session"]}
        assert [action for _, action, _ in calls] == ["new", "upload", "exec"] + ([] if cleanup_fails else ["stop"])
    if cleanup_fails:
        assert "Retry cleanup with make colab-stop" in capsys.readouterr().err
        assert session.tracked_session() == {"name": launch["session"], "ready": True}
    else:
        assert session.tracked_session() is None


@pytest.mark.parametrize("codex_status, expected", [("0", 1), ("7", 7)])
def test_archive_failure_keeps_workspace_and_stops_colab(fake_cli, fake_codex, monkeypatch, capsys, codex_status, expected):
    research, _ = fake_codex
    monkeypatch.setenv("CODEX_STATUS", codex_status)
    monkeypatch.setenv("WORLDSMITH_EVALUATION_BACKEND", "colab")

    def fail_archive(*args):
        raise OSError("Could not write archive")

    monkeypatch.setattr(workspace, "archive_session", fail_archive)
    assert launcher.main(["maniskill"]) == expected
    assert "Could not archive the research session" in capsys.readouterr().err
    assert (research / "JOURNAL.md").read_text() == "research notes"
    assert session.tracked_session() is None
