"""AutoResearch startup, research bootstrap, and owner process lifecycle."""

import json
import signal
import tarfile

import pytest

from test_colab_remote import fake_cli
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


def test_failed_initialization_stops_session_without_launching_research(fake_cli, monkeypatch):
    calls, _, _ = fake_cli
    monkeypatch.setattr(session.ColabSession, "execute", lambda *a, **kw: "WORLDSMITH_SETUP GPU mismatch")
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **kw: pytest.fail("research started"))
    with pytest.raises(session.ColabError, match="GPU mismatch"):
        owner.run_research("all")
    assert [action for _, action, _ in calls] == ["new", "upload", "stop"]
    assert session.tracked_session() is None


@pytest.mark.parametrize("status, interrupt, cleanup_fails", [
    (0, False, False), (7, False, False), (7, False, True), (0, False, True), (0, True, False),
])
def test_owner_shutdown_preserves_exit_status(fake_cli, tmp_path, monkeypatch, capsys, status, interrupt, cleanup_fails):
    calls, _, _ = fake_cli
    real_start = owner.start_session
    def start(selector):
        runtime = real_start(selector)
        if cleanup_fails:
            def fail_stop(self):
                raise session.ColabError("Could not stop Colab")
            monkeypatch.setattr(session.ColabSession, "stop", fail_stop)
        return runtime
    monkeypatch.setattr(owner, "start_session", start)
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/run-autoresearch.sh"
    if interrupt:
        script.write_text('#!/bin/sh\nkill -TERM "$PPID"\nsleep 30\n')
    else:
        script.write_text(f'#!/bin/sh\ntest -n "$WORLDSMITH_COLAB_SESSION" || exit 99\nexit {status}\n')
    script.chmod(0o700)
    monkeypatch.setattr(owner, "ROOT", root)
    if interrupt:
        previous = signal.signal(signal.SIGTERM, owner.handle_interrupt)
        try:
            with pytest.raises(SystemExit) as exit_info:
                owner.run_research("all")
            assert exit_info.value.code == 143
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        assert owner.run_research("all") == (status or int(cleanup_fails))
    if cleanup_fails:
        assert "Retry cleanup with make colab-stop" in capsys.readouterr().err
        assert session.tracked_session()["ready"]
    else:
        assert calls[-1][1] == "stop"
        assert session.tracked_session() is None
