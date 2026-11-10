import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pytest

from worldsmith import research_lifecycle as lifecycle


@pytest.fixture
def workspace(tmp_path):
    workspace = tmp_path / "autoresearch"
    workspace.mkdir()
    for name in ("controller.py", "JOURNAL.md"):
        shutil.copyfile(lifecycle.BASELINE / name, workspace / name)
    (workspace / "API.md").write_text("public API")
    return workspace, tmp_path / "research-history"


def test_start_archive_reset_and_start_from_candidate(workspace):
    research, history = workspace
    session_id = lifecycle.start_session(research, history)
    assert lifecycle.state(research)["session_id"] == session_id
    (research / "JOURNAL.md").write_text("session results")
    candidate = research / "runs/mcp-first/code/autoresearch"
    candidate.mkdir(parents=True)
    (candidate / "controller.py").write_text("candidate")
    (candidate / "skill.py").write_text("candidate skill")
    (research / "controller.py").write_text("final controller")
    (research / "skill.py").write_text("final skill")
    (research / "notes.txt").write_text("temporary research notes")

    archive_id = lifecycle.archive_session(research, history)
    archive_path = history / f"{archive_id}.zip"
    assert not archive_path.stat().st_mode & 0o222
    with ZipFile(archive_path) as archive:
        assert archive.read("workspace/JOURNAL.md") == b"session results"
        assert archive.read("workspace/controller.py") == b"final controller"
        assert archive.read("workspace/skill.py") == b"final skill"
        assert archive.read("workspace/notes.txt") == b"temporary research notes"

    lifecycle.reset_workspace(research, history, archive_id)
    assert (research / "controller.py").read_bytes() == (lifecycle.BASELINE / "controller.py").read_bytes()
    assert not (research / "runs").exists()
    assert not (research / "notes.txt").exists()
    assert (research / "API.md").read_text() == "public API"
    assert archive_path.is_file()

    source = f"{archive_id}/mcp-first"
    assert lifecycle.start_session(research, history, source) != session_id
    assert (research / "controller.py").read_text() == "candidate"
    assert (research / "skill.py").read_text() == "candidate skill"
    (research / "skill.py").write_text("new skill")
    second_archive = lifecycle.archive_session(research, history)
    lifecycle.reset_workspace(research, history, second_archive)
    assert (research / "skill.py").read_text() == "candidate skill"


def test_start_and_reset_refuse_unarchived_changes(workspace):
    research, history = workspace
    (research / "JOURNAL.md").write_text("old session notes")
    with pytest.raises(RuntimeError, match="contains research work"):
        lifecycle.start_session(research, history)
    archive_id = lifecycle.archive_session(research, history)

    notes = research / "new-notes.txt"
    notes.write_text("written after archiving")
    with pytest.raises(RuntimeError, match="differs from the archive"):
        lifecycle.reset_workspace(research, history, archive_id)
    assert notes.read_text() == "written after archiving"
    assert (research / "JOURNAL.md").read_text() == "old session notes"


def test_reset_refuses_corrupted_archive(workspace):
    research, history = workspace
    (research / "JOURNAL.md").write_text("keep this")
    archive_id = lifecycle.archive_session(research, history)
    archive = history / f"{archive_id}.zip"
    archive.chmod(0o644)
    archive.write_bytes(b"not a zip")

    with pytest.raises(BadZipFile):
        lifecycle.reset_workspace(research, history, archive_id)
    assert (research / "JOURNAL.md").read_text() == "keep this"


def test_interrupted_reset_restores_workspace(workspace, monkeypatch):
    research, history = workspace
    (research / "JOURNAL.md").write_text("keep this")
    archive_id = lifecycle.archive_session(research, history)
    rename = Path.rename

    def interrupt(source, destination):
        if destination == research and source.name.startswith("tmp"):
            raise KeyboardInterrupt
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.reset_workspace(research, history, archive_id)
    assert (research / "JOURNAL.md").read_text() == "keep this"
    assert (history / f"{archive_id}.zip").is_file()


def test_interrupted_archive_is_not_published(workspace, monkeypatch):
    research, history = workspace
    (research / "JOURNAL.md").write_text("keep this")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ZipFile, "write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.archive_session(research, history)
    assert not list(history.glob("*.zip"))
    assert (research / "JOURNAL.md").read_text() == "keep this"


def test_lifecycle_refuses_to_archive_during_evaluation(workspace):
    research, history = workspace
    with lifecycle.workspace_lock(history):
        with pytest.raises(RuntimeError, match="Another evaluation or lifecycle command is active"):
            lifecycle.archive_session(research, history)
    assert (research / "controller.py").is_file()
