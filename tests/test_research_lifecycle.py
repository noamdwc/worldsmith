import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from worldsmith import research_lifecycle as lifecycle


@pytest.fixture
def workspace(tmp_path):
    research = tmp_path / "autoresearch"
    research.mkdir()
    (research / "API.md").write_text("public API")
    return research, tmp_path / "research-history"


def test_start_archive_and_reset(workspace):
    research, history = workspace
    lifecycle.start_session(research)
    assert (research / "controller.py").read_bytes() == (lifecycle.BASELINE / "controller.py").read_bytes()
    files = {
        "JOURNAL.md": "session notes",
        "controller.py": "final controller",
        "skill.py": "final skill",
        "runs/mcp-first/code/autoresearch/controller.py": "candidate",
        "runs/mcp-first/evaluation.json": "status",
        "runs/mcp-first/results/summary.json": "metrics",
        "runs/mcp-first/results/episodes.jsonl": "traces",
    }
    for name, content in files.items():
        path = research / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    archive_id = lifecycle.archive_session(research, history)
    archive = history / archive_id / "workspace"
    metadata = json.loads((archive.parent / "metadata.json").read_text())
    assert metadata["git_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=lifecycle.ROOT, text=True,
    ).strip()
    archived_at = datetime.fromisoformat(metadata["archived_at"])
    assert archived_at.tzinfo == UTC
    assert archive_id.startswith(f"{archived_at:%Y%m%dT%H%M%SZ}-")
    assert lifecycle.archive_session(research, history) != archive_id

    lifecycle.reset_workspace(research, history, archive_id)

    for name, content in files.items():
        assert (archive / name).read_text() == content
    for name in lifecycle.BASELINE_FILES:
        assert (research / name).read_bytes() == (lifecycle.BASELINE / name).read_bytes()
    assert not (research / "runs").exists()
    assert not (research / "skill.py").exists()
    assert (research / "API.md").read_text() == "public API"


def test_start_and_reset_refuse_unarchived_changes(workspace):
    research, history = workspace
    lifecycle.start_session(research)
    journal = research / "JOURNAL.md"
    journal.write_text("old session notes")
    with pytest.raises(RuntimeError, match="contains research work"):
        lifecycle.start_session(research)
    archive_id = lifecycle.archive_session(research, history)
    journal.write_text("new session notes")

    with pytest.raises(RuntimeError, match="differs from the archive"):
        lifecycle.reset_workspace(research, history, archive_id)
    assert journal.read_text() == "new session notes"
    assert (research / "controller.py").is_file()


def test_reset_refuses_missing_archive_file(workspace):
    research, history = workspace
    lifecycle.start_session(research)
    archive_id = lifecycle.archive_session(research, history)
    (history / archive_id / "workspace/JOURNAL.md").unlink()

    with pytest.raises(RuntimeError, match="differs from the archive"):
        lifecycle.reset_workspace(research, history, archive_id)
    assert (research / "JOURNAL.md").is_file()


def test_interrupted_archive_cannot_be_used_to_reset(workspace, monkeypatch):
    research, history = workspace
    lifecycle.start_session(research)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(lifecycle, "copy_entry", interrupt)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.archive_session(research, history)
    [archive] = history.iterdir()
    with pytest.raises(RuntimeError, match="completed archive"):
        lifecycle.reset_workspace(research, history, archive.name)
    assert (research / "JOURNAL.md").is_file()


def test_interrupted_reset_can_retry_from_same_archive(workspace, monkeypatch):
    research, history = workspace
    lifecycle.start_session(research)
    (research / "JOURNAL.md").write_text("keep this")
    archive_id = lifecycle.archive_session(research, history)
    unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        if path == research / "controller.py":
            raise KeyboardInterrupt
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(KeyboardInterrupt):
            lifecycle.reset_workspace(research, history, archive_id)
    assert not (research / "JOURNAL.md").exists()
    lifecycle.reset_workspace(research, history, archive_id)
    assert (history / archive_id / "workspace/JOURNAL.md").read_text() == "keep this"
    assert (research / "JOURNAL.md").read_bytes() == (lifecycle.BASELINE / "JOURNAL.md").read_bytes()
