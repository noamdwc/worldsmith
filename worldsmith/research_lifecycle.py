"""Owner commands for starting, archiving, and resetting AutoResearch."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "autoresearch"
HISTORY = ROOT / "research-history"
BASELINE = Path(__file__).with_name("research_baseline")
GUIDES = {".codex", "AGENTS.md", "API.md", "RESEARCH.md", "TASKS.md"}
BASELINE_FILES = ("controller.py", "JOURNAL.md")


def research_entries(workspace: Path) -> list[Path]:
    if workspace.is_symlink():
        raise ValueError("The research workspace must be a local directory, not a symlink.")
    return sorted(path for path in workspace.iterdir() if path.name not in GUIDES)


def copy_entry(source: Path, target: Path, *, skip_cache: bool = False):
    if source.is_symlink():
        raise ValueError(f"Cannot copy a symlink: {source}")
    if source.is_dir():
        target.mkdir()
        for child in source.iterdir():
            if not (skip_cache and child.name == "__pycache__"):
                copy_entry(child, target / child.name, skip_cache=skip_cache)
        shutil.copystat(source, target)
    elif source.is_file():
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported research entry: {source}")


def is_archived(current: Path, saved: Path) -> bool:
    if current.is_symlink() or saved.is_symlink():
        return False
    if current.is_dir():
        return (
            saved.is_dir() and current.stat().st_mode == saved.stat().st_mode
            and all(is_archived(child, saved / child.name) for child in current.iterdir())
        )
    return (
        current.is_file() and saved.is_file()
        and current.read_bytes() == saved.read_bytes()
        and current.stat().st_mode == saved.stat().st_mode
    )


def start_session(workspace: Path):
    for entry in research_entries(workspace):
        if entry.name not in BASELINE_FILES or not is_archived(entry, BASELINE / entry.name):
            raise RuntimeError("Workspace contains research work. Archive and reset it before starting.")
    for name in BASELINE_FILES:
        if not (workspace / name).exists():
            shutil.copy2(BASELINE / name, workspace / name)


def archive_session(workspace: Path, history: Path) -> str:
    archive_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
    archive = history / archive_id
    archive.mkdir(parents=True)
    saved_workspace = archive / "workspace"
    saved_workspace.mkdir()
    for entry in research_entries(workspace):
        copy_entry(entry, saved_workspace / entry.name)
    if not all(is_archived(entry, saved_workspace / entry.name) for entry in research_entries(workspace)):
        raise RuntimeError("Workspace changed during archiving. Stop the researcher and retry.")
    # An interrupted copy leaves an inspectable folder that reset will refuse.
    (archive / ".complete").touch()
    return archive_id


def reset_workspace(workspace: Path, history: Path, archive_id: str):
    if not archive_id or not all(c.isascii() and (c.isalnum() or c == "-") for c in archive_id):
        raise ValueError("Invalid archive ID.")
    archive = history / archive_id
    saved_workspace = archive / "workspace"
    if (
        archive.is_symlink() or saved_workspace.is_symlink()
        or not saved_workspace.is_dir() or not (archive / ".complete").is_file()
    ):
        raise RuntimeError("A completed archive is required before resetting.")
    entries = research_entries(workspace)
    # A subset is safe too, so an interrupted reset can be retried.
    if not all(is_archived(entry, saved_workspace / entry.name) for entry in entries):
        raise RuntimeError("Workspace differs from the archive. Archive the current workspace first.")
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    start_session(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start", help="Initialize the original baseline in a clean workspace")
    commands.add_parser("archive", help="Copy the current research files to history")
    reset = commands.add_parser("reset", help="Restore the original baseline after archiving")
    reset.add_argument("archive_id")
    args = parser.parse_args()
    try:
        if args.command == "start":
            start_session(WORKSPACE)
        elif args.command == "archive":
            print(archive_session(WORKSPACE, HISTORY))
        else:
            reset_workspace(WORKSPACE, HISTORY, args.archive_id)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
