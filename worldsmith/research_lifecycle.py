"""Owner commands for local AutoResearch sessions and verified archives."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "autoresearch"
HISTORY = ROOT / "research-history"
BASELINE = Path(__file__).with_name("research_baseline")
GUIDES = {"AGENTS.md", "API.md", "RESEARCH.md", "TASKS.md", ".codex", "__init__.py"}
NON_CODE = GUIDES | {"runs", "JOURNAL.md", ".session.json"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def workspace_lock(history: Path):
    history.mkdir(parents=True, exist_ok=True)
    with (history / ".lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation or lifecycle command is active.") from None
        yield lock.fileno()


def tree(root: Path, exclude: set[str] = frozenset(), *, skip_cache: bool = False):
    for path in sorted(root.iterdir()):
        if path.name in exclude or (skip_cache and path.name == "__pycache__"):
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unsupported workspace entry: {path}")
        yield path
        if path.is_dir():
            yield from tree(path, skip_cache=skip_cache)


def manifest(root: Path) -> dict[str, str | None]:
    entries = {}
    for path in tree(root):
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            entries[name] = None
        else:
            with path.open("rb") as content:
                entries[name] = hashlib.file_digest(content, "sha256").hexdigest()
    return entries


def snapshot_code(workspace: Path, destination: Path):
    destination.mkdir(parents=True)
    # A regular package prevents imports from falling back to the editable workspace.
    (destination / "__init__.py").touch()
    for path in tree(workspace, NON_CODE, skip_cache=True):
        target = destination / path.relative_to(workspace)
        if path.is_dir():
            target.mkdir()
        else:
            shutil.copy(path, target)
    if not (destination / "controller.py").is_file():
        raise ValueError("Workspace has no controller.py to evaluate.")


def state(workspace: Path) -> dict:
    path = workspace / ".session.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def archive_path(history: Path, archive_id: str) -> Path:
    if not archive_id or not all(c.isascii() and (c.isalnum() or c == "-") for c in archive_id):
        raise ValueError("Invalid archive or evaluation ID.")
    return history / f"{archive_id}.zip"


def verified_manifest(archive: ZipFile) -> dict:
    expected = json.loads(archive.read("manifest.json"))
    names = set(archive.namelist())
    if len(names) != len(archive.namelist()) or names != {
        "manifest.json", *(name + "/" if digest is None else name for name, digest in expected.items())
    }:
        raise ValueError("Archive contents do not match its manifest.")
    for name, digest in expected.items():
        if digest is not None and hashlib.sha256(archive.read(name)).hexdigest() != digest:
            raise ValueError(f"Archive verification failed: {name}")
    return expected


def archive_session(workspace: Path, history: Path, root: Path = ROOT) -> str:
    archive_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
    with workspace_lock(history), TemporaryDirectory(dir=history) as temporary:
        before = manifest(workspace)
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "worldsmith", "pyproject.toml", "uv.lock"],
            cwd=root, text=True,
        )
        metadata = json.dumps({
            "archive_id": archive_id,
            "archived_at": now(),
            "session": state(workspace),
            "git_revision": git,
            "harness_changes": changes.splitlines(),
            "python": sys.version,
        }, indent=2).encode()
        lockfile = (root / "uv.lock").read_bytes()
        staged = Path(temporary) / "archive.zip"
        expected = {f"workspace/{name}": digest for name, digest in before.items()}
        with ZipFile(staged, "w", ZIP_DEFLATED) as archive:
            for name in before:
                archive.write(workspace / name, f"workspace/{name}")
            for name, data in (("session.json", metadata), ("uv.lock", lockfile)):
                archive.writestr(name, data)
                expected[name] = hashlib.sha256(data).hexdigest()
            archive.writestr("manifest.json", json.dumps(expected, indent=2))
        with ZipFile(staged) as archive:
            verified_manifest(archive)
        if manifest(workspace) != before:
            raise RuntimeError("Workspace changed during archiving. Stop the researcher and retry.")
        staged.chmod(0o444)
        with staged.open("rb") as content:
            os.fsync(content.fileno())
        archive_path(history, archive_id).hardlink_to(staged)
        directory = os.open(history, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return archive_id


def populate_baseline(destination: Path, history: Path, source: str):
    if source == "original":
        shutil.copyfile(BASELINE / "controller.py", destination / "controller.py")
    else:
        archive_id, separator, run_id = source.partition("/")
        prefix = "workspace/"
        if separator:
            archive_path(history, run_id)  # Validate the evaluation ID before using it as a path.
            prefix += f"runs/{run_id}/code/autoresearch/"
        with ZipFile(archive_path(history, archive_id)) as archive:
            entries = verified_manifest(archive)
            if entries.get(prefix + "controller.py") is None:
                raise ValueError("The selected archive or evaluation has no controller snapshot.")
            for name, digest in entries.items():
                if not name.startswith(prefix):
                    continue
                relative = PurePosixPath(name[len(prefix):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid archive member: {name}")
                if not relative.parts or relative.parts[0] in NON_CODE or "__pycache__" in relative.parts:
                    continue
                target = destination / relative
                if digest is None:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
                    target.chmod((archive.getinfo(name).external_attr >> 16) & 0o777)
    shutil.copyfile(BASELINE / "JOURNAL.md", destination / "JOURNAL.md")


def replace_workspace(workspace: Path, replacement: Path, history: Path, expected: dict):
    # Keep the old tree in a recoverable location until the new tree is installed.
    backup = history / f".reset-backup-{uuid4().hex}"
    workspace.rename(backup)
    try:
        if manifest(backup) != expected:
            raise RuntimeError("Workspace changed during replacement. Nothing was discarded.")
        replacement.rename(workspace)
    except BaseException:
        if not workspace.exists():
            backup.rename(workspace)
        raise
    shutil.rmtree(backup)


def prepare_workspace(workspace: Path, destination: Path, history: Path, source: str):
    populate_baseline(destination, history, source)
    for name in GUIDES:
        guide = workspace / name
        if guide.is_dir():
            shutil.copytree(guide, destination / name)
        elif guide.is_file():
            shutil.copyfile(guide, destination / name)


def start_session(workspace: Path, history: Path, source: str = "original") -> str:
    with workspace_lock(history), TemporaryDirectory(dir=history) as temporary:
        current = state(workspace)
        previous = Path(temporary) / "previous"
        previous.mkdir()
        populate_baseline(previous, history, current.get("baseline", "original"))
        before = manifest(workspace)
        clean = {
            name: digest for name, digest in before.items()
            if name.split("/", 1)[0] not in GUIDES | {".session.json"}
        }
        if current.get("session_id") or clean != manifest(previous):
            raise RuntimeError("Workspace contains research work. Archive and reset it before starting.")
        replacement = Path(temporary) / "next"
        replacement.mkdir()
        prepare_workspace(workspace, replacement, history, source)
        session_id = uuid4().hex
        write_json(replacement / ".session.json", {
            "session_id": session_id, "baseline": source, "started_at": now(),
        })
        replace_workspace(workspace, replacement, history, before)
    return session_id


def reset_workspace(workspace: Path, history: Path, archive_id: str):
    with workspace_lock(history), TemporaryDirectory(dir=history) as temporary:
        with ZipFile(archive_path(history, archive_id)) as archive:
            entries = verified_manifest(archive)
            expected = {
                name.removeprefix("workspace/"): digest for name, digest in entries.items()
                if name.startswith("workspace/")
            }
            source = json.loads(archive.read("session.json"))["session"].get("baseline", "original")
        if manifest(workspace) != expected:
            raise RuntimeError("Workspace differs from the archive. Archive the current workspace first.")
        replacement = Path(temporary)
        prepare_workspace(workspace, replacement, history, source)
        if source != "original":
            write_json(replacement / ".session.json", {"baseline": source})
        replace_workspace(workspace, replacement, history, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Initialize a clean research session")
    start.add_argument("--from", dest="source", default="original", metavar="ARCHIVE[/RUN]")
    commands.add_parser("archive", help="Save and verify the current workspace")
    reset = commands.add_parser("reset", help="Restore the starting baseline after verifying an archive")
    reset.add_argument("archive_id")
    args = parser.parse_args()
    try:
        if args.command == "start":
            print(start_session(WORKSPACE, HISTORY, args.source))
        elif args.command == "archive":
            print(archive_session(WORKSPACE, HISTORY))
        else:
            reset_workspace(WORKSPACE, HISTORY, args.archive_id)
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
