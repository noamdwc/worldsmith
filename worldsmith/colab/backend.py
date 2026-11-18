"""Dispatch the existing evaluator to a ready Colab session and retrieve its artifacts."""

import gzip
import json
import os
import shutil
import tarfile
from contextlib import contextmanager, suppress
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

from worldsmith.research_feedback import read_task_results
from worldsmith.research_mcp.constant import WORKER_OVERRIDE
from worldsmith.research_mcp.evaluation_io import EvaluationFailure, EvaluationTimeout, task_context

from .remote_job import JOB_ALREADY_STARTED, SESSION_BUSY, UPLOAD_FAILED, regular_tar_member
from .session import REMOTE_PROJECT, ColabError, ColabSession, session_lock
from .storage import PREFIX, R2, ObjectMissing, StorageError

# Four workers per simulator completed the full development suite on Colab L4
# within the existing deadline. Local evaluations use their own defaults.
COLAB_DEFAULT_WORKERS = 4

# The remote supervisor enforces the evaluator deadline. Allow CLI and storage
# overhead outside that budget so diagnostics can also arrive for timed-out jobs.
TRANSFER_GRACE_SECONDS = 90
RECOVERY_WINDOW_SECONDS = 15


def seconds_remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise EvaluationTimeout
    return remaining


@contextmanager
def run_colab(
    job_id: str, code: Path, output: Path, deadline: float, *,
    tasks_by_group: dict[str, list[str]], seeds: list[int], digest: str,
):
    try:
        with session_lock("evaluation"):
            session = ColabSession.active(os.environ.get("WORLDSMITH_COLAB_SESSION"))
            storage = R2()
            workers = WORKER_OVERRIDE if WORKER_OVERRIDE is not None else COLAB_DEFAULT_WORKERS
            request = {
                "job_id": job_id,
                "controller_sha256": digest,
                "seeds": seeds,
                "groups": {
                    group: {"tasks": tasks, "workers": workers}
                    for group, tasks in tasks_by_group.items()
                },
            }
            with TemporaryDirectory() as temporary:
                manifest = dispatch(
                    session, storage, request, code, output, Path(temporary), deadline,
                )
                try:
                    yield lambda: storage.put(f"{PREFIX}/{job_id}/output/completed.json", manifest)
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    raise ColabError("Colab returned invalid evaluation artifacts") from None
    except (ColabError, StorageError) as error:
        raise EvaluationFailure(str(error)) from None


def dispatch(session, storage, request, code, output, staging, deadline):
    job_id = request["job_id"]
    prefix = f"{PREFIX}/{job_id}"
    candidate = staging / "candidate.tar.gz"
    with tarfile.open(candidate, "w:gz") as archive:
        archive.add(code / "autoresearch", arcname="autoresearch")
    request_file = staging / "request.json"
    request_file.write_text(json.dumps(request) + "\n")
    storage.put(f"{prefix}/input/candidate.tar.gz", candidate)
    storage.put(f"{prefix}/input/request.json", request_file)

    output_key = f"{prefix}/output/artifacts.tar.gz"
    config = staging / "dispatch.json"
    config.write_text(json.dumps({
        "request": request,
        "output_url": storage.signed_url(
            output_key, upload=True,
            expires=ceil(seconds_remaining(deadline)) + 2 * TRANSFER_GRACE_SECONDS,
        ),
    }))
    session._command(
        "upload", str(candidate), f"/content/{job_id}.tar.gz",
        timeout=seconds_remaining(deadline),
    )
    # Upload capabilities as a file so the CLI never records them in cell history.
    session._command(
        "upload", str(config), f"/content/{job_id}.json",
        timeout=seconds_remaining(deadline),
    )
    budget = seconds_remaining(deadline)
    group = next(iter(request["groups"]))
    environment = REMOTE_PROJECT if group == "pymunk" else f"{REMOTE_PROJECT}/simulation_environments/{group}"
    command = [
        f"{environment}/.venv/bin/python", "-B", "-m", "worldsmith.colab.remote_job",
        job_id, str(budget),
    ]
    try:
        stdout = session.execute(
            "import subprocess\n"
            f"with subprocess.Popen({command!r}, cwd={REMOTE_PROJECT!r}, start_new_session=True,\n"
            "                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as job:\n"
            f"    print(f'WORLDSMITH_JOB {job_id} {{job.wait()}}', flush=True)\n",
            timeout=ceil(budget) + TRANSFER_GRACE_SECONDS,
        )
    except ColabError as error:
        notice = str(error)
        wait_for_artifacts = True
    else:
        marker = f"WORLDSMITH_JOB {job_id} "
        markers = {line.removeprefix(marker) for line in stdout.splitlines() if line.startswith(marker)}
        status = markers.pop() if len(markers) == 1 else None
        terminal = {
            "0": "",
            str(UPLOAD_FAILED): "R2 remote artifact upload failed",
            str(SESSION_BUSY): "Colab still has an evaluation in progress",
            str(JOB_ALREADY_STARTED): "Colab already started this evaluation",
        }
        wait_for_artifacts = status not in terminal
        notice = terminal.get(status, "Colab remote evaluation command did not complete")

    # Dispatch once. A committed R2 object is authoritative even if the CLI
    # lost its reply, and an in-flight upload can finish after a disconnect.
    bundle = staging / "output.tar.gz"
    recovery_deadline = monotonic() + (RECOVERY_WINDOW_SECONDS if wait_for_artifacts else 0)
    while True:
        try:
            storage.download(output_key, bundle)
            break
        except ObjectMissing as error:
            remaining = recovery_deadline - monotonic()
            if remaining <= 0:
                if wait_for_artifacts:
                    with suppress(ColabError):
                        if not session.is_assigned():
                            notice += ". Colab session disappeared during evaluation"
                detail = f"{notice}. " if notice else ""
                raise ColabError(detail + str(error)) from None
            sleep(min(1, remaining))

    artifacts = staging / "artifacts"
    manifest_file = artifacts / "manifest.json"
    results = artifacts / "results"
    try:
        with gzip.open(bundle, "rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                archive.extractall(artifacts, filter=regular_tar_member)
            # Read through the gzip footer before accepting the archive.
            while compressed.read(1024 * 1024):
                pass
        manifest = json.loads(manifest_file.read_text())
        validate_manifest(manifest, request, results)
        outcome = manifest["status"]
        group = manifest.get("group")
        if outcome == "controller_error":
            feedback = (results / group / "controller_feedback.txt").read_text().strip()
    except (KeyError, TypeError, ValueError, OSError, EOFError, tarfile.TarError):
        raise ColabError("Colab returned invalid evaluation artifacts") from None

    if results.is_dir():
        shutil.move(results, output)
    for log in artifacts.glob("*-stderr.txt"):
        shutil.move(log, output.parent / log.name)

    if outcome == "completed":
        return manifest_file
    if outcome == "controller_error":
        raise EvaluationFailure(feedback)
    elif outcome == "simulation_error":
        raise EvaluationFailure("Simulation execution failed" + task_context(output / group))
    elif outcome == "timed_out":
        raise EvaluationTimeout(task_context(output / group))
    elif outcome == "snapshot_changed":
        raise EvaluationFailure("Controller snapshot changed during evaluation")


def validate_manifest(manifest, request, results: Path):
    if any(manifest[key] != value for key, value in request.items()):
        raise ValueError("unexpected evaluation request")
    outcome = manifest["status"]
    group = manifest.get("group")
    groups = request["groups"]
    if outcome == "completed" or (outcome == "snapshot_changed" and group is None):
        if group is not None:
            raise ValueError("unexpected simulator")
    elif outcome in {"controller_error", "simulation_error", "timed_out", "snapshot_changed"}:
        if group not in groups or not (results.parent / f"{group}-stderr.txt").is_file():
            raise ValueError("missing simulator diagnostics")
        if outcome in {"controller_error", "simulation_error"}:
            code = manifest["exit_status"]
            if type(code) is not int or not code:
                raise ValueError("invalid evaluator exit status")
    else:
        raise ValueError("unexpected evaluation outcome")

    actual_groups = {directory.name for directory in results.iterdir()} if results.is_dir() else set()
    if outcome == "snapshot_changed" and group is None:
        if actual_groups:
            raise ValueError("unexpected simulator outputs")
        return
    for name, settings in groups.items():
        actual_groups.discard(name)
        directory = results / name
        expected_tasks = set(settings["tasks"])
        present_tasks = {task.name for task in directory.iterdir() if task.is_dir()} if directory.is_dir() else set()
        if not present_tasks <= expected_tasks:
            raise ValueError("unexpected task outputs")
        if name != group and present_tasks != expected_tasks:
            raise ValueError("missing task outputs")
        current_task = None
        if name == group and (directory / "task.txt").is_file():
            current_task = (directory / "task.txt").read_text()
            if current_task not in expected_tasks:
                raise ValueError("unexpected simulator diagnostic")
        for task in present_tasks - {current_task}:
            task_output = directory / task
            if outcome == "completed":
                if not all((task_output / file).is_file() for file in ("summary.json", "episodes.jsonl")):
                    raise ValueError("missing task output")
            else:
                read_task_results(task_output, request["seeds"])
        if name == group:
            break
    if actual_groups:
        raise ValueError("unexpected simulator outputs")
