"""Dispatch the existing evaluator to a ready Colab session and retrieve its artifacts."""

import json
import os
import shutil
import tarfile
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from worldsmith.benchmark.task_catalog import DEFAULT_WORKERS
from worldsmith.research_mcp.constant import WORKER_OVERRIDE
from worldsmith.research_mcp.evaluation_io import EvaluationFailure, EvaluationTimeout, task_context

from .remote_job import SESSION_BUSY, UPLOAD_FAILED, regular_tar_member
from .session import REMOTE_PROJECT, ColabError, ColabSession, session_lock
from .storage import PREFIX, R2, StorageError

# The remote supervisor enforces the evaluator deadline. Allow CLI and storage
# overhead outside that budget so diagnostics can also arrive for timed-out jobs.
TRANSFER_GRACE_SECONDS = 90


def seconds_remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise EvaluationTimeout
    return remaining


def run_colab(
    job_id: str, code: Path, output: Path, deadline: float, *,
    tasks_by_group: dict[str, list[str]], seeds: list[int], digest: str,
):
    try:
        with session_lock("evaluation"):
            session = ColabSession.active(os.environ.get("WORLDSMITH_COLAB_SESSION"))
            storage = R2()
            request = {
                "job_id": job_id,
                "controller_sha256": digest,
                "seeds": seeds,
                "groups": {
                    group: {"tasks": tasks, "workers": WORKER_OVERRIDE or DEFAULT_WORKERS[group]}
                    for group, tasks in tasks_by_group.items()
                },
            }
            with TemporaryDirectory() as temporary:
                dispatch(
                    session, storage, request, code, output, Path(temporary), deadline,
                )
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
        "output_url": storage.output_url(
            output_key, expires=ceil(seconds_remaining(deadline)) + 2 * TRANSFER_GRACE_SECONDS,
        ),
    }))
    session.command(
        "upload", str(candidate), f"/content/{job_id}.tar.gz",
        timeout=seconds_remaining(deadline),
    )
    # Upload capabilities as a file so the CLI never records them in cell history.
    session.command(
        "upload", str(config), f"/content/{job_id}.json",
        timeout=seconds_remaining(deadline),
    )
    budget = seconds_remaining(deadline)
    command = [
        f"{REMOTE_PROJECT}/.venv/bin/python", "-B", "-m", "worldsmith.colab.remote_job",
        job_id, str(budget),
    ]
    stdout = session.execute(
        "import subprocess\n"
        f"with subprocess.Popen({command!r}, cwd={REMOTE_PROJECT!r}, start_new_session=True,\n"
        "                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as job:\n"
        f"    print(f'WORLDSMITH_JOB {job_id} {{job.wait()}}', flush=True)\n",
        timeout=ceil(budget) + TRANSFER_GRACE_SECONDS,
    )
    markers = {line for line in stdout.splitlines() if line.startswith(f"WORLDSMITH_JOB {job_id} ")}
    if markers == {f"WORLDSMITH_JOB {job_id} {UPLOAD_FAILED}"}:
        raise StorageError("R2 remote artifact upload failed")
    if markers == {f"WORLDSMITH_JOB {job_id} {SESSION_BUSY}"}:
        raise ColabError("Colab still has an evaluation in progress")
    if markers != {f"WORLDSMITH_JOB {job_id} 0"}:
        if not session.is_assigned():
            raise ColabError("Colab session disappeared during evaluation")
        raise ColabError("Colab remote evaluation command did not complete")

    bundle = staging / "output.tar.gz"
    storage.download(output_key, bundle)
    artifacts = staging / "artifacts"
    try:
        with tarfile.open(bundle) as archive:
            archive.extractall(artifacts, filter=regular_tar_member)
        manifest_file = artifacts / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        outcome = manifest["status"]
        group = manifest.get("group")
        if manifest["job_id"] != job_id:
            raise ValueError("unexpected job")
        if outcome in {"controller_error", "simulation_error", "timed_out"} and group not in request["groups"]:
            raise ValueError("unexpected simulator")
        results = artifacts / "results"
        if outcome == "completed":
            for name, settings in request["groups"].items():
                for task in settings["tasks"]:
                    task_output = results / name / task
                    if not all((task_output / file).is_file() for file in ("summary.json", "episodes.jsonl")):
                        raise ValueError("missing task output")
        elif outcome == "controller_error":
            feedback = (results / group / "controller_feedback.txt").read_text().strip()
    except (KeyError, TypeError, ValueError, OSError, EOFError, tarfile.TarError):
        raise ColabError("Colab returned invalid evaluation artifacts") from None

    if results.is_dir():
        shutil.move(results, output)
    for log in artifacts.glob("*-stderr.txt"):
        shutil.move(log, output.parent / log.name)

    if outcome == "completed":
        storage.put(f"{prefix}/output/completed.json", manifest_file)
    elif outcome == "controller_error":
        raise EvaluationFailure(feedback)
    elif outcome == "simulation_error":
        raise EvaluationFailure("Simulation execution failed" + task_context(output / group))
    elif outcome == "timed_out":
        raise EvaluationTimeout(task_context(output / group))
    elif outcome == "snapshot_changed":
        raise EvaluationFailure("Controller snapshot changed during evaluation")
    else:
        raise ColabError("Colab returned an unknown evaluation status")
