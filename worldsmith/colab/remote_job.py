"""Run a single staged evaluation inside Colab and upload its private artifacts."""

import fcntl
import json
import os
import subprocess
import sys
import tarfile
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from urllib.request import Request, urlopen

from worldsmith.research_mcp.evaluation_io import group_command, run_group_process, snapshot_hash

UPLOAD_FAILED = 10
SESSION_BUSY = 11
JOB_ALREADY_STARTED = 12


def regular_tar_member(member, destination):
    if not (member.isfile() or member.isdir()):
        raise ValueError("Evaluation archives must contain only regular files and directories")
    return tarfile.data_filter(member, destination)


def evaluate(request, code: Path, artifacts: Path, deadline: float, runtime: Path):
    digest = request["controller_sha256"]
    if snapshot_hash(code) != digest:
        return {"status": "snapshot_changed"}
    for group, settings in request["groups"].items():
        environment = runtime if group == "pymunk" else runtime / "simulation_environments" / group
        output = artifacts / "results" / group
        command = group_command(
            str(environment / ".venv/bin/python"), output,
            tasks=settings["tasks"], seeds=request["seeds"], workers=settings["workers"],
        )
        with (artifacts / f"{group}-stderr.txt").open("w") as log:
            try:
                result = run_group_process(
                    command, cwd=code, env={
                        **os.environ,
                        "WORLDSMITH_LIBERO_INSTALL": str(runtime / "simulation_environments/libero"),
                        "MS_ASSET_DIR": str(runtime / "maniskill-assets"),
                    },
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(0, deadline - monotonic()),
                )
            except subprocess.TimeoutExpired:
                return {"status": "timed_out", "group": group}

        if result.returncode:
            status = "controller_error" if (output / "controller_feedback.txt").is_file() else "simulation_error"
            return {"status": status, "group": group, "exit_status": result.returncode}
        if snapshot_hash(code) != digest:
            return {"status": "snapshot_changed", "group": group}
    return {"status": "completed"}


def run_job(job_id: str, budget: float, staging: Path) -> int:
    config = staging / f"{job_id}.json"
    candidate = staging / f"{job_id}.tar.gz"
    with (staging / "worldsmith-evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return SESSION_BUSY
        try:
            try:
                (staging / f"{job_id}.started").touch(exist_ok=False)
            except FileExistsError:
                return JOB_ALREADY_STARTED
            deadline = monotonic() + budget
            dispatch = json.loads(config.read_text())
            config.unlink()
            with TemporaryDirectory(prefix=f"{job_id}-", dir=staging) as temporary:
                directory = Path(temporary)
                code = directory / "code"
                code.mkdir()
                with tarfile.open(candidate) as archive:
                    archive.extractall(code, filter=regular_tar_member)
                candidate.unlink()

                artifacts = directory / "artifacts"
                artifacts.mkdir()
                request = dispatch["request"]
                outcome = evaluate(request, code, artifacts, deadline, Path.cwd())
                (artifacts / "manifest.json").write_text(json.dumps({**request, **outcome}))
                bundle = directory / "output.tar.gz"
                with tarfile.open(bundle, "w:gz") as archive:
                    archive.add(artifacts, arcname=".")
                try:
                    with bundle.open("rb") as body, urlopen(
                        Request(
                            dispatch["output_url"], data=body, method="PUT",
                            headers={"Content-Length": str(bundle.stat().st_size), "If-None-Match": "*"},
                        ),
                        timeout=30,
                    ):
                        pass
                except (OSError, HTTPException):
                    return UPLOAD_FAILED
        finally:
            config.unlink(missing_ok=True)
            candidate.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_job(sys.argv[1], float(sys.argv[2]), Path("/content")))
