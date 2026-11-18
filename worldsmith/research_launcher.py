"""Launch AutoResearch with local or Colab evaluation."""

import argparse
import os
import signal
import subprocess
import sys
from contextlib import suppress

from worldsmith import research_lifecycle as workspace
from worldsmith.colab import session as colab

SUITE_TASKS = {
    "all": "the full development suite",
    "maniskill": "tabletop manipulation and quadruped locomotion",
    "libero": "the LIBERO-PRO development suite",
    "robosuite": "the door opening, nut assembly, and hammer handover tasks",
}
WORKSPACE_ERRORS = (OSError, ValueError, RuntimeError, subprocess.CalledProcessError)


def handle_interrupt(signum, frame):
    raise SystemExit(128 + signum)


def run_codex(suite: str, env: dict[str, str]) -> int:
    prompt = f"Run the three-iteration AutoResearch session in RESEARCH.md on {SUITE_TASKS[suite]} and summarize your findings."
    command = ["codex", "exec", "-c", 'model_reasoning_effort="low"', "--sandbox", "workspace-write", prompt]
    try:
        researcher = subprocess.Popen(command, cwd=workspace.WORKSPACE, env=env, start_new_session=True)
    except OSError as error:
        print(f"Could not start Codex. {error}", file=sys.stderr)
        return 127 if isinstance(error, FileNotFoundError) else 126

    try:
        status = researcher.wait()
    except SystemExit as interrupt:
        with suppress(ProcessLookupError):
            os.killpg(researcher.pid, interrupt.code - 128)
        try:
            researcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(researcher.pid, signal.SIGKILL)
            researcher.wait()
        return interrupt.code
    return status if status >= 0 else 128 - status


def run_workspace(suite: str, env: dict[str, str]) -> int:
    workspace.start_session(workspace.WORKSPACE)
    status = run_codex(suite, env)
    try:
        archive_id = workspace.archive_session(workspace.WORKSPACE, workspace.HISTORY)
    except WORKSPACE_ERRORS as error:
        print(error, file=sys.stderr)
        print("Could not archive the research session. Workspace kept for inspection.", file=sys.stderr)
        return status or 1
    print(f"Archived session in research-history/{archive_id}")
    if status:
        print(f"Codex exited with status {status}. Workspace kept for inspection.", file=sys.stderr)
        return status

    workspace.reset_workspace(workspace.WORKSPACE, workspace.HISTORY, archive_id)
    return 0


def run_research(suite: str) -> int:
    env = dict(os.environ)
    env["WORLDSMITH_RESEARCH_SUITE"] = suite
    backend = env.setdefault("WORLDSMITH_EVALUATION_BACKEND", "colab")
    if backend == "local":
        return run_workspace(suite, env)
    if backend != "colab":
        raise ValueError("WORLDSMITH_EVALUATION_BACKEND must be local or colab")

    session = colab.start_session(suite)
    status = 1
    try:
        env["WORLDSMITH_COLAB_SESSION"] = session.name
        status = run_workspace(suite, env)
    finally:
        try:
            colab.stop_tracked_session(expected_name=session.name)
        except (colab.ColabError, OSError) as error:
            print(f"Could not clean up {session.name}. {error}. Retry cleanup with make colab-stop.", file=sys.stderr)
            status = status or 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", default="all", choices=SUITE_TASKS)
    args = parser.parse_args(argv)
    previous = {sig: signal.signal(sig, handle_interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return run_research(args.suite)
    except WORKSPACE_ERRORS as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
