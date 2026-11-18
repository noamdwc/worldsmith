"""Owner commands for the AutoResearch Colab runtime."""

import argparse
import os
import signal
import subprocess
import sys
from contextlib import suppress

from worldsmith.research_lifecycle import ROOT

from .environment_cache import ENVIRONMENT_PYTHONS
from .session import (
    ColabError, ColabSession, build_environments, session_lock, start_session, stop_after_run, tracked_session,
)
from .storage import StorageError


def handle_interrupt(signum, frame):
    raise SystemExit(128 + signum)


def run_research(suite: str) -> int:
    env = dict(os.environ)
    backend = env.setdefault("WORLDSMITH_EVALUATION_BACKEND", "colab")
    if backend not in {"local", "colab"}:
        raise ColabError("WORLDSMITH_EVALUATION_BACKEND must be local or colab")
    if backend == "local":
        return subprocess.call([str(ROOT / "scripts/run-autoresearch.sh"), suite], env=env)

    session = start_session(suite)
    status = 1
    try:
        env["WORLDSMITH_COLAB_SESSION"] = session.name
        with subprocess.Popen(
            [str(ROOT / "scripts/run-autoresearch.sh"), suite], env=env, start_new_session=True,
        ) as researcher:
            try:
                status = researcher.wait()
            except SystemExit as interrupt:
                signum = interrupt.code - 128
                with suppress(ProcessLookupError):
                    os.killpg(researcher.pid, signum)
                try:
                    researcher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(researcher.pid, signal.SIGKILL)
                raise
    finally:
        try:
            with session_lock("lifecycle"):
                state = tracked_session()
                if state is not None and state["name"] == session.name:
                    if not stop_after_run(session) and status == 0:
                        status = 1
        except (ColabError, OSError) as error:
            print(f"Could not clean up {session.name}. {error}. Check make colab-status.", file=sys.stderr)
            status = status or 1
    return status if status >= 0 else 128 - status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    for action in ("status", "stop"):
        subcommands.add_parser(action)
    for action in ("start", "build-envs"):
        command = subcommands.add_parser(action)
        command.add_argument("simulator", nargs="?", default="all", choices=("all", *ENVIRONMENT_PYTHONS))
    run = subcommands.add_parser("run")
    run.add_argument("suite", nargs="?", default="all", choices=("all", "maniskill", "libero", "robosuite"))
    args = parser.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_interrupt)
    try:
        if args.action == "run":
            return run_research(args.suite)
        if args.action == "build-envs":
            return build_environments(args.simulator)
        if args.action == "start":
            start_session(args.simulator)
        elif args.action == "status":
            state = tracked_session()
            if state is None:
                print("No Worldsmith Colab session is tracked")
            else:
                session = ColabSession(state["name"])
                availability = "assigned" if session.is_assigned() else "unavailable"
                print(f"Colab session {session.name} is {availability}")
                if not state["ready"]:
                    print("Worldsmith initialization is incomplete")
        elif args.action == "stop":
            with session_lock("lifecycle"):
                state = tracked_session()
                if state is not None:
                    ColabSession(state["name"]).stop()
    except (ColabError, StorageError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
