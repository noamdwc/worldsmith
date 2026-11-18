"""Prepare and manage Worldsmith Colab runtimes."""

import argparse
import signal
import sys

from .environment_cache import ENVIRONMENT_PYTHONS
from .session import (
    ColabError, ColabSession, build_environments, start_session, stop_tracked_session, tracked_session,
)
from .storage import StorageError


def handle_interrupt(signum, frame):
    raise SystemExit(128 + signum)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    for action in ("status", "stop"):
        subcommands.add_parser(action)
    for action in ("start", "build-envs"):
        command = subcommands.add_parser(action)
        command.add_argument("simulator", nargs="?", default="all", choices=("all", *ENVIRONMENT_PYTHONS))
    args = parser.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_interrupt)
    try:
        if args.action == "build-envs":
            return build_environments(args.simulator)
        if args.action == "start":
            session = start_session(args.simulator)
            print(f"Inspect {session.name} with `make colab-status`.")
            print("Release the runtime with `make colab-stop`.")
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
            stop_tracked_session()
    except (ColabError, StorageError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
