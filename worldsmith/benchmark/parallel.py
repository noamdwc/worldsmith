"""Run a task in spawned processes while streaming results in seed order."""

import pickle
import signal
from itertools import cycle
from multiprocessing import get_context
from multiprocessing.connection import wait
from threading import current_thread, main_thread
from time import monotonic
from traceback import format_exc

from worldsmith.benchmark.results import TaskEvaluation, record_episodes
from worldsmith.benchmark.runner import ControllerError, run_scored_episode

# Bound stalled owner CLI runs. AutoResearch keeps its shorter suite deadline.
WORKER_RESPONSE_TIMEOUT_SECONDS = 600


class WorkerFailure(RuntimeError):
    """The evaluation infrastructure failed before it could produce a result."""


def _exit_on_sigterm(signum, frame):
    raise SystemExit(128 + signum)


def _run_worker(connection, open_adapter, factory, task):
    # The parent owns interruption and shuts down all workers together.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with connection, open_adapter() as adapter:
        while True:
            try:
                command, value = connection.recv()
            except EOFError:
                return
            if command == "summarize":
                result = adapter.summarize(value), adapter.metadata()
            else:
                try:
                    result = run_scored_episode(adapter, factory, value, task)
                except ControllerError as exc:
                    # Preserve the private traceback as well as the public message.
                    exc.add_note(format_exc())
                    result = exc
            try:
                connection.send(result)
            except BrokenPipeError:
                return


def _receive(connection, sentinels, task):
    ready = wait([connection, *sentinels], timeout=WORKER_RESPONSE_TIMEOUT_SECONDS)
    if not ready:
        raise WorkerFailure(
            f"{task}: evaluation worker did not respond within {WORKER_RESPONSE_TIMEOUT_SECONDS} seconds"
        )
    if connection in ready:
        try:
            result = connection.recv()
        except (EOFError, OSError) as exc:
            raise WorkerFailure(f"{task}: evaluation worker disconnected") from exc
        if isinstance(result, ControllerError):
            raise result
        return result
    process = sentinels[ready[0]]
    raise WorkerFailure(f"{task}: {process.name} exited unexpectedly (status {process.exitcode})")


def _stop_workers(processes, connections, *, grace_seconds):
    for connection in connections:
        connection.close()
    # Idle workers close their adapters on EOF. A stalled native call must not
    # hold up failure or interrupt handling, and the grace period is shared.
    deadline = monotonic() + grace_seconds
    for process in processes:
        process.join(max(0, deadline - monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(f"{process.name}, exit status {process.exitcode}")
        process.close()
    return failed


def evaluate_parallel(open_adapter, factory, seeds, record, *, task, workers):
    context = get_context("spawn")
    count = min(workers, len(seeds))
    connections = []
    processes = []
    cleanup_grace_seconds = 1
    # Owner CLIs can be stopped with SIGTERM as well as Ctrl-C. Respect an
    # embedding application's signal handler if it has installed one.
    handle_sigterm = (
        current_thread() is main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    )
    if handle_sigterm:
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        for index in range(count):
            parent, child = context.Pipe()
            connections.append(parent)
            with child:
                process = context.Process(
                    target=_run_worker, args=(child, open_adapter, factory, task),
                    name=f"evaluation worker {index + 1}",
                )
                try:
                    process.start()
                except (AttributeError, TypeError, pickle.PicklingError) as exc:
                    raise ValueError(
                        f"{task}: parallel evaluation requires an importable, picklable "
                        "adapter opener and controller factory"
                    ) from exc
                processes.append(process)
        sentinels = {process.sentinel: process for process in processes}

        # Keep at most one outstanding episode per worker. Consume in the
        # original order even when a later seed finishes first.
        for connection, seed in zip(connections, seeds):
            connection.send(("episode", seed))

        def episodes():
            for index, connection in zip(range(len(seeds)), cycle(connections)):
                result = _receive(connection, sentinels, task)
                if index + count < len(seeds):
                    connection.send(("episode", seeds[index + count]))
                yield result

        rows, scores = record_episodes(episodes(), record)
        # Some adapters expose metadata only after reset. Reuse the worker
        # that ran the final seed instead of opening another simulator.
        last = connections[(len(seeds) - 1) % count]
        last.send(("summarize", scores))
        metrics, metadata = _receive(last, sentinels, task)
        # Native libraries can take longer to exit after a completed run.
        cleanup_grace_seconds = 5
    finally:
        try:
            failed = _stop_workers(processes, connections, grace_seconds=cleanup_grace_seconds)
        finally:
            if handle_sigterm:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if failed:
        raise WorkerFailure(f"{task}: evaluation worker cleanup failed ({', '.join(failed)})")
    return TaskEvaluation(rows, metrics), metadata
