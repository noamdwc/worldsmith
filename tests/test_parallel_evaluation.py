import json
import multiprocessing
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from time import monotonic, sleep

import pytest

from worldsmith.benchmark import parallel
from worldsmith.benchmark.evaluator import evaluate_task, evaluate_to_directory
from worldsmith.benchmark.parallel import WorkerFailure
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import TASK_GROUPS


class LoggedAdapter:
    def __init__(self, directory, mode):
        self.directory = directory
        self.mode = mode

    def reset(self, seed):
        self.seed = seed
        (self.directory / f"{seed}.pid").write_text(str(os.getpid()))
        if self.mode == "stall":
            sleep(30)
        if seed == 7 and self.mode in {"out-of-order", "crash"}:
            deadline = monotonic() + 10
            while not (self.directory / "8.done").exists():
                if monotonic() >= deadline:
                    raise RuntimeError("Second worker did not complete")
                sleep(0.01)
            if self.mode == "crash":
                sleep(30)
        if seed == 8 and self.mode == "crash":
            deadline = monotonic() + 10
            while not (self.directory / "7.pid").exists() and monotonic() < deadline:
                sleep(0.01)
            (self.directory / "8.done").touch()
            os._exit(17)
        return seed, {"seed": seed}

    def encode_action(self, action):
        return action

    def step(self, action):
        return self.seed + action, 1, self.seed % 2 == 1, self.seed % 2 == 0, {}

    def score(self, episode):
        (self.directory / f"{self.seed}.done").touch()
        return {"points": episode["final_observation"]}

    def summarize(self, episodes):
        return {"points_in_order": [row["points"] for row in episodes]}

    def metadata(self):
        return {"last_seed": self.seed}


@contextmanager
def open_logged_adapter(directory, mode="normal"):
    directory.mkdir(exist_ok=True)
    pid = os.getpid()
    (directory / f"open-{pid}").touch()
    try:
        yield LoggedAdapter(directory, mode)
    finally:
        (directory / f"closed-{pid}").touch()


def make_controller():
    return lambda observation: observation * 2


def failing_factory():
    raise RuntimeError("private exception detail")


def test_parallel_seed_order_scores_and_worker_reuse(tmp_path):
    sequential, parallel = tmp_path / "sequential", tmp_path / "parallel"
    log = tmp_path / "workers"
    expected = evaluate_to_directory(
        partial(open_logged_adapter, tmp_path / "serial"), make_controller, 6, 7, sequential,
    )
    actual = evaluate_to_directory(
        partial(open_logged_adapter, log, "out-of-order"), make_controller, 6, 7, parallel, workers=2,
    )
    assert actual == expected
    assert actual["points_in_order"] == [21, 24, 27, 30, 33, 36]
    assert actual["last_seed"] == 12
    assert (parallel / "episodes.jsonl").read_bytes() == (sequential / "episodes.jsonl").read_bytes()
    pids = [(log / f"{seed}.pid").read_text() for seed in range(7, 13)]
    assert pids[0] != pids[1] and set(pids) == {pids[0], pids[1]}
    assert len(list(log.glob("open-*"))) == 2
    assert {file.name for file in log.glob("closed-*")} == {f"closed-{pid}" for pid in pids}


def test_worker_crash_aborts_wait_and_reaps_other_workers(tmp_path):
    log = tmp_path / "workers"
    previous = multiprocessing.active_children()
    with pytest.raises(WorkerFailure, match="evaluation worker"):
        evaluate_to_directory(
            partial(open_logged_adapter, log, "crash"), make_controller, 2, 7,
            tmp_path / "evaluation", workers=2,
        )
    assert multiprocessing.active_children() == previous
    for marker in log.glob("*.pid"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    assert not (tmp_path / "evaluation/summary.json").exists()


def test_stalled_worker_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "WORKER_RESPONSE_TIMEOUT_SECONDS", 0.1)
    previous = multiprocessing.active_children()
    with pytest.raises(WorkerFailure, match="did not respond"):
        evaluate_to_directory(
            partial(open_logged_adapter, tmp_path / "workers", "stall"), make_controller,
            2, 7, tmp_path / "evaluation", workers=2,
        )
    assert multiprocessing.active_children() == previous


def test_interrupt_closes_and_reaps_parallel_adapters(tmp_path):
    def interrupt(episode, metrics):
        raise KeyboardInterrupt

    previous = multiprocessing.active_children()
    with pytest.raises(KeyboardInterrupt):
        evaluate_task(
            partial(open_logged_adapter, tmp_path, "out-of-order"), make_controller,
            [7, 8], interrupt, task="counter", workers=2,
        )
    assert multiprocessing.active_children() == previous
    assert len(list(tmp_path.glob("closed-*"))) == 2


def test_sigterm_reaps_stalled_workers(tmp_path):
    script = """
import sys
from functools import partial
from pathlib import Path
from test_parallel_evaluation import evaluate_to_directory, make_controller, open_logged_adapter

directory = Path(sys.argv[1])
evaluate_to_directory(
    partial(open_logged_adapter, directory, "stall"), make_controller, 2, 7,
    directory / "evaluation", workers=2,
)
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)}
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)], env=env, start_new_session=True,
    ) as process:
        try:
            markers = [tmp_path / f"{seed}.pid" for seed in (7, 8)]
            deadline = monotonic() + 10
            while not all(marker.is_file() and marker.stat().st_size for marker in markers):
                assert process.poll() is None and monotonic() < deadline
                sleep(0.01)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=5) == 128 + signal.SIGTERM
            for marker in markers:
                with pytest.raises(ProcessLookupError):
                    os.kill(int(marker.read_text()), 0)
        finally:
            # Prevent a leak even if the evaluator regresses.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_controller_error_keeps_diagnostic_and_public_message(tmp_path):
    with pytest.raises(ControllerError, match="private exception detail") as error:
        evaluate_to_directory(
            partial(open_logged_adapter, tmp_path / "workers"), failing_factory, 2, 7,
            tmp_path / "evaluation", task="counter", workers=4,
        )
    assert error.value.public_message == "counter, seed 7: make_controller raised an exception"
    assert "failing_factory" in error.value.__notes__[0]


def test_parallel_rejects_unpicklable_factory(tmp_path):
    with pytest.raises(ValueError, match="importable, picklable"):
        evaluate_to_directory(
            partial(open_logged_adapter, tmp_path / "workers"), lambda: None, 1, 7,
            tmp_path / "evaluation", workers=2,
        )


@pytest.mark.parametrize("cli", [
    ["worldsmith.benchmark.evaluator"],
    ["worldsmith", "evaluate"],
    ["worldsmith.research_evaluation", "pymunk", "unused"],
])
def test_cli_rejects_zero_workers(cli):
    extra = ["--task", "interception", "--output", "unused"] if "pymunk" not in cli else []
    result = subprocess.run(
        [sys.executable, "-m", *cli, *extra, "--workers", "0"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "worker count must be a positive integer" in result.stderr


def test_legacy_cli_parallel_preserves_heldout_split_and_trace(tmp_path):
    for workers in (1, 2):
        subprocess.run(
            [sys.executable, "-m", "worldsmith", "evaluate", "--task", "all", "--split", "heldout",
             "--workers", str(workers), "--output", str(tmp_path / str(workers))],
            check=True, stdout=subprocess.DEVNULL, timeout=30,
        )
    for name in ("summary.json", "episodes.jsonl"):
        assert (tmp_path / "1" / name).read_bytes() == (tmp_path / "2" / name).read_bytes()
    summary = json.loads((tmp_path / "2/summary.json").read_text())
    assert summary["seeds"] == list(range(1000, 1008))


def test_research_cli_parallel_uses_owner_seed_file(tmp_path):
    starts = {name: 1000 + index * 8 for index, name in enumerate(TASK_GROUPS["pymunk"])}
    # The original seed-file format covers both groups in the legacy suite.
    starts.update({name: 2000 + index * 8 for index, name in enumerate(TASK_GROUPS["maniskill"])})
    seed_file = tmp_path / "heldout.json"
    seed_file.write_text(json.dumps(starts))
    output = tmp_path / "owner"
    subprocess.run(
        [sys.executable, "-B", "-m", "worldsmith.research_evaluation", "pymunk", str(output),
         "--heldout-seed-file", str(seed_file), "--workers", "2"],
        check=True, timeout=30,
    )
    for name in TASK_GROUPS["pymunk"]:
        summary = json.loads((output / name / "summary.json").read_text())
        assert summary["seeds"] == list(range(starts[name], starts[name] + 8))
