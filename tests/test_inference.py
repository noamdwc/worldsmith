"""Inference contracts without a GPU, model download, or Colab runtime."""

import json
import signal
import subprocess
from types import SimpleNamespace
from urllib.error import URLError

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from worldsmith.inference import server
from worldsmith.inference.client import create_client
from worldsmith.inference.config import API_KEY, BASE_URL, MODEL, MODEL_REVISION, server_command
from worldsmith.inference.smoke import run_smoke


def completion(*, content=None, calls=None, finish="stop"):
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": content, "tool_calls": calls,
            }}]}


def tool_call(arguments=None):
    return {"id": "call-distance", "type": "function", "function": {
        "name": "calculate_distance",
        "arguments": json.dumps(arguments if arguments is not None else {"x1": 0, "y1": 0, "x2": 3, "y2": 4}),
    }}


def test_smoke_uses_standard_sdk_and_preserves_tool_conversation():
    requests = []
    replies = [completion(content="Hello!"), completion(calls=[tool_call()], finish="tool_calls"),
               completion(content="The distance is 5.")]

    def respond(request):
        requests.append(json.loads(request.content))
        assert str(request.url) == BASE_URL + "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        return httpx.Response(200, json=replies.pop(0))

    with create_client() as client:
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        result = run_smoke(client)
    assert result["distance"] == 5
    assert all(request["model"] == MODEL and request["temperature"] == 0 for request in requests)
    assert requests[1]["tool_choice"] == "auto"
    assert requests[1]["tools"][0]["function"]["parameters"]["required"] == ["x1", "y1", "x2", "y2"]
    assert requests[2]["messages"][1]["tool_calls"][0]["id"] == "call-distance"
    assert requests[2]["messages"][2] == {"role": "tool", "tool_call_id": "call-distance", "content": "5.0"}


@pytest.mark.parametrize("bad_call", [
    tool_call({"x1": 0, "y1": 0, "x2": 3}),
    tool_call({"x1": False, "y1": 0, "x2": 3, "y2": 4}),
    tool_call({"x1": "0", "y1": 0, "x2": 3, "y2": 4}),
    tool_call({"x1": 0, "y1": 0, "x2": 6, "y2": 8}),
    tool_call({"x1": float("nan"), "y1": 0, "x2": 3, "y2": 4}),
    {**tool_call(), "function": {"name": "wrong", "arguments": "{}"}},
    {**tool_call(), "function": {"name": "calculate_distance", "arguments": "not json"}},
])
def test_smoke_rejects_invalid_model_arguments(bad_call):
    replies = [completion(content="Hello"), completion(calls=[bad_call], finish="tool_calls")]
    with create_client() as client:
        client._client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=replies.pop(0)),
        ))
        with pytest.raises(ValueError):
            run_smoke(client)
    assert not replies


@pytest.mark.parametrize("failure, expected", [
    (httpx.ConnectError("unavailable"), APIConnectionError),
    (httpx.ReadTimeout("late"), APITimeoutError),
    (503, APIStatusError),
])
def test_sdk_errors_propagate_without_retries(failure, expected):
    requests = []

    def respond(request):
        requests.append(request)
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "model unavailable"}})
        raise failure

    with create_client(timeout=3) as client:
        assert client.timeout == 3 and client.max_retries == 0
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(expected):
            run_smoke(client)
    assert len(requests) == 1


@pytest.fixture
def server_process(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=12345, returncode=None, poll=lambda: None, wait=lambda **kw: None)
    starts, signals = [], []

    def popen(command, **kwargs):
        starts.append((command, kwargs))
        return process

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server.os, "killpg", lambda *args: signals.append(args))
    # Do not bind a real port in CPU-only unit tests.
    from contextlib import nullcontext
    monkeypatch.setattr(server.socket, "socket", lambda: nullcontext(SimpleNamespace(bind=lambda addr: None)))
    return server.ModelServer(log=tmp_path / "vllm.log"), process, starts, signals


def test_server_pins_model_and_stops_workers_on_context_failure(server_process, monkeypatch):
    from contextlib import nullcontext
    model_server, process, starts, signals = server_process
    monkeypatch.setattr(server, "urlopen", lambda *a, **kw: nullcontext())
    with pytest.raises(ValueError, match="caller failed"):
        with model_server:
            with pytest.raises(RuntimeError, match="already been started"):
                model_server.start()
            raise ValueError("caller failed")
    command, kwargs = starts[0]
    assert command == server_command(model_server.python)
    assert command[command.index("--revision") + 1] == MODEL_REVISION
    assert command[command.index("--tokenizer-revision") + 1] == MODEL_REVISION
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert kwargs["start_new_session"]
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    model_server.stop()
    assert len(signals) == 2


@pytest.mark.parametrize("failure", ["exit", "timeout", "interrupt"])
def test_failed_start_cleans_up_and_reports_log(server_process, monkeypatch, failure):
    model_server, process, _, signals = server_process
    if failure == "exit":
        process.returncode = 1
        process.poll = lambda: 1
        expected = RuntimeError
    elif failure == "interrupt":
        process.poll = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        expected = KeyboardInterrupt
    else:
        ticks = iter([0, 0, 1000])
        monkeypatch.setattr(server, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(server, "sleep", lambda seconds: None)
        expected = TimeoutError
    monkeypatch.setattr(server, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(URLError("not ready")))
    with pytest.raises(expected) as error:
        model_server.start()
    if failure != "interrupt":
        assert str(model_server.log) in str(error.value)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert model_server.process is None


def test_stop_kills_unresponsive_process_group(server_process):
    model_server, process, _, signals = server_process
    waits = []

    def wait(*, timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("vllm", timeout)

    process.wait = wait
    model_server.process = process
    model_server.stop()
    assert waits == [30, 10]
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


@pytest.mark.parametrize("failure", ["signal", "wait"])
def test_graceful_shutdown_errors_still_force_cleanup(server_process, monkeypatch, failure):
    model_server, process, _, signals = server_process
    model_server.process = process
    waits = []

    def killpg(pid, sig):
        signals.append((pid, sig))
        if failure == "signal" and sig == signal.SIGTERM:
            raise PermissionError("graceful signal failed")

    def wait(*, timeout):
        waits.append(timeout)
        if failure == "wait" and timeout == 30:
            raise OSError("graceful wait failed")

    monkeypatch.setattr(server.os, "killpg", killpg)
    process.wait = wait
    with pytest.raises(OSError, match="graceful"):
        model_server.stop()
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert waits[-1] == 10
    assert model_server.process is None
    model_server.stop()
    assert len(signals) == 2


@pytest.mark.parametrize("failure", ["signal", "wait"])
def test_failed_forced_cleanup_retains_process_for_retry(server_process, monkeypatch, failure):
    model_server, process, _, signals = server_process
    model_server.process = process

    def killpg(pid, sig):
        signals.append((pid, sig))
        if failure == "signal" and sig == signal.SIGKILL:
            raise PermissionError("forced signal failed")

    def wait(*, timeout):
        if failure == "wait" and timeout == 10:
            raise subprocess.TimeoutExpired("vllm", timeout)

    monkeypatch.setattr(server.os, "killpg", killpg)
    process.wait = wait
    with pytest.raises((PermissionError, subprocess.TimeoutExpired)):
        model_server.stop()
    assert model_server.process is process
    with pytest.raises(RuntimeError, match="already been started"):
        model_server.start()

    monkeypatch.setattr(server.os, "killpg", lambda *args: signals.append(args))
    process.wait = lambda **kwargs: None
    model_server.stop()
    assert model_server.process is None
    model_server.stop()
    assert len(signals) == 4


def test_stop_handles_already_gone_process_group(server_process, monkeypatch):
    model_server, process, _, signals = server_process
    model_server.process = process

    def killpg(pid, sig):
        signals.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(server.os, "killpg", killpg)
    model_server.stop()
    model_server.stop()
    assert model_server.process is None
    assert len(signals) == 2


def test_startup_error_is_preserved_when_cleanup_fails(server_process, monkeypatch):
    model_server, process, _, _ = server_process
    process.returncode = 1
    process.poll = lambda: 1

    def killpg(pid, sig):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(server.os, "killpg", killpg)
    with pytest.raises(RuntimeError, match="vLLM exited with status 1") as error:
        model_server.start()
    assert error.value.__notes__ == ["vLLM cleanup also failed: cleanup denied"]
    assert model_server.process is process


@pytest.mark.parametrize("caller_error", [None, ValueError("caller failed")])
def test_context_cleanup_failure_reports_error_without_masking_caller(server_process, monkeypatch, caller_error):
    from contextlib import nullcontext
    model_server, process, _, _ = server_process
    monkeypatch.setattr(server, "urlopen", lambda *args, **kwargs: nullcontext())

    def killpg(pid, sig):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(server.os, "killpg", killpg)
    expected = PermissionError if caller_error is None else ValueError
    with pytest.raises(expected) as error:
        with model_server:
            if caller_error is not None:
                raise caller_error
    if caller_error is not None:
        assert error.value is caller_error
        assert error.value.__notes__ == ["vLLM cleanup also failed: cleanup denied"]
    assert model_server.process is process


@pytest.mark.parametrize("timeout", [0, -1])
def test_server_rejects_invalid_startup_timeout(timeout):
    with pytest.raises(ValueError, match="positive"):
        server.ModelServer(startup_timeout=timeout)
