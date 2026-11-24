"""Observable SLM conversation, action, and shared-runner behavior."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from worldsmith.api import Action
from worldsmith.benchmark.adapters.pymunk_adapter import PymunkAdapter
from worldsmith.benchmark.runner import ControllerError, run_episode
from worldsmith.benchmark.tasks.navigation import Navigation
from worldsmith.inference.controller import ACT_TOOL, SLMController, run_slm_episode


class Tokenizer:
    """Variable token cost, including the tool schema and chat formatting."""

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        assert tools == [ACT_TOOL] and tokenize and add_generation_prompt
        return list(range(len(json.dumps(messages)) + len(json.dumps(tools)) + 10))


class Inference:
    def __init__(self):
        self.requests = []
        self.message = None
        self.finish_reason = "tool_calls"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(deepcopy(request))
        assert request["tools"] == [ACT_TOOL]
        assert request["tool_choice"]["function"]["name"] == "act"
        assert request["parallel_tool_calls"] is False
        message = self.message or ChatCompletionMessage(
            role="assistant", tool_calls=[{
                "id": f"call-{len(self.requests)}", "type": "function",
                "function": {"name": "act", "arguments": '{"heading": 0, "throttle": 0.5}'},
            }],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)])


class ShortNavigation(Navigation):
    step_limit = 12


@pytest.mark.parametrize("limit,success,expected", [(12, False, [5, 5, 2]), (3, True, [3])])
def test_repeat_horizon_and_immediate_termination(limit, success, expected):
    class Task(ShortNavigation):
        step_limit = limit

        def succeeded(self, measurements):
            return success and len(measurements) == 4

    client = Inference()
    adapter = PymunkAdapter(Task())
    episode, decisions = run_slm_episode(adapter, client, Tokenizer(), 0, task="navigation")
    assert episode["length"] == limit
    assert decisions == len(expected)
    assert episode["steps"][-1]["terminated"] is success
    assert episode["steps"][-1]["truncated"] is not success
    assert all(step["action"] == {"heading": 0, "throttle": 0.5} for step in episode["steps"])
    assert episode["final_observation"]["time"] == pytest.approx(limit * 0.1)
    first = client.requests[0]["messages"]
    assert Task.objective in first[0]["content"]
    assert "5 environment steps" in first[0]["content"]
    assert f"{limit} environment steps" in first[0]["content"]
    assert f"Current step: 0/{limit}" in first[1]["content"]
    assert "covariance" in first[1]["content"]
    if decisions > 1:
        messages = client.requests[1]["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
        assert messages[2]["tool_calls"][0]["id"] == messages[3]["tool_call_id"]
        assert json.loads(messages[3]["content"])["executed_steps"] == 5
        assert "Current step: 5/12" in messages[-1]["content"]
        assert "Remaining steps: 7" in messages[-1]["content"]
        assert "Elapsed simulation time: 0.5" in messages[-1]["content"]
    run_slm_episode(adapter, client, Tokenizer(), 1, task="navigation")
    assert len(client.requests[decisions]["messages"]) == 2
    assert "Current step: 0/" in client.requests[decisions]["messages"][-1]["content"]


def controller(client, **kwargs):
    return SLMController(client, Tokenizer(), task_description="A task", max_steps=80, **kwargs)


def observation():
    return PymunkAdapter(Navigation()).reset(0)[0]


def complete_decision(control, obs, step):
    assert control(obs) == Action(0, 0.5)
    control.record_result({"current_step": step, "executed_steps": 5, "terminated": False, "truncated": False})


def test_dynamic_trimming_preserves_complete_recent_interactions():
    client = Inference()
    control = controller(client)
    obs = observation()
    complete_decision(control, obs, 5)
    complete_decision(control, obs, 5)
    control(obs)
    three = deepcopy(client.requests[-1]["messages"])
    # Budget exactly enough for the newest completed interaction and current query.
    recent = [three[0], *three[4:]]
    control.context_window = len(Tokenizer().apply_chat_template(
        recent, tools=[ACT_TOOL], tokenize=True, add_generation_prompt=True,
    )) + control.response_tokens
    control.record_result({"current_step": 5, "executed_steps": 5, "terminated": False, "truncated": False})
    control(obs)
    messages = client.requests[-1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2]["tool_calls"][0]["id"] == "call-3"
    assert messages[3]["tool_call_id"] == "call-3"
    assert messages[0] == three[0]
    assert messages[-1] == three[-1]
    assert len(Tokenizer().apply_chat_template(
        messages, tools=[ACT_TOOL], tokenize=True, add_generation_prompt=True,
    )) + control.response_tokens <= control.context_window


def test_larger_current_observation_evicts_all_history():
    client = Inference()
    control = controller(client)
    obs = observation()
    complete_decision(control, obs, 5)
    control(obs)
    initial = client.requests[-1]["messages"]
    # Leave room for mandatory context only; a larger new observation forces eviction.
    mandatory = [initial[0], initial[-1]]
    control.context_window = len(Tokenizer().apply_chat_template(
        mandatory, tools=[ACT_TOOL], tokenize=True, add_generation_prompt=True,
    )) + control.response_tokens + 100
    control.record_result({"current_step": 5, "executed_steps": 5, "terminated": False, "truncated": False})
    control(replace(obs, robot_id="r" * 50))
    assert [m["role"] for m in client.requests[-1]["messages"]] == ["system", "user"]
    assert "r" * 50 in client.requests[-1]["messages"][-1]["content"]


def test_mandatory_context_exhaustion_never_calls_model():
    client = Inference()
    control = controller(client, context_window=300, response_tokens=100)
    with pytest.raises(ValueError, match="Mandatory controller context"):
        control(observation())
    assert client.requests == []


@pytest.mark.parametrize("mutation", ["missing", "multiple", "name", "id", "json", "extra", "bool", "nan", "bounds", "length"])
def test_malformed_calls_fail_before_simulator_action(mutation):
    client = Inference()
    message = ChatCompletionMessage(role="assistant", tool_calls=[{
        "id": "call-1", "type": "function",
        "function": {"name": "act", "arguments": '{"heading": 0, "throttle": 1}'},
    }])
    if mutation == "missing":
        message.tool_calls = None
    elif mutation == "multiple":
        message.tool_calls *= 2
    elif mutation == "name":
        message.tool_calls[0].function.name = "other"
    elif mutation == "id":
        message.tool_calls[0].id = ""
    elif mutation == "length":
        client.finish_reason = "length"
    else:
        message.tool_calls[0].function.arguments = {
            "json": "{", "extra": '{"heading": 0, "throttle": 1, "x": 1}',
            "bool": '{"heading": true, "throttle": 1}',
            "nan": '{"heading": NaN, "throttle": 1}',
            "bounds": '{"heading": 0, "throttle": 2}',
        }[mutation]
    client.message = message
    adapter = PymunkAdapter(ShortNavigation())
    with pytest.raises(ControllerError):
        run_slm_episode(adapter, client, Tokenizer(), 0, task="navigation")
    assert len(adapter.measurements) == 1


@pytest.mark.parametrize("repeat", [0, -1, 1.5, True])
def test_repeat_requires_positive_integer(repeat):
    with pytest.raises(ValueError, match="positive integer"):
        controller(Inference(), action_repeat=repeat)
    with pytest.raises(ValueError, match="positive integer"):
        run_episode(None, None, 0, action_repeat=repeat)
