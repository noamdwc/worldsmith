"""Observable SLM conversation, action, and shared-runner behavior."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from worldsmith.api import Action
from worldsmith.benchmark.adapters.pymunk_adapter import PymunkAdapter
from worldsmith.benchmark.results import snapshot
from worldsmith.benchmark.runner import ControllerError, run_episode
from worldsmith.benchmark.tasks.navigation import Navigation
from worldsmith.harness.controller import ACT_TOOL, SLMController, count_prompt_tokens, run_slm_episode


class Tokenizer:
    """Variable token cost, including the tool schema and chat formatting."""

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        assert tools == [ACT_TOOL] and tokenize and add_generation_prompt
        for message in messages:
            if message["role"] == "assistant":
                assert isinstance(message["tool_calls"][0]["function"]["arguments"], dict)
        return list(range(len(json.dumps(messages)) + len(json.dumps(tools)) + 10))


class Inference:
    def __init__(self):
        self.requests = []
        self.message = None
        self.finish_reason = "stop"
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


@pytest.mark.parametrize("options,limit,success,decision_steps,last_block", [
    ({}, 12, False, list(range(12)), 1),
    ({"action_repeat": 5}, 12, False, [0, 5, 10], 2),
    ({"action_repeat": 5}, 3, True, [0], 3),
])
def test_shared_runner_decisions_trace_and_partial_feedback(options, limit, success, decision_steps, last_block):
    class Task(ShortNavigation):
        def succeeded(self, measurements):
            return success and len(measurements) >= 4

    observations, commands, feedback = [], [], []

    def control(observation):
        observations.append(snapshot(observation))
        action = Action(heading=len(commands) * 0.1, throttle=0.5)
        commands.append(action)
        return action

    adapter = PymunkAdapter(Task())
    episode = run_episode(adapter, lambda: control, 0, action_result=feedback.append, **options)
    assert episode["length"] == len(adapter.measurements) - 1 == limit
    assert observations == [episode["steps"][step]["observation"] for step in decision_steps]
    assert len(commands) == len(decision_steps)
    for start, end, command in zip(decision_steps, [*decision_steps[1:], limit], commands, strict=True):
        assert all(row["action"] == snapshot(command) for row in episode["steps"][start:end])
    assert len(feedback) == len(decision_steps)
    assert feedback[-1] == {
        "current_step": limit, "executed_steps": last_block,
        "terminated": success, "truncated": not success,
    }

    # Replay recorded commands to verify every input observation and resulting
    # simulator state, including observations between model decisions.
    replay = PymunkAdapter(Task())
    observation, _ = replay.reset(0)
    for row in episode["steps"]:
        assert row["observation"] == snapshot(observation)
        observation, _, terminated, truncated, info = replay.step(Action(**row["action"]))
        assert row["info"] == snapshot(info)
        assert (row["terminated"], row["truncated"]) == (terminated, truncated)
    assert episode["final_observation"] == snapshot(observation)
    assert adapter.world.ground_truth() == replay.world.ground_truth()


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
    control.context_window = count_prompt_tokens(Tokenizer(), recent) + control.response_tokens
    control.record_result({"current_step": 5, "executed_steps": 5, "terminated": False, "truncated": False})
    control(obs)
    messages = client.requests[-1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2]["tool_calls"][0]["id"] == "call-3"
    assert messages[3]["tool_call_id"] == "call-3"
    assert messages[0] == three[0]
    assert messages[-1] == three[-1]
    assert count_prompt_tokens(Tokenizer(), messages) + control.response_tokens <= control.context_window


def test_larger_current_observation_evicts_all_history():
    client = Inference()
    control = controller(client)
    obs = observation()
    complete_decision(control, obs, 5)
    control(obs)
    initial = client.requests[-1]["messages"]
    # Leave room for mandatory context only; a larger new observation forces eviction.
    mandatory = [initial[0], initial[-1]]
    control.context_window = count_prompt_tokens(Tokenizer(), mandatory) + control.response_tokens + 100
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


@pytest.mark.parametrize("size", [3, 4, 6, 7, 12, 14])
def test_native_vector_dimensions_and_bounds(size):
    import numpy as np
    from worldsmith.harness.action_tools import decode_vector_action, vector_tool

    low, high = -np.ones(size), np.ones(size)
    tool = vector_tool(low, high, "Native control")
    assert tool["function"]["name"] == "act"
    assert tool["function"]["parameters"]["properties"]["action"]["maxItems"] == size
    action = decode_vector_action({"action": [0.25] * size}, low=low, high=high, dtype=np.float32)
    assert action.shape == (size,) and action.dtype == np.float32
    for invalid in ([0] * (size + 1), [True] * size, [float("nan")] * size, [1.01] * size):
        with pytest.raises(ValueError):
            decode_vector_action({"action": invalid}, low=low, high=high, dtype=np.float32)


@pytest.mark.parametrize("group,task,size", [
    ("maniskill", "push_cube", 4), ("libero", "open_middle_drawer", 7),
    ("robosuite", "handover_hammer", 14), ("robosuite", "wipe_table", 6),
])
def test_native_observations_actions_time_and_shared_lifecycle(group, task, size):
    import numpy as np

    class Adapter:
        def __init__(self):
            self.low, self.high = -np.ones(size), np.ones(size)
            space = SimpleNamespace(low=self.low, high=self.high, dtype=np.float32)
            native = SimpleNamespace(
                elapsed_steps=SimpleNamespace(item=lambda: self.steps), control_timestep=0.05,
            )
            self.env = SimpleNamespace(
                sim=SimpleNamespace(data=SimpleNamespace(time=0)), action_space=space, unwrapped=native,
            )
            self.actions = []

        def metadata(self):
            return {"max_episode_steps": 6, "action_mode": "pd_ee_delta_pos" if group == "maniskill" else "OSC_POSE"}

        def reset(self, seed):
            self.steps = 0
            return {"agent": {"qpos": np.zeros((1, size))}}, {}

        def encode_action(self, action):
            assert action.shape == (size,)
            assert action.dtype == (np.float32 if group == "maniskill" else np.float64)
            return action.tolist()

        def step(self, action):
            self.steps += 1
            self.actions.append(action.copy())
            self.env.sim.data.time += 0.05
            return {"agent": {"qpos": np.ones((1, size))}}, None, False, self.steps == 6, {}

    requests = []

    def create(**request):
        requests.append(deepcopy(request))
        assert len(request["tools"]) == 1
        assert request["tools"][0]["function"]["parameters"]["properties"]["action"]["minItems"] == size
        message = ChatCompletionMessage(role="assistant", tool_calls=[{
            "id": str(len(requests)), "type": "function", "function": {
                "name": "act", "arguments": json.dumps({"action": [0.1] * size}),
            },
        }])
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [0])
    adapter = Adapter()
    episode, decisions = run_slm_episode(adapter, client, tokenizer, 0, task=task)
    assert episode["length"] == 6 and decisions == 2 and len(adapter.actions) == 6
    assert "Current step: 5/6" in requests[1]["messages"][-1]["content"]
    assert "Elapsed simulation time: 0.25" in requests[1]["messages"][-1]["content"]
    assert '"qpos": [[1.0' in requests[1]["messages"][-1]["content"]


def test_pymunk_evaluation_uses_owner_adapter_without_autoresearch(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from worldsmith.harness import evaluate

    output = tmp_path / "navigation.json"
    monkeypatch.setattr(evaluate, "load_tokenizer", Tokenizer)
    monkeypatch.setattr(evaluate, "create_client", lambda: nullcontext(Inference()))
    monkeypatch.setattr(evaluate, "TASK_CATALOG", {"navigation": lambda: pytest.fail("Loaded legacy controller")})
    monkeypatch.setattr("sys.argv", ["evaluate", "--task", "navigation", "--output", str(output)])
    assert evaluate.main() == 0
    result = json.loads(output.read_text())
    assert result["status"] == "completed"
    assert result["environment_steps"] == 80
    assert result["slm_decisions"] == result["model_requests"] == 16
