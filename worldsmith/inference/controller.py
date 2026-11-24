"""Frozen SLM control of the existing Pymunk heading/throttle action contract."""

import json
import math
from copy import deepcopy
from dataclasses import asdict

from worldsmith.api import Action, Observation
from worldsmith.benchmark.runner import run_episode

from .config import MODEL, MODEL_REVISION
from .prompts import QUERY_PROMPT, STARTING_PROMPT

CONTEXT_WINDOW = 8192  # Matches the pinned vLLM server's max-model-len.
RESPONSE_TOKENS = 256
ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Set the robot's absolute heading and speed fraction.",
        "parameters": {
            "type": "object",
            "properties": {
                "heading": {"type": "number", "description": "Radians from the positive x axis."},
                "throttle": {"type": "number", "minimum": 0, "maximum": 1,
                             "description": "Fraction of robot_max_speed."},
            },
            "required": ["heading", "throttle"],
            "additionalProperties": False,
        },
    },
}


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)


def count_prompt_tokens(tokenizer, messages) -> int:
    # Match vLLM's chat preprocessing: templates receive parsed tool arguments,
    # while the OpenAI API requires JSON strings on the wire.
    template_messages = deepcopy(messages)
    for message in template_messages:
        if message["role"] == "assistant":
            for call in message["tool_calls"]:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return len(tokenizer.apply_chat_template(
        template_messages, tools=[ACT_TOOL], tokenize=True, add_generation_prompt=True,
    ))


class SLMController:
    def __init__(
        self, client, tokenizer, *, task_description: str, max_steps: int,
        action_repeat: int = 5, context_window: int = CONTEXT_WINDOW,
        response_tokens: int = RESPONSE_TOKENS,
    ):
        if type(action_repeat) is not int or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not 0 < response_tokens < context_window <= CONTEXT_WINDOW:
            raise ValueError("Require 0 < response_tokens < context_window <= server context window")
        self.client = client
        self.tokenizer = tokenizer
        self.action_repeat = action_repeat
        self.max_steps = max_steps
        self.context_window = context_window
        self.response_tokens = response_tokens
        self.starting = {"role": "system", "content": STARTING_PROMPT.format(
            task_description=task_description, action_repeat=action_repeat, max_steps=max_steps,
        )}
        self.interactions = []
        self.current_step = 0
        self.decisions = 0

    def __call__(self, observation: Observation) -> Action:
        query = {"role": "user", "content": QUERY_PROMPT.format(
            current_step=self.current_step, max_steps=self.max_steps,
            remaining_steps=self.max_steps - self.current_step, elapsed_time=observation.time,
            observation=json.dumps(asdict(observation), allow_nan=False),
        )}
        while True:
            messages = [self.starting, *[message for group in self.interactions for message in group], query]
            if count_prompt_tokens(self.tokenizer, messages) + self.response_tokens <= self.context_window:
                break
            if not self.interactions:
                raise ValueError("Mandatory controller context exceeds token budget")
            # Remove the whole query/call/result group to keep tool-call IDs paired.
            del self.interactions[0]
        reply = self.client.chat.completions.create(
            model=MODEL, messages=messages, tools=[ACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "act"}},
            parallel_tool_calls=False, temperature=0, max_tokens=self.response_tokens,
        )
        # vLLM 0.10.2 reports "stop" for named tool choices; automatic calls use "tool_calls".
        if len(reply.choices) != 1 or reply.choices[0].finish_reason not in ("stop", "tool_calls"):
            raise ValueError("Expected exactly one complete action tool call")
        assistant = reply.choices[0].message
        calls = assistant.tool_calls
        if not calls or len(calls) != 1:
            raise ValueError("Expected exactly one action tool call")
        call = calls[0]
        if call.type != "function" or call.function.name != "act" or not call.id:
            raise ValueError("Unexpected action tool or missing tool-call ID")
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict) or arguments.keys() != {"heading", "throttle"}:
            raise ValueError("Action arguments must contain heading and throttle")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in arguments.values()):
            raise ValueError("Action arguments must be finite JSON numbers")
        if not 0 <= arguments["throttle"] <= 1:
            raise ValueError("Action throttle must be in [0, 1]")
        self.interactions.append([query, assistant.model_dump(exclude_none=True)])
        self.decisions += 1
        return Action(**arguments)

    def record_result(self, result: dict):
        self.current_step = result["current_step"]
        self.interactions[-1].append({
            "role": "tool", "tool_call_id": self.interactions[-1][1]["tool_calls"][0]["id"],
            "content": json.dumps(result),
        })


def run_slm_episode(adapter, client, tokenizer, seed: int, *, task: str, action_repeat: int = 5):
    """Use the shared runner and a fresh conversation for every episode."""
    controller = None

    def factory():
        nonlocal controller
        metadata = adapter.metadata()
        controller = SLMController(
            client, tokenizer, task_description=metadata["objective"],
            max_steps=metadata["max_episode_steps"], action_repeat=action_repeat,
        )
        return controller

    episode = run_episode(
        adapter, factory, seed, task=task, action_repeat=action_repeat,
        action_result=lambda result: controller.record_result(result),
    )
    return episode, controller.decisions
