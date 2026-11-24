"""Real chat, automatic tool selection, and one tool-result round trip."""

import json
import math

from .client import create_client
from .config import MODEL

DISTANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_distance",
        "description": "Calculate the Euclidean distance between two 2D points.",
        "parameters": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in ("x1", "y1", "x2", "y2")},
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
    },
}


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def run_smoke(client) -> dict:
    chat = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        temperature=0, max_tokens=64,
    )
    greeting = chat.choices[0].message.content
    if not greeting or not greeting.strip() or chat.choices[0].finish_reason != "stop":
        raise ValueError("Chat smoke test returned no complete text response")

    messages = [{"role": "user", "content": (
        "Use calculate_distance to find the distance from (0, 0) to (3, 4)."
    )}]
    reply = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[DISTANCE_TOOL], tool_choice="auto",
        temperature=0, max_tokens=256,
    )
    assistant = reply.choices[0].message
    calls = assistant.tool_calls
    if reply.choices[0].finish_reason != "tool_calls" or not calls or len(calls) != 1:
        raise ValueError("Tool smoke test expected exactly one complete tool call")
    call = calls[0]
    if call.type != "function" or call.function.name != "calculate_distance" or not call.id:
        raise ValueError("Tool smoke test returned an unexpected function or missing call ID")
    arguments = json.loads(call.function.arguments)
    expected = {"x1": 0, "y1": 0, "x2": 3, "y2": 4}
    if not isinstance(arguments, dict) or arguments.keys() != expected.keys():
        raise ValueError("Tool arguments do not match the distance schema")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in arguments.values()):
        raise ValueError("Tool arguments must be finite JSON numbers")
    if arguments != expected:
        raise ValueError(f"Tool arguments do not match the requested points: {arguments}")
    distance = calculate_distance(**arguments)
    final = client.chat.completions.create(
        model=MODEL,
        messages=[*messages, assistant.model_dump(exclude_none=True), {
            "role": "tool", "tool_call_id": call.id, "content": json.dumps(distance),
        }],
        tools=[DISTANCE_TOOL], tool_choice="none", temperature=0, max_tokens=128,
    )
    answer = final.choices[0].message.content
    if not answer or not answer.strip() or final.choices[0].finish_reason != "stop":
        raise ValueError("Tool result round trip returned no complete text response")
    return {"chat": greeting, "tool": call.function.name, "arguments": arguments,
            "distance": distance, "answer": answer}


def main():
    from .server import ModelServer

    with ModelServer(), create_client() as client:
        print(json.dumps(run_smoke(client), indent=2))


if __name__ == "__main__":
    main()
