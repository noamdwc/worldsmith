"""Native vector action schemas and validation for the existing 3D simulators."""

import math

import numpy as np


def vector_tool(low, high, description):
    return {
        "type": "function",
        "function": {
            "name": "act", "description": description,
            "parameters": {
                "type": "object",
                "properties": {"action": {
                    "type": "array", "minItems": len(low), "maxItems": len(low),
                    "items": {"type": "number", "minimum": float(min(low)), "maximum": float(max(high))},
                    "description": f"Native action vector. Lower bounds: {low.tolist()}; upper bounds: {high.tolist()}.",
                }},
                "required": ["action"], "additionalProperties": False,
            },
        },
    }


def decode_vector_action(arguments, *, low, high, dtype):
    if not isinstance(arguments, dict) or arguments.keys() != {"action"}:
        raise ValueError("Action arguments must contain the native action vector")
    values = arguments["action"]
    if not isinstance(values, list) or len(values) != len(low):
        raise ValueError(f"Action must have {len(low)} entries")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Action entries must be finite JSON numbers")
    if any(value < lower or value > upper for value, lower, upper in zip(values, low, high, strict=True)):
        raise ValueError("Action entries exceed native bounds")
    return np.array(values, dtype=dtype)


def action_description(group, mode):
    if group == "maniskill":
        if mode == "pd_joint_delta_pos":
            return ("Set 12 leg joint offsets, each scaled by 0.7 radians from the current joint position. "
                    "Channels: hips, thighs, calves; each in front-left, front-right, rear-left, rear-right order.")
        if mode == "pd_ee_delta_pose":
            return ("Set base-frame XYZ translation deltas (0.1 m full scale), three rotation commands "
                    "(rotation vector clipped to unit length and multiplied by -0.1 for XYZ Euler increments), "
                    "and gripper (-1 closes, +1 opens).")
        if mode == "pd_ee_delta_pos":
            return ("Set base-frame XYZ translation deltas (0.1 m full scale), followed by gripper "
                    "(-1 closes, +1 opens) when present. A three-channel action controls only XYZ.")
        raise ValueError(f"Unsupported ManiSkill action mode: {mode}")
    return ("Set world-frame tool pose deltas. Each arm uses XYZ translation (0.05 m full scale), "
            "then axis-angle rotation XYZ (0.5 radians full scale), then gripper (negative opens, positive "
            "closes, zero holds). For two arms, concatenate pickup then receiving arm. A six-channel "
            "action controls pose only, without a gripper.")
