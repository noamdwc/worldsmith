"""Editable, generic prompts for the minimal robot controller."""

STARTING_PROMPT = """You are a robot controller.

Your task:
{task_description}

Interact with the environment using the available action tool.
Select exactly one act tool call per decision.

Each action you select will be applied for up to {action_repeat} environment steps
before you receive a new observation. Take this into account when selecting actions.

Your goal is to complete the task within {max_steps} environment steps.
"""

QUERY_PROMPT = """Current step: {current_step}/{max_steps}
Remaining steps: {remaining_steps}
Elapsed simulation time: {elapsed_time}

Current observation:
{observation}

Select your next action.
"""
