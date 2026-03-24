"""
Prompt templating for the TOOL_CALL contract.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .tools import Tool

DEFAULT_SYSTEM_INSTRUCTIONS = """You are an assistant that can call tools when helpful.

Tool call contract:
- Emit TOOL_CALL with JSON: {"tool_name": "<name>", "parameters": {...}}
- Include every required parameter. Ask for missing details instead of guessing.
- Wait for tool results before giving a final answer.
- Do not invent tool outputs; only report what was returned.
- Keep tool payloads compact (<=8k chars) and emit one tool call at a time.
"""

REASONING_STRATEGIES: Dict[str, str] = {
    "react": (
        "\n\nReasoning strategy — ReAct (Reason + Act):\n"
        "For each step, follow the Thought → Action → Observation cycle:\n"
        "1. Thought: Explain your reasoning about what to do next\n"
        "2. Action: Call a tool or provide your final answer\n"
        "3. Observation: Analyze the tool result before proceeding\n"
        "Always think before acting. Never skip the Thought step."
    ),
    "cot": (
        "\n\nReasoning strategy — Chain of Thought:\n"
        "Think step by step. Before taking any action, write out your "
        "complete chain of reasoning. Break complex problems into smaller "
        "steps and solve each one before moving to the next. Show your "
        "work explicitly."
    ),
    "plan_then_act": (
        "\n\nReasoning strategy — Plan Then Act:\n"
        "First, create a numbered plan of all steps needed to complete the task. "
        "Then execute each step in order, marking them as done. "
        "If a step fails, revise the plan before continuing. "
        "Always present the plan before taking any action."
    ),
}


class PromptBuilder:
    """Render a system prompt that includes tool schemas."""

    def __init__(
        self,
        base_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        reasoning_strategy: Optional[str] = None,
    ):
        if reasoning_strategy is not None and reasoning_strategy not in REASONING_STRATEGIES:
            raise ValueError(
                f"Invalid reasoning_strategy '{reasoning_strategy}'. "
                f"Valid values: {sorted(REASONING_STRATEGIES.keys())}"
            )
        self.base_instructions = base_instructions
        self.reasoning_strategy = reasoning_strategy

    def build(self, tools: List[Tool]) -> str:
        tool_blocks = []
        for tool in tools:
            tool_blocks.append(json.dumps(tool.schema(), indent=2))
        tools_text = "\n\n".join(tool_blocks)

        prompt = (
            f"{self.base_instructions.strip()}\n\n"
            f"Available tools (JSON schema):\n\n{tools_text}\n\n"
            "If a relevant tool exists, respond with a TOOL_CALL first. "
            "When no tool is useful, answer directly."
        )

        if self.reasoning_strategy is not None:
            prompt += REASONING_STRATEGIES[self.reasoning_strategy]

        return prompt


__all__ = ["PromptBuilder", "DEFAULT_SYSTEM_INSTRUCTIONS", "REASONING_STRATEGIES"]
