"""
Prompt templating for the TOOL_CALL contract.
"""

from __future__ import annotations

import json
from typing import List

from .tools import Tool

DEFAULT_SYSTEM_INSTRUCTIONS = """You are an assistant that can call tools when helpful.

Tool call contract:
- Emit TOOL_CALL with JSON: {"tool_name": "<name>", "parameters": {...}}
- Include every required parameter. Ask for missing details instead of guessing.
- Wait for tool results before giving a final answer.
- Do not invent tool outputs; only report what was returned.
- Keep tool payloads compact (<=8k chars) and emit one tool call at a time.
"""


class PromptBuilder:
    """Render a system prompt that includes tool schemas."""

    def __init__(self, base_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS):
        self.base_instructions = base_instructions

    def build(self, tools: List[Tool]) -> str:
        tool_blocks = []
        for tool in tools:
            tool_blocks.append(json.dumps(tool.schema(), indent=2))
        tools_text = "\n\n".join(tool_blocks)

        return (
            f"{self.base_instructions.strip()}\n\n"
            f"Available tools (JSON schema):\n\n{tools_text}\n\n"
            "If a relevant tool exists, respond with a TOOL_CALL first. "
            "When no tool is useful, answer directly."
        )


__all__ = ["PromptBuilder", "DEFAULT_SYSTEM_INSTRUCTIONS"]
