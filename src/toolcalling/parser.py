"""
Parser for TOOL_CALL directives emitted by language models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import ToolCall


@dataclass
class ParseResult:
    """Result of attempting to parse a tool call."""

    tool_call: Optional[ToolCall]
    raw_text: str


class ToolCallParser:
    """Robustly extract TOOL_CALL directives from model output."""

    def __init__(self, marker: str = "TOOL_CALL"):
        self.marker = marker

    def parse(self, text: str) -> ParseResult:
        """
        Attempt to parse a tool call from the provided text.

        Supports fenced code blocks, inline JSON, and newline-heavy outputs.
        """
        if self.marker not in text:
            return ParseResult(tool_call=None, raw_text=text)

        candidate = self._extract_candidate_block(text)
        if candidate is None:
            return ParseResult(tool_call=None, raw_text=text)

        tool_data = self._load_json(candidate)
        if not tool_data:
            return ParseResult(tool_call=None, raw_text=text)

        tool_name = tool_data.get("tool_name") or tool_data.get("tool")
        parameters: Dict[str, Any] = tool_data.get("parameters") or tool_data.get("params") or {}

        if not tool_name:
            return ParseResult(tool_call=None, raw_text=text)

        return ParseResult(tool_call=ToolCall(tool_name=tool_name, parameters=parameters), raw_text=text)

    def _extract_candidate_block(self, text: str) -> Optional[str]:
        """Pull out the JSON substring that likely contains the TOOL_CALL payload."""
        if "```" in text:
            fenced_blocks = re.findall(r"```.*?```", text, re.DOTALL)
            for block in fenced_blocks:
                if self.marker in block:
                    text = block.strip("` \n")
                    break

        try:
            subset = text[text.index(self.marker) :]
        except ValueError:
            return None

        json_match = re.search(r"\{.*\}", subset, re.DOTALL)
        if not json_match:
            return None
        return json_match.group()

    def _load_json(self, candidate: str) -> Optional[Dict[str, Any]]:
        """Attempt JSON parsing with lenient fallbacks."""
        attempts = [
            candidate,
            candidate.replace("'", '"'),
            candidate.replace("\n", "\\n"),
        ]
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        return None


__all__ = ["ToolCallParser", "ParseResult"]
