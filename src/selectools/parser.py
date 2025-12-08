"""
Parser for TOOL_CALL directives emitted by language models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import ToolCall


@dataclass
class ParseResult:
    """Result of attempting to parse a tool call."""

    tool_call: Optional[ToolCall]
    raw_text: str


class ToolCallParser:
    """Robustly extract TOOL_CALL directives from model output."""

    def __init__(self, marker: str = "TOOL_CALL", max_payload_chars: int = 8000):
        self.marker = marker
        self.max_payload_chars = max_payload_chars

    def parse(self, text: str) -> ParseResult:
        """
        Attempt to parse a tool call from the provided text.

        Supports fenced code blocks, inline JSON, and newline-heavy outputs.
        """
        candidates = self._extract_candidate_blocks(text)
        for candidate in candidates:
            if self.max_payload_chars and len(candidate) > self.max_payload_chars:
                continue
            tool_data = self._load_json(candidate)
            if not tool_data:
                continue
            tool_name = tool_data.get("tool_name") or tool_data.get("tool") or tool_data.get("name")
            parameters: Dict[str, Any] = (
                tool_data.get("parameters") or tool_data.get("params") or {}
            )
            if tool_name:
                return ParseResult(
                    tool_call=ToolCall(tool_name=tool_name, parameters=parameters), raw_text=text
                )
        return ParseResult(tool_call=None, raw_text=text)

    def _extract_candidate_blocks(self, text: str) -> List[str]:
        """Pull out all JSON substrings that might contain the TOOL_CALL payload."""
        blocks: List[str] = []

        marker_positions = [m.start() for m in re.finditer(self.marker, text)]
        for pos in marker_positions:
            subset = text[pos:]
            blocks.extend(self._find_balanced_json(subset))

        fenced_blocks = re.findall(r"```.*?```", text, re.DOTALL)
        for block in fenced_blocks:
            if self.marker in block or "tool_name" in block or "parameters" in block:
                cleaned = block.strip("` \n")
                blocks.extend(self._find_balanced_json(cleaned))

        if not blocks:
            blocks.extend(self._find_balanced_json(text))

        # Deduplicate while preserving order
        deduped = []
        seen = set()
        for block in blocks:
            if block in seen:
                continue
            deduped.append(block)
            seen.add(block)
        return deduped

    def _load_json(self, candidate: str) -> Optional[Dict[str, Any]]:
        """Attempt JSON parsing with lenient fallbacks."""
        normalized = candidate
        if self.marker in normalized:
            normalized = normalized.split(self.marker, maxsplit=1)[-1]
        normalized = normalized.strip("` \n:")

        attempts = [
            normalized,
            normalized.replace("'", '"'),
            normalized.replace("\n", "\\n"),
        ]
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        return None

    def _find_balanced_json(self, text: str) -> List[str]:
        """Collect balanced JSON-like substrings from text."""
        candidates: List[str] = []
        starts = [m.start() for m in re.finditer(r"\{", text)]
        for start in starts:
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1])
                        break
        return candidates


__all__ = ["ToolCallParser", "ParseResult"]
