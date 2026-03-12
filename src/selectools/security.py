"""
Tool output screening — detect prompt injection in tool results.

When a tool returns untrusted content (web pages, emails, user-uploaded
files), that content is fed back to the LLM as context.  An attacker can
embed instructions in the content that hijack the agent's behaviour.

This module provides pattern-based screening that runs *after* tool
execution and *before* the result is appended to conversation history.
Suspicious outputs are replaced with a safe placeholder.

Usage:
    Mark individual tools with ``screen_output=True`` in the ``@tool``
    decorator, or enable screening globally via
    ``AgentConfig(screen_tool_output=True)``.

    Custom patterns can be supplied via ``AgentConfig(output_screening_patterns=[...])``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_DEFAULT_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*:\s*you", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE),
    re.compile(r"</s>", re.IGNORECASE),
    re.compile(r"IMPORTANT:\s*override", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
]


@dataclass
class ScreeningResult:
    """Result of screening a tool output.

    Attributes:
        safe: Whether the content passed screening.
        content: The (possibly replaced) content.
        matched_patterns: List of pattern descriptions that matched.
    """

    safe: bool
    content: str
    matched_patterns: List[str]


def screen_output(
    content: str,
    *,
    extra_patterns: Optional[List[str]] = None,
) -> ScreeningResult:
    """Screen tool output for prompt injection patterns.

    Args:
        content: Raw tool output string.
        extra_patterns: Additional regex patterns (strings) to check.

    Returns:
        A :class:`ScreeningResult` indicating whether the content is safe.
    """
    patterns = list(_DEFAULT_INJECTION_PATTERNS)
    if extra_patterns:
        for pat in extra_patterns:
            patterns.append(re.compile(pat, re.IGNORECASE))

    matched: List[str] = []
    for pattern in patterns:
        if pattern.search(content):
            matched.append(pattern.pattern)

    if matched:
        safe_content = (
            "[Tool output blocked: potential prompt injection detected. "
            f"{len(matched)} suspicious pattern(s) found.]"
        )
        return ScreeningResult(
            safe=False,
            content=safe_content,
            matched_patterns=matched,
        )

    return ScreeningResult(safe=True, content=content, matched_patterns=[])


__all__ = ["ScreeningResult", "screen_output"]
