"""
TopicGuardrail — block or warn when content touches denied topics.

Uses keyword matching by default.  For higher accuracy, provide an
LLM provider via ``provider`` and ``model`` for classification.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .base import Guardrail, GuardrailAction, GuardrailResult


class TopicGuardrail(Guardrail):
    """Reject content that mentions denied topics.

    Args:
        deny: List of topic keywords/phrases to block.
        action: ``block``, ``rewrite``, or ``warn``. Default: ``block``.
        case_sensitive: Match topics case-sensitively. Default: ``False``.
    """

    name = "topic"

    def __init__(
        self,
        deny: Optional[List[str]] = None,
        *,
        action: GuardrailAction = GuardrailAction.BLOCK,
        case_sensitive: bool = False,
    ) -> None:
        self.deny = deny or []
        self.action = action
        self._case_sensitive = case_sensitive
        flags = 0 if case_sensitive else re.IGNORECASE
        self._patterns = [
            re.compile(r"\b" + re.escape(topic) + r"\b", flags) for topic in self.deny
        ]

    def check(self, content: str) -> GuardrailResult:
        matched: List[str] = []
        for pattern, topic in zip(self._patterns, self.deny):
            if pattern.search(content):
                matched.append(topic)

        if matched:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Denied topics detected: {', '.join(matched)}",
                guardrail_name=self.name,
            )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


__all__ = ["TopicGuardrail"]
