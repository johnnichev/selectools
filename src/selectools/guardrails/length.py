"""
LengthGuardrail — enforce minimum and maximum content length.

Lightweight guardrail that checks character or word count.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import Guardrail, GuardrailAction, GuardrailResult


class LengthGuardrail(Guardrail):
    """Enforce content length constraints.

    Args:
        max_chars: Maximum number of characters.  Default: ``None`` (unlimited).
        min_chars: Minimum number of characters.  Default: ``None``.
        max_words: Maximum number of words.  Default: ``None`` (unlimited).
        min_words: Minimum number of words.  Default: ``None``.
        action: ``block``, ``rewrite`` (truncate to max), or ``warn``.
            Default: ``block``.
    """

    name = "length"

    def __init__(
        self,
        *,
        max_chars: Optional[int] = None,
        min_chars: Optional[int] = None,
        max_words: Optional[int] = None,
        min_words: Optional[int] = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> None:
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._max_words = max_words
        self._min_words = min_words
        self.action = action
        self._word_re = re.compile(r"\S+")

    def check(self, content: str) -> GuardrailResult:
        char_count = len(content)
        words = self._word_re.findall(content)
        word_count = len(words)

        if self._min_chars is not None and char_count < self._min_chars:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content has {char_count} chars, minimum is {self._min_chars}",
                guardrail_name=self.name,
            )

        if self._max_chars is not None and char_count > self._max_chars:
            if self.action == GuardrailAction.REWRITE:
                truncated = content[: self._max_chars]
                return GuardrailResult(
                    passed=False,
                    content=truncated,
                    reason=f"Content truncated from {char_count} to {self._max_chars} chars",
                    guardrail_name=self.name,
                )
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content has {char_count} chars, maximum is {self._max_chars}",
                guardrail_name=self.name,
            )

        if self._min_words is not None and word_count < self._min_words:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content has {word_count} words, minimum is {self._min_words}",
                guardrail_name=self.name,
            )

        if self._max_words is not None and word_count > self._max_words:
            if self.action == GuardrailAction.REWRITE:
                truncated = " ".join(words[: self._max_words])
                return GuardrailResult(
                    passed=False,
                    content=truncated,
                    reason=f"Content truncated from {word_count} to {self._max_words} words",
                    guardrail_name=self.name,
                )
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content has {word_count} words, maximum is {self._max_words}",
                guardrail_name=self.name,
            )

        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


__all__ = ["LengthGuardrail"]
