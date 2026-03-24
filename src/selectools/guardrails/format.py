"""
FormatGuardrail — validate output format constraints.

Checks that content is valid JSON, contains required keys, or matches
an expected structure.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .base import Guardrail, GuardrailAction, GuardrailResult


class FormatGuardrail(Guardrail):
    """Validate that content matches expected format constraints.

    Args:
        require_json: If ``True``, content must parse as valid JSON. Default: ``False``.
        required_keys: JSON object must contain all of these top-level keys.
        max_length: Maximum character count. Default: ``None`` (unlimited).
        min_length: Minimum character count. Default: ``None`` (no minimum).
        action: ``block``, ``rewrite``, or ``warn``.  Default: ``block``.
    """

    name = "format"

    def __init__(
        self,
        *,
        require_json: bool = False,
        required_keys: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> None:
        self._require_json = require_json
        self._required_keys = required_keys or []
        self._max_length = max_length
        self._min_length = min_length
        self.action = action

    def check(self, content: str) -> GuardrailResult:
        if self._min_length is not None and len(content) < self._min_length:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content length {len(content)} below minimum {self._min_length}",
                guardrail_name=self.name,
            )

        if self._max_length is not None and len(content) > self._max_length:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Content length {len(content)} exceeds maximum {self._max_length}",
                guardrail_name=self.name,
            )

        if self._require_json:
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError) as exc:
                return GuardrailResult(
                    passed=False,
                    content=content,
                    reason=f"Invalid JSON: {exc}",
                    guardrail_name=self.name,
                )

            if self._required_keys and isinstance(parsed, dict):
                missing = [k for k in self._required_keys if k not in parsed]
                if missing:
                    return GuardrailResult(
                        passed=False,
                        content=content,
                        reason=f"Missing required JSON keys: {', '.join(missing)}",
                        guardrail_name=self.name,
                    )
            elif self._required_keys:
                return GuardrailResult(
                    passed=False,
                    content=content,
                    guardrail_name=self.name,
                    reason="JSON value is not an object; cannot check required keys",
                )

        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


__all__ = ["FormatGuardrail"]
