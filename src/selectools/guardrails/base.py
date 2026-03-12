"""
Base guardrail protocol and result types.

Guardrails validate content before (input) and after (output) LLM calls.
Each guardrail returns a ``GuardrailResult`` indicating whether the content
passed, and optionally provides rewritten content or a rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GuardrailAction(str, Enum):
    """Action to take when a guardrail check fails."""

    BLOCK = "block"
    REWRITE = "rewrite"
    WARN = "warn"


@dataclass
class GuardrailResult:
    """Result of a single guardrail check.

    Attributes:
        passed: Whether the content passed the guardrail check.
        content: The (possibly rewritten) content.  When ``passed`` is
            ``True``, this equals the original.  When the guardrail rewrites,
            this contains the sanitised version.
        reason: Human-readable explanation when the check fails.
        guardrail_name: Name of the guardrail that produced this result.
    """

    passed: bool
    content: str
    reason: Optional[str] = None
    guardrail_name: Optional[str] = None


class Guardrail:
    """Base class for all guardrails.

    Subclass and override :meth:`check` to implement custom validation.

    Attributes:
        name: Human-readable name for logging / tracing.
        action: What to do on failure — ``block`` (raise), ``rewrite``
            (return sanitised content), or ``warn`` (log and continue).
    """

    name: str = "guardrail"
    action: GuardrailAction = GuardrailAction.BLOCK

    def check(self, content: str) -> GuardrailResult:
        """Evaluate *content* and return a :class:`GuardrailResult`.

        Subclasses **must** override this method.
        """
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)


class GuardrailError(Exception):
    """Raised when a guardrail with ``action=block`` rejects content."""

    def __init__(self, guardrail_name: str, reason: str) -> None:
        self.guardrail_name = guardrail_name
        self.reason = reason
        super().__init__(f"Guardrail '{guardrail_name}' blocked: {reason}")


__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailError",
    "GuardrailResult",
]
