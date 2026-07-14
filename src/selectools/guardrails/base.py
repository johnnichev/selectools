"""
Base guardrail protocol and result types.

Guardrails validate content before (input) and after (output) LLM calls.
Each guardrail returns a ``GuardrailResult`` indicating whether the content
passed, and optionally provides rewritten content or a rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from selectools.stability import stable


@stable
class GuardrailAction(str, Enum):
    """Action to take when a guardrail check fails."""

    BLOCK = "block"
    REWRITE = "rewrite"
    WARN = "warn"


@stable
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
        trips: ``(guardrail_name, action)`` pairs for every guardrail that
            triggered while producing this result (v1.2). Unlike the
            comma-joined ``guardrail_name``, this survives names containing
            commas and records each guardrail's own action.
    """

    passed: bool
    content: str
    reason: Optional[str] = None
    guardrail_name: Optional[str] = None
    trips: List[Tuple[str, str]] = field(default_factory=list)


@stable
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

    async def acheck(self, content: str) -> GuardrailResult:
        """Async version of :meth:`check`.

        The default runs the sync ``check()`` in a thread executor to avoid
        blocking the event loop.  Override for native async implementations
        (e.g. LLM-based guardrails with async provider calls).
        """
        import asyncio

        return await asyncio.to_thread(self.check, content)


@stable
class GuardrailError(Exception):
    """Raised when a guardrail with ``action=block`` rejects content.

    Attributes:
        guardrail_name: Name of the blocking guardrail.
        reason: Human-readable rejection reason.
        prior_trips: ``(guardrail_name, action)`` pairs for guardrails that
            triggered (rewrite/warn) earlier in the same chain, before the
            block — so observability does not lose trips that already
            mutated content (v1.2).
        agent_trace: When the exception propagates out of an agent run, the
            agent attaches the run's ``AgentTrace`` here (the run never
            returns an ``AgentResult`` on a block, so this is the only way
            to inspect the recorded ``GUARDRAIL`` steps). ``None`` when
            raised outside an agent (v1.2).
    """

    def __init__(
        self,
        guardrail_name: str,
        reason: str,
        prior_trips: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.guardrail_name = guardrail_name
        self.reason = reason
        self.prior_trips: List[Tuple[str, str]] = list(prior_trips or [])
        self.agent_trace: Optional[object] = None
        super().__init__(f"Guardrail '{guardrail_name}' blocked: {reason}")

    def __reduce__(self) -> "tuple[type, tuple[str, str, List[Tuple[str, str]]]]":
        # Default exception pickling replays the rendered message into the
        # two-arg __init__ and fails; reconstruct from the original args
        # (same fix family as selectools.exceptions, PR #100).
        return (self.__class__, (self.guardrail_name, self.reason, self.prior_trips))


__stability__ = "stable"

__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailError",
    "GuardrailResult",
]
