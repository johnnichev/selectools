"""
GuardrailsPipeline — ordered list of input and output guardrails.

The pipeline runs each guardrail in sequence.  If a guardrail rewrites
content, subsequent guardrails see the rewritten version.  If a guardrail
blocks, processing stops immediately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from selectools.stability import stable

from .base import Guardrail, GuardrailAction, GuardrailError, GuardrailResult

logger = logging.getLogger("selectools.guardrails")


@stable
@dataclass
class GuardrailsPipeline:
    """Ordered pipeline of input and output guardrails.

    Args:
        input: Guardrails evaluated on user messages **before** the LLM call.
        output: Guardrails evaluated on the LLM response **after** it returns.
    """

    input: List[Guardrail] = field(default_factory=list)
    output: List[Guardrail] = field(default_factory=list)

    def check_input(self, content: str) -> GuardrailResult:
        """Run all *input* guardrails against *content*.

        Returns the final :class:`GuardrailResult` (which may contain
        rewritten content from earlier guardrails in the chain).

        Raises:
            GuardrailError: If any guardrail with ``action=block`` fails.
        """
        return self._run_chain(self.input, content)

    def check_output(self, content: str) -> GuardrailResult:
        """Run all *output* guardrails against *content*.

        Returns the final :class:`GuardrailResult`.

        Raises:
            GuardrailError: If any guardrail with ``action=block`` fails.
        """
        return self._run_chain(self.output, content)

    async def acheck_input(self, content: str) -> GuardrailResult:
        """Async version of :meth:`check_input`."""
        return await self._arun_chain(self.input, content)

    async def acheck_output(self, content: str) -> GuardrailResult:
        """Async version of :meth:`check_output`."""
        return await self._arun_chain(self.output, content)

    @staticmethod
    def _run_chain(guardrails: List[Guardrail], content: str) -> GuardrailResult:
        current = content
        triggered_names: List[str] = []
        for g in guardrails:
            result = g.check(current)
            if not result.passed:
                triggered_names.append(g.name)
                if g.action == GuardrailAction.BLOCK:
                    raise GuardrailError(
                        guardrail_name=g.name,
                        reason=result.reason or "Check failed",
                    )
                if g.action == GuardrailAction.WARN:
                    logger.warning(
                        "Guardrail '%s' warning: %s",
                        g.name,
                        result.reason or "Check failed",
                    )
                    continue
                if g.action == GuardrailAction.REWRITE:
                    current = result.content
                    continue
            else:
                current = result.content
        guardrail_name = ", ".join(triggered_names) if triggered_names else None
        return GuardrailResult(passed=True, content=current, guardrail_name=guardrail_name)

    @staticmethod
    async def _arun_chain(guardrails: List[Guardrail], content: str) -> GuardrailResult:
        """Async version of ``_run_chain`` — calls ``acheck()`` on each guardrail."""
        current = content
        triggered_names: List[str] = []
        for g in guardrails:
            result = await g.acheck(current)
            if not result.passed:
                triggered_names.append(g.name)
                if g.action == GuardrailAction.BLOCK:
                    raise GuardrailError(
                        guardrail_name=g.name,
                        reason=result.reason or "Check failed",
                    )
                if g.action == GuardrailAction.WARN:
                    logger.warning(
                        "Guardrail '%s' warning: %s",
                        g.name,
                        result.reason or "Check failed",
                    )
                    continue
                if g.action == GuardrailAction.REWRITE:
                    current = result.content
                    continue
            else:
                current = result.content
        guardrail_name = ", ".join(triggered_names) if triggered_names else None
        return GuardrailResult(passed=True, content=current, guardrail_name=guardrail_name)


__all__ = ["GuardrailsPipeline"]
