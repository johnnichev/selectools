"""
GuardrailsPipeline — ordered list of input and output guardrails.

The pipeline runs each guardrail in sequence.  If a guardrail rewrites
content, subsequent guardrails see the rewritten version.  If a guardrail
blocks, processing stops immediately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from selectools.stability import stable

from .base import Guardrail, GuardrailAction, GuardrailError, GuardrailResult

logger = logging.getLogger("selectools.guardrails")


@stable
@dataclass
class GuardrailsPipeline:
    """Ordered pipeline of input, output, and tool-args guardrails.

    Args:
        input: Guardrails evaluated on user messages **before** the LLM call.
        output: Guardrails evaluated on the LLM response **after** it returns.
        tool_args: Guardrails evaluated on tool-call **arguments** before the
            tool executes (added in v1.1, opt-in). Output guardrails only see
            the model's free-text content; anything the model carries via a
            native tool call bypasses them entirely. Each guardrail in this
            list receives the JSON-serialized arguments dict as its content;
            ``rewrite`` results are parsed back into the arguments, ``block``
            raises :class:`GuardrailError` before the tool runs.
        tool_results: Guardrails evaluated on a tool's **return value** after
            execution, before the result re-enters the model context (added
            in v1.2, opt-in). The other half of the tool-boundary surface:
            ``tool_args`` gates what goes INTO a tool, this gates what comes
            OUT (external API responses, retrieved chunks, oversized blobs).
            Tool results are plain strings, so guardrails receive them as-is;
            ``rewrite`` replaces the result. ``block`` contains the content
            (a blocked marker replaces it so history/memory stay coherent)
            and the agent loop raises :class:`GuardrailError` once the tool
            batch has been processed. Also applied to tool-result cache
            hits.
    """

    input: List[Guardrail] = field(default_factory=list)
    output: List[Guardrail] = field(default_factory=list)
    tool_args: List[Guardrail] = field(default_factory=list)
    tool_results: List[Guardrail] = field(default_factory=list)

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

    def check_tool_args(self, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Run all *tool_args* guardrails against a tool call's arguments.

        The arguments dict is JSON-serialized and fed through the chain like
        any other content, so text-oriented guardrails (PII, injection,
        length, topic) work unchanged.

        Returns:
            ``(parameters, triggered_names)`` — the (possibly rewritten)
            arguments dict, and a comma-joined string of triggered guardrail
            names (``None`` if nothing triggered).

        Raises:
            GuardrailError: If any guardrail with ``action=block`` fails, or
                if a ``rewrite`` guardrail produces something that is no
                longer a valid JSON object.
        """
        if not self.tool_args:
            return parameters, None
        serialized = json.dumps(parameters, ensure_ascii=False, default=str)
        result = self._run_chain(self.tool_args, serialized)
        if result.content == serialized:
            return parameters, result.guardrail_name
        return self._parse_rewritten_args(result), result.guardrail_name

    async def acheck_tool_args(
        self, parameters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Async version of :meth:`check_tool_args`."""
        if not self.tool_args:
            return parameters, None
        serialized = json.dumps(parameters, ensure_ascii=False, default=str)
        result = await self._arun_chain(self.tool_args, serialized)
        if result.content == serialized:
            return parameters, result.guardrail_name
        return self._parse_rewritten_args(result), result.guardrail_name

    def check_tool_result(self, result: str) -> Tuple[str, Optional[str]]:
        """Run all *tool_results* guardrails against a tool's return value.

        Returns:
            ``(result, triggered_names)`` — the (possibly rewritten) result
            string, and a comma-joined string of triggered guardrail names
            (``None`` if nothing triggered).

        Raises:
            GuardrailError: If any guardrail with ``action=block`` fails.
        """
        chain_result = self._run_chain(self.tool_results, result)
        return chain_result.content, chain_result.guardrail_name

    async def acheck_tool_result(self, result: str) -> Tuple[str, Optional[str]]:
        """Async version of :meth:`check_tool_result`."""
        chain_result = await self._arun_chain(self.tool_results, result)
        return chain_result.content, chain_result.guardrail_name

    @staticmethod
    def _parse_rewritten_args(result: GuardrailResult) -> Dict[str, Any]:
        """Parse a rewritten tool-args payload back into an arguments dict."""
        try:
            rewritten = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise GuardrailError(
                guardrail_name=result.guardrail_name or "tool_args",
                reason=f"rewrite produced content that is not valid JSON: {exc}",
            ) from exc
        if not isinstance(rewritten, dict):
            raise GuardrailError(
                guardrail_name=result.guardrail_name or "tool_args",
                reason=(
                    "rewrite produced valid JSON but not an object; tool arguments "
                    f"must stay a JSON object, got {type(rewritten).__name__}"
                ),
            )
        return rewritten

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


__stability__ = "stable"

__all__ = ["GuardrailsPipeline"]
