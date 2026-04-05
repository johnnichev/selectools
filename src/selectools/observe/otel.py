"""OpenTelemetry observer for selectools agents.

Maps selectools observer events to OpenTelemetry spans following the
GenAI semantic conventions. Requires ``opentelemetry-api`` (optional).

Usage::

    from selectools.observe.otel import OTelObserver
    agent = Agent(tools=[...], config=AgentConfig(observers=[OTelObserver()]))
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..observer import AgentObserver
from ..stability import beta

logger = logging.getLogger(__name__)


@beta
class OTelObserver(AgentObserver):
    """Send selectools agent traces to OpenTelemetry.

    Creates spans for agent runs, LLM calls, and tool executions following
    the `OpenTelemetry GenAI semantic conventions
    <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.

    Args:
        tracer_name: Name for the OTel tracer (default: ``"selectools"``).
    """

    def __init__(self, tracer_name: str = "selectools") -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise ImportError(
                "opentelemetry-api is required for OTelObserver. "
                "Install it with: pip install opentelemetry-api"
            ) from exc
        self._tracer = trace.get_tracer(tracer_name)
        self._trace_mod = trace
        self._spans: Dict[str, Any] = {}
        self._llm_starts: Dict[str, float] = {}

    # ── Run lifecycle ─────────────────────────────────────────────────

    def on_run_start(
        self,
        run_id: str,
        messages: Any,
        system_prompt: str,
    ) -> None:
        """Start a root span for the agent run."""
        span = self._tracer.start_span(
            "agent.run",
            attributes={
                "gen_ai.system": "selectools",
                "gen_ai.operation.name": "agent.run",
                "selectools.run_id": run_id,
            },
        )
        self._spans[run_id] = span

    def on_run_end(self, run_id: str, result: Any) -> None:
        """End the root span with usage metadata."""
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
            span.set_attribute(
                "gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0
            )
            span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", 0) or 0)
            if hasattr(usage, "total_cost_usd"):
                span.set_attribute("selectools.cost_usd", usage.total_cost_usd or 0.0)
        if hasattr(result, "iterations"):
            span.set_attribute("selectools.iterations", result.iterations or 0)
        span.end()

    # ── LLM calls ─────────────────────────────────────────────────────

    def on_llm_start(
        self,
        run_id: str,
        messages: Any,
        model: str,
        system_prompt: str,
    ) -> None:
        """Start a child span for an LLM call."""
        parent = self._spans.get(run_id)
        ctx = self._trace_mod.set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            "gen_ai.chat",
            context=ctx,
            attributes={
                "gen_ai.request.model": model or "",
                "gen_ai.system": "selectools",
            },
        )
        key = f"{run_id}:llm"
        self._spans[key] = span
        self._llm_starts[key] = time.time()

    def on_llm_end(
        self,
        run_id: str,
        content: str,
        usage: Any,
    ) -> None:
        """End the LLM call span."""
        key = f"{run_id}:llm"
        span = self._spans.pop(key, None)
        if span is None:
            return
        start = self._llm_starts.pop(key, None)
        if start:
            span.set_attribute("selectools.duration_ms", (time.time() - start) * 1000)
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
            span.set_attribute(
                "gen_ai.usage.output_tokens", getattr(usage, "completion_tokens", 0) or 0
            )
        span.end()

    # ── Tool execution ────────────────────────────────────────────────

    def on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Start a child span for tool execution."""
        parent = self._spans.get(run_id)
        ctx = self._trace_mod.set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            "tool.execute",
            context=ctx,
            attributes={
                "selectools.tool.name": tool_name,
                "selectools.tool.call_id": call_id or "",
            },
        )
        self._spans[f"{run_id}:tool:{call_id}"] = span

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        """End the tool execution span."""
        key = f"{run_id}:tool:{call_id}"
        span = self._spans.pop(key, None)
        if span is None:
            return
        span.set_attribute("selectools.tool.duration_ms", duration_ms)
        span.set_attribute("selectools.tool.result_length", len(result) if result else 0)
        span.end()

    def on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Record an error on the tool span."""
        key = f"{run_id}:tool:{call_id}"
        span = self._spans.pop(key, None)
        if span is None:
            return
        span.set_attribute("error", True)
        span.set_attribute("selectools.tool.error", str(error))
        span.set_attribute("selectools.tool.duration_ms", duration_ms)
        span.end()


__all__ = ["OTelObserver"]
