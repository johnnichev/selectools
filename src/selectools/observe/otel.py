"""OpenTelemetry observer for selectools agents.

Maps selectools observer events to OpenTelemetry spans following the
GenAI semantic conventions. Requires ``opentelemetry-api`` (optional).

Usage::

    from selectools.observe.otel import OTelObserver
    agent = Agent(tools=[...], config=AgentConfig(observers=[OTelObserver()]))
"""

from __future__ import annotations

import logging
import threading
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
        self._llm_counter: int = 0
        self._tool_counter: int = 0
        self._lock = threading.Lock()

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
        with self._lock:
            self._spans[run_id] = span

    def on_run_end(self, run_id: str, result: Any) -> None:
        """End the root span with usage metadata.

        Also cleans up any orphaned child spans (LLM/tool) that were started
        but never ended due to abnormal exits.
        """
        # Clean up orphaned child spans first
        prefix = f"{run_id}:"
        with self._lock:
            orphaned_keys = [k for k in self._spans if k.startswith(prefix)]
            orphans = []
            for key in orphaned_keys:
                orphan = self._spans.pop(key, None)
                self._llm_starts.pop(key, None)
                if orphan is not None:
                    orphans.append(orphan)
            span = self._spans.pop(run_id, None)

        for orphan in orphans:
            try:
                status = self._trace_mod.StatusCode.ERROR
                orphan.set_status(status, "Span orphaned — run ended before span closed")
            except Exception:
                # StatusCode may not be available in all OTel versions;
                # setting an attribute is a safe fallback.
                orphan.set_attribute("error", True)
                orphan.set_attribute(
                    "selectools.error", "Span orphaned — run ended before span closed"
                )
            orphan.end()

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
        with self._lock:
            self._llm_counter += 1
            counter = self._llm_counter
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
        key = f"{run_id}:llm:{counter}"
        with self._lock:
            self._spans[key] = span
            self._llm_starts[key] = time.time()

    def on_llm_end(
        self,
        run_id: str,
        content: str,
        usage: Any,
    ) -> None:
        """End the most recent LLM call span for this run."""
        prefix = f"{run_id}:llm:"
        with self._lock:
            # Find the highest-numbered LLM span for this run_id
            matching = [k for k in self._spans if k.startswith(prefix)]
            if not matching:
                return
            key = max(matching, key=lambda k: int(k.rsplit(":", 1)[1]))
            span = self._spans.pop(key, None)
            start = self._llm_starts.pop(key, None)
        if span is None:
            return
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            span = self._spans.pop(key, None)
        if span is None:
            return
        span.set_attribute("error", True)
        span.set_attribute("selectools.tool.error", str(error))
        span.set_attribute("selectools.tool.duration_ms", duration_ms)
        span.end()


__all__ = ["OTelObserver"]
