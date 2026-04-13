"""Langfuse observer for selectools agents.

Sends selectools traces to Langfuse for observability, cost tracking,
and debugging. Requires ``langfuse`` (optional).

Usage::

    from selectools.observe.langfuse import LangfuseObserver
    agent = Agent(tools=[...], config=AgentConfig(observers=[LangfuseObserver()]))
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from ..observer import AgentObserver
from ..stability import beta

logger = logging.getLogger(__name__)


@beta
class LangfuseObserver(AgentObserver):
    """Send selectools agent traces to Langfuse.

    Args:
        public_key: Langfuse public key (or ``LANGFUSE_PUBLIC_KEY`` env var).
        secret_key: Langfuse secret key (or ``LANGFUSE_SECRET_KEY`` env var).
        host: Langfuse host for self-hosted (or ``LANGFUSE_HOST`` env var).
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        try:
            from langfuse import Langfuse
        except ImportError as exc:
            raise ImportError(
                "langfuse is required for LangfuseObserver. "
                "Install it with: pip install langfuse"
            ) from exc
        self._langfuse = Langfuse(
            public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
            host=host or os.getenv("LANGFUSE_HOST"),
        )
        # One root span per agent run. In Langfuse 3.x each root span is
        # also a trace — update_trace on the root span sets trace-level
        # fields (name, output, metadata, tags).
        self._traces: Dict[str, Any] = {}
        self._generations: Dict[str, Any] = {}
        self._llm_counter: int = 0
        self._lock = threading.Lock()

    # ── Run lifecycle ─────────────────────────────────────────────────

    def on_run_start(
        self,
        run_id: str,
        messages: Any,
        system_prompt: str,
    ) -> None:
        """Create a Langfuse root span for the agent run.

        In Langfuse 3.x the root-level ``Langfuse.trace()`` helper was
        removed. A top-level ``start_span`` creates the trace implicitly
        and returns a ``LangfuseSpan`` from which child spans and
        generations can be started.
        """
        root = self._langfuse.start_span(
            name="agent.run",
            input=str(messages)[:2000] if messages else "",
            metadata={"system_prompt_length": len(system_prompt) if system_prompt else 0},
        )
        with self._lock:
            self._traces[run_id] = root

    def on_run_end(self, run_id: str, result: Any) -> None:
        """Update the root span + trace and flush.

        Also cleans up any orphaned child spans (LLM/tool) that were
        started but never ended due to abnormal exits.
        """
        # Clean up orphaned child spans first
        prefix = f"{run_id}:"
        with self._lock:
            orphaned_keys = [k for k in self._generations if k.startswith(prefix)]
            orphans = []
            for key in orphaned_keys:
                orphan = self._generations.pop(key, None)
                if orphan is not None:
                    orphans.append((key, orphan))
            root = self._traces.pop(run_id, None)

        for key, orphan in orphans:
            try:
                orphan.update(
                    output="ERROR: Orphaned — run ended before span closed",
                    level="ERROR",
                )
                orphan.end()
            except Exception:
                logger.debug("Failed to close orphaned Langfuse span %s", key)

        if root is None:
            return

        output = getattr(result, "content", str(result))
        metadata: Dict[str, Any] = {}
        if hasattr(result, "usage") and result.usage:
            metadata["total_tokens"] = getattr(result.usage, "total_tokens", 0)
            metadata["total_cost_usd"] = getattr(result.usage, "total_cost_usd", 0.0)
        if hasattr(result, "iterations"):
            metadata["iterations"] = result.iterations

        try:
            # Update trace-level fields (name, output, metadata).
            root.update_trace(output=output, metadata=metadata)
            # Also set the root span's own output and metadata, then end it.
            root.update(output=output, metadata=metadata)
            root.end()
        except Exception:
            logger.warning("Failed to finalize Langfuse root span", exc_info=True)

        try:
            self._langfuse.flush()
        except Exception:
            logger.warning("Failed to flush Langfuse traces", exc_info=True)

    # ── LLM calls ─────────────────────────────────────────────────────

    def on_llm_start(
        self,
        run_id: str,
        messages: Any,
        model: str,
        system_prompt: str,
    ) -> None:
        """Create a Langfuse generation for an LLM call.

        In Langfuse 3.x, child spans / generations are started **from
        the parent span** via ``root.start_generation(...)``. This
        automatically attaches them to the same trace.
        """
        with self._lock:
            self._llm_counter += 1
            counter = self._llm_counter
            root = self._traces.get(run_id)
        if root is None:
            return
        gen = root.start_generation(
            name="llm.call",
            model=model or "unknown",
            input=str(messages)[:2000] if messages else "",
        )
        with self._lock:
            self._generations[f"{run_id}:llm:{counter}"] = gen

    def on_llm_end(
        self,
        run_id: str,
        content: str,
        usage: Any,
    ) -> None:
        """Update the most recent generation for this run, then end it."""
        prefix = f"{run_id}:llm:"
        with self._lock:
            matching = [k for k in self._generations if k.startswith(prefix)]
            if not matching:
                return
            key = max(matching, key=lambda k: int(k.rsplit(":", 1)[1]))
            gen = self._generations.pop(key, None)
        if gen is None:
            return

        # Langfuse 3.x generation update takes ``usage_details`` (new name,
        # same shape as the 2.x ``usage`` dict) and ``cost_details``.
        update_kwargs: Dict[str, Any] = {"output": (content or "")[:2000]}
        if usage:
            update_kwargs["usage_details"] = {
                "input": getattr(usage, "prompt_tokens", 0) or 0,
                "output": getattr(usage, "completion_tokens", 0) or 0,
                "total": getattr(usage, "total_tokens", 0) or 0,
            }
            cost_usd = getattr(usage, "cost_usd", None) or getattr(usage, "total_cost_usd", None)
            if cost_usd:
                update_kwargs["cost_details"] = {"total": float(cost_usd)}
        try:
            gen.update(**update_kwargs)
            gen.end()
        except Exception:
            logger.debug("Failed to update/end Langfuse generation", exc_info=True)

    # ── Tool execution ────────────────────────────────────────────────

    def on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Create a Langfuse child span for tool execution."""
        with self._lock:
            root = self._traces.get(run_id)
        if root is None:
            return
        span = root.start_span(
            name=f"tool.{tool_name}",
            input=str(tool_args)[:1000] if tool_args else "",
        )
        with self._lock:
            self._generations[f"{run_id}:tool:{call_id}"] = span

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        """Update the tool span with results and end it."""
        key = f"{run_id}:tool:{call_id}"
        with self._lock:
            span = self._generations.pop(key, None)
        if span is None:
            return
        try:
            span.update(
                output=(result or "")[:2000],
                metadata={"duration_ms": duration_ms},
            )
            span.end()
        except Exception:
            logger.debug("Failed to update/end Langfuse tool span", exc_info=True)

    def on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Record an error on the tool span and end it."""
        key = f"{run_id}:tool:{call_id}"
        with self._lock:
            span = self._generations.pop(key, None)
        if span is None:
            return
        try:
            span.update(
                output=f"ERROR: {error}",
                level="ERROR",
                metadata={"duration_ms": duration_ms},
            )
            span.end()
        except Exception:
            logger.debug("Failed to record error on Langfuse tool span", exc_info=True)

    # ── Cleanup ───────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Flush remaining traces. Call on application shutdown."""
        try:
            self._langfuse.flush()
        except Exception:
            logger.warning("Failed to flush Langfuse on shutdown", exc_info=True)


__all__ = ["LangfuseObserver"]
