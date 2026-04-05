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
        self._traces: Dict[str, Any] = {}
        self._generations: Dict[str, Any] = {}
        self._llm_counter: int = 0

    # ── Run lifecycle ─────────────────────────────────────────────────

    def on_run_start(
        self,
        run_id: str,
        messages: Any,
        system_prompt: str,
    ) -> None:
        """Create a Langfuse trace for the agent run."""
        trace = self._langfuse.trace(
            id=run_id,
            name="agent.run",
            metadata={"system_prompt_length": len(system_prompt) if system_prompt else 0},
        )
        self._traces[run_id] = trace

    def on_run_end(self, run_id: str, result: Any) -> None:
        """Update the trace with final results and flush.

        Also cleans up any orphaned generations/spans (LLM/tool) that were
        started but never ended due to abnormal exits.
        """
        # Clean up orphaned child generations/spans first
        prefix = f"{run_id}:"
        orphaned_keys = [k for k in self._generations if k.startswith(prefix)]
        for key in orphaned_keys:
            orphan = self._generations.pop(key, None)
            if orphan is not None:
                try:
                    orphan.update(
                        output="ERROR: Orphaned — run ended before span closed",
                        level="ERROR",
                    )
                except Exception:
                    logger.debug("Failed to update orphaned Langfuse span %s", key)

        trace = self._traces.pop(run_id, None)
        if trace is None:
            return
        metadata: Dict[str, Any] = {}
        if hasattr(result, "usage") and result.usage:
            metadata["total_tokens"] = getattr(result.usage, "total_tokens", 0)
            metadata["total_cost_usd"] = getattr(result.usage, "total_cost_usd", 0.0)
        if hasattr(result, "iterations"):
            metadata["iterations"] = result.iterations
        trace.update(
            output=getattr(result, "content", str(result)),
            metadata=metadata,
        )
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
        """Create a Langfuse generation for an LLM call."""
        self._llm_counter += 1
        trace = self._traces.get(run_id)
        if trace is None:
            return
        gen = trace.generation(
            name="llm.call",
            model=model or "unknown",
            input=str(messages)[:2000] if messages else "",
        )
        self._generations[f"{run_id}:llm:{self._llm_counter}"] = gen

    def on_llm_end(
        self,
        run_id: str,
        content: str,
        usage: Any,
    ) -> None:
        """Update the most recent generation for this run."""
        prefix = f"{run_id}:llm:"
        matching = [k for k in self._generations if k.startswith(prefix)]
        if not matching:
            return
        key = max(matching, key=lambda k: int(k.rsplit(":", 1)[1]))
        gen = self._generations.pop(key, None)
        if gen is None:
            return
        update_kwargs: Dict[str, Any] = {"output": (content or "")[:2000]}
        if usage:
            update_kwargs["usage"] = {
                "input": getattr(usage, "prompt_tokens", 0) or 0,
                "output": getattr(usage, "completion_tokens", 0) or 0,
                "total": getattr(usage, "total_tokens", 0) or 0,
            }
        gen.update(**update_kwargs)

    # ── Tool execution ────────────────────────────────────────────────

    def on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Create a Langfuse span for tool execution."""
        trace = self._traces.get(run_id)
        if trace is None:
            return
        span = trace.span(
            name=f"tool.{tool_name}",
            input=str(tool_args)[:1000] if tool_args else "",
        )
        self._generations[f"{run_id}:tool:{call_id}"] = span

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        """Update the tool span with results."""
        key = f"{run_id}:tool:{call_id}"
        span = self._generations.pop(key, None)
        if span is None:
            return
        span.update(
            output=(result or "")[:2000],
            metadata={"duration_ms": duration_ms},
        )

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
        span = self._generations.pop(key, None)
        if span is None:
            return
        span.update(
            output=f"ERROR: {error}",
            level="ERROR",
            metadata={"duration_ms": duration_ms},
        )

    # ── Cleanup ───────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Flush remaining traces. Call on application shutdown."""
        try:
            self._langfuse.flush()
        except Exception:
            logger.warning("Failed to flush Langfuse on shutdown", exc_info=True)


__all__ = ["LangfuseObserver"]
