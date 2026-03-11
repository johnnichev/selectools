"""
Observer protocol for agent lifecycle events.

Provides a class-based alternative to the hooks dict for structured
observability integrations (Langfuse, OpenTelemetry, Datadog, etc.).

Every callback receives a ``run_id`` so observers can correlate events
across concurrent requests without closures or thread-local state.
Tool callbacks additionally receive a ``call_id`` so parallel tool
executions can be matched between start and end.

All methods have no-op defaults — subclass and override only the events
you care about.

Example::

    class MyObserver(AgentObserver):
        def on_llm_start(self, run_id, messages, model, system_prompt):
            print(f"[{run_id}] LLM call to {model}")

        def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
            print(f"[{run_id}] {tool_name} finished in {duration_ms:.1f}ms")

    agent = Agent(
        tools=[...],
        provider=provider,
        config=AgentConfig(observers=[MyObserver()]),
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .types import AgentResult, Message
from .usage import UsageStats


class AgentObserver:
    """Base class for agent lifecycle observers.

    Override any subset of these methods to receive notifications.
    All methods are intentionally synchronous — implementations that need
    async I/O should queue work internally (Langfuse, OTel SDKs already
    do this).
    """

    # ------------------------------------------------------------------
    # Run-level events
    # ------------------------------------------------------------------

    def on_run_start(
        self,
        run_id: str,
        messages: List[Message],
        system_prompt: str,
    ) -> None:
        """Called at the beginning of run/arun/astream."""

    def on_run_end(
        self,
        run_id: str,
        result: AgentResult,
    ) -> None:
        """Called when the agent produces its final result."""

    # ------------------------------------------------------------------
    # LLM-level events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        run_id: str,
        messages: List[Message],
        model: str,
        system_prompt: str,
    ) -> None:
        """Called before each LLM provider call."""

    def on_llm_end(
        self,
        run_id: str,
        response: str,
        usage: Optional[UsageStats],
    ) -> None:
        """Called after each LLM provider call completes."""

    def on_cache_hit(
        self,
        run_id: str,
        model: str,
        response: str,
    ) -> None:
        """Called when an LLM response is served from cache.

        This fires *in addition to* the paired ``on_llm_start``/``on_llm_end``
        so integrators can distinguish cached from live LLM calls.
        """

    def on_usage(
        self,
        run_id: str,
        usage: UsageStats,
    ) -> None:
        """Called after each LLM call with the per-call usage stats.

        Fires for every LLM invocation (including cache hits and structured
        retries), enabling real-time token/cost tracking without waiting for
        ``on_run_end``.
        """

    # ------------------------------------------------------------------
    # Tool-level events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Called before a tool is executed.

        ``call_id`` is unique per tool invocation and matches the
        corresponding ``on_tool_end`` or ``on_tool_error`` call, even
        when tools run in parallel.
        """

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        """Called after a tool executes successfully.

        ``duration_ms`` is the wall-clock execution time in **milliseconds**,
        consistent with ``TraceStep.duration_ms``.
        """

    def on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Called when a tool execution raises an exception.

        ``duration_ms`` is the wall-clock execution time in **milliseconds**,
        consistent with ``TraceStep.duration_ms``.
        """

    # ------------------------------------------------------------------
    # Policy events
    # ------------------------------------------------------------------

    def on_policy_decision(
        self,
        run_id: str,
        tool_name: str,
        decision: str,
        reason: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Called after the tool policy evaluates a tool call.

        ``decision`` is one of ``"allow"``, ``"deny"``, or ``"review"``.
        For ``"review"`` decisions, a subsequent approval/denial may follow
        via ``confirm_action``.
        """

    # ------------------------------------------------------------------
    # Structured output events
    # ------------------------------------------------------------------

    def on_structured_validate(
        self,
        run_id: str,
        success: bool,
        attempt: int,
        error: Optional[str] = None,
    ) -> None:
        """Called after structured output validation (pass or fail).

        ``attempt`` is 1-based (first attempt = 1).  On failure, ``error``
        contains the validation error message and the agent will retry if
        iterations remain.
        """

    # ------------------------------------------------------------------
    # Iteration events
    # ------------------------------------------------------------------

    def on_iteration_start(
        self,
        run_id: str,
        iteration: int,
        messages: List[Message],
    ) -> None:
        """Called at the start of each agent loop iteration.

        ``iteration`` is 1-based.  Observers can use this to group the
        LLM/tool events that follow into logical iterations.
        """

    def on_iteration_end(
        self,
        run_id: str,
        iteration: int,
        response: str,
    ) -> None:
        """Called at the end of each agent loop iteration."""

    # ------------------------------------------------------------------
    # Streaming tool events
    # ------------------------------------------------------------------

    def on_tool_chunk(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        chunk: str,
    ) -> None:
        """Called for each chunk emitted by a streaming tool.

        This is high-frequency — only override if you need real-time
        progress tracking for streaming tool output.
        """

    # ------------------------------------------------------------------
    # Batch events
    # ------------------------------------------------------------------

    def on_batch_start(
        self,
        batch_id: str,
        prompts_count: int,
    ) -> None:
        """Called before ``batch()``/``abatch()`` begins processing.

        ``batch_id`` is a UUID that links all child ``run_id`` values
        produced by this batch.
        """

    def on_batch_end(
        self,
        batch_id: str,
        results_count: int,
        errors_count: int,
        total_duration_ms: float,
    ) -> None:
        """Called after all batch items have completed."""

    # ------------------------------------------------------------------
    # Provider fallback events
    # ------------------------------------------------------------------

    def on_provider_fallback(
        self,
        run_id: str,
        failed_provider: str,
        next_provider: str,
        error: Exception,
    ) -> None:
        """Called when ``FallbackProvider`` switches to the next provider.

        Surfaces the provider-level fallback event so observers can track
        provider reliability without inspecting ``FallbackProvider`` internals.
        """

    # ------------------------------------------------------------------
    # LLM retry events
    # ------------------------------------------------------------------

    def on_llm_retry(
        self,
        run_id: str,
        attempt: int,
        max_retries: int,
        error: Exception,
        backoff_seconds: float,
    ) -> None:
        """Called when an LLM call fails and is about to be retried.

        ``attempt`` is the just-failed attempt number (1-based).
        ``backoff_seconds`` is the total sleep time before the next attempt.
        """

    # ------------------------------------------------------------------
    # Memory events
    # ------------------------------------------------------------------

    def on_memory_trim(
        self,
        run_id: str,
        messages_removed: int,
        messages_remaining: int,
        reason: str,
    ) -> None:
        """Called when conversation memory trims messages.

        ``reason`` is one of ``"enforce_limits"`` (sliding window /
        max-tokens trim) or ``"tool_pair_boundary"`` (orphaned tool
        messages dropped after trim).
        """

    # ------------------------------------------------------------------
    # Error events
    # ------------------------------------------------------------------

    def on_error(
        self,
        run_id: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> None:
        """Called when the agent encounters an unrecoverable error."""


# ======================================================================
# Built-in observers
# ======================================================================


class LoggingObserver(AgentObserver):
    """Observer that writes structured JSON events to Python's logging module.

    Every event is emitted as a JSON object on a single log line, making it
    easy to ingest into ELK, CloudWatch, Datadog Logs, or any log aggregator.

    Usage::

        import logging
        logging.basicConfig(level=logging.INFO)

        agent = Agent(
            tools=[...],
            provider=provider,
            config=AgentConfig(observers=[LoggingObserver()]),
        )

    Args:
        logger_name: Name of the Python logger. Default: ``"selectools.observer"``.
        level: Log level for events. Default: ``logging.INFO``.
    """

    def __init__(
        self,
        logger_name: str = "selectools.observer",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def _emit(self, event: str, run_id: str, **data: Any) -> None:
        payload: Dict[str, Any] = {
            "event": event,
            "run_id": run_id,
            "ts": time.time(),
        }
        payload.update({k: v for k, v in data.items() if v is not None})
        self._logger.log(self._level, json.dumps(payload, default=str))

    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self._emit("run_start", run_id, message_count=len(messages))

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        self._emit(
            "run_end",
            run_id,
            iterations=result.iterations,
            trace_steps=len(result.trace) if result.trace else 0,
            content_length=len(result.content) if result.content else 0,
        )

    def on_llm_start(
        self,
        run_id: str,
        messages: List[Message],
        model: str,
        system_prompt: str,
    ) -> None:
        self._emit("llm_start", run_id, model=model, message_count=len(messages))

    def on_llm_end(
        self,
        run_id: str,
        response: str,
        usage: Optional[UsageStats],
    ) -> None:
        data: Dict[str, Any] = {"response_length": len(response) if response else 0}
        if usage:
            data.update(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=usage.cost_usd,
                model=usage.model,
            )
        self._emit("llm_end", run_id, **data)

    def on_cache_hit(self, run_id: str, model: str, response: str) -> None:
        self._emit("cache_hit", run_id, model=model, response_length=len(response))

    def on_usage(self, run_id: str, usage: UsageStats) -> None:
        self._emit(
            "usage",
            run_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            model=usage.model,
        )

    def on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        self._emit("tool_start", run_id, call_id=call_id, tool_name=tool_name)

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        self._emit(
            "tool_end",
            run_id,
            call_id=call_id,
            tool_name=tool_name,
            duration_ms=round(duration_ms, 2),
            result_length=len(result),
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
        self._emit(
            "tool_error",
            run_id,
            call_id=call_id,
            tool_name=tool_name,
            error=str(error),
            duration_ms=round(duration_ms, 2),
        )

    def on_policy_decision(
        self,
        run_id: str,
        tool_name: str,
        decision: str,
        reason: str,
        tool_args: Dict[str, Any],
    ) -> None:
        self._emit(
            "policy_decision",
            run_id,
            tool_name=tool_name,
            decision=decision,
            reason=reason,
        )

    def on_structured_validate(
        self,
        run_id: str,
        success: bool,
        attempt: int,
        error: Optional[str] = None,
    ) -> None:
        self._emit(
            "structured_validate",
            run_id,
            success=success,
            attempt=attempt,
            error=error,
        )

    def on_iteration_start(
        self,
        run_id: str,
        iteration: int,
        messages: List[Message],
    ) -> None:
        self._emit("iteration_start", run_id, iteration=iteration)

    def on_iteration_end(
        self,
        run_id: str,
        iteration: int,
        response: str,
    ) -> None:
        self._emit(
            "iteration_end",
            run_id,
            iteration=iteration,
            response_length=len(response) if response else 0,
        )

    def on_batch_start(self, batch_id: str, prompts_count: int) -> None:
        self._emit("batch_start", batch_id, prompts_count=prompts_count)

    def on_batch_end(
        self,
        batch_id: str,
        results_count: int,
        errors_count: int,
        total_duration_ms: float,
    ) -> None:
        self._emit(
            "batch_end",
            batch_id,
            results_count=results_count,
            errors_count=errors_count,
            total_duration_ms=round(total_duration_ms, 2),
        )

    def on_provider_fallback(
        self,
        run_id: str,
        failed_provider: str,
        next_provider: str,
        error: Exception,
    ) -> None:
        self._emit(
            "provider_fallback",
            run_id,
            failed_provider=failed_provider,
            next_provider=next_provider,
            error=str(error),
        )

    def on_llm_retry(
        self,
        run_id: str,
        attempt: int,
        max_retries: int,
        error: Exception,
        backoff_seconds: float,
    ) -> None:
        self._emit(
            "llm_retry",
            run_id,
            attempt=attempt,
            max_retries=max_retries,
            error=str(error),
            backoff_seconds=round(backoff_seconds, 2),
        )

    def on_memory_trim(
        self,
        run_id: str,
        messages_removed: int,
        messages_remaining: int,
        reason: str,
    ) -> None:
        self._emit(
            "memory_trim",
            run_id,
            messages_removed=messages_removed,
            messages_remaining=messages_remaining,
            reason=reason,
        )

    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._emit("error", run_id, error=str(error), error_type=type(error).__name__)


__all__ = ["AgentObserver", "LoggingObserver"]
