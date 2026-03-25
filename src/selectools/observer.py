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
from typing import Any, Callable, Dict, List, Optional

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
    # Session events
    # ------------------------------------------------------------------

    def on_session_load(
        self,
        run_id: str,
        session_id: str,
        message_count: int,
    ) -> None:
        """Called when a session is loaded from a session store."""

    def on_session_save(
        self,
        run_id: str,
        session_id: str,
        message_count: int,
    ) -> None:
        """Called when a session is saved to a session store."""

    # ------------------------------------------------------------------
    # Memory summarization events
    # ------------------------------------------------------------------

    def on_memory_summarize(
        self,
        run_id: str,
        summary: str,
    ) -> None:
        """Called when a conversation summary is generated after trim."""

    # ------------------------------------------------------------------
    # Entity extraction events
    # ------------------------------------------------------------------

    def on_entity_extraction(
        self,
        run_id: str,
        entities_extracted: int,
    ) -> None:
        """Called after entities are extracted from conversation messages."""

    # ------------------------------------------------------------------
    # Knowledge graph extraction events
    # ------------------------------------------------------------------

    def on_kg_extraction(
        self,
        run_id: str,
        triples_extracted: int,
    ) -> None:
        """Called after relationship triples are extracted from conversation messages."""

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

    # ------------------------------------------------------------------
    # Budget events
    # ------------------------------------------------------------------

    def on_budget_exceeded(
        self,
        run_id: str,
        reason: str,
        tokens_used: int,
        cost_used: float,
    ) -> None:
        """Called when the agent stops because a token or cost budget was exceeded."""

    # ------------------------------------------------------------------
    # Cancellation events
    # ------------------------------------------------------------------

    def on_cancelled(
        self,
        run_id: str,
        iteration: int,
        reason: str,
    ) -> None:
        """Called when the agent run is cancelled via a CancellationToken."""

    # ------------------------------------------------------------------
    # Model switching events
    # ------------------------------------------------------------------

    def on_model_switch(
        self,
        run_id: str,
        iteration: int,
        old_model: str,
        new_model: str,
    ) -> None:
        """Called when the model_selector switches the model for an iteration."""

    # ------------------------------------------------------------------
    # Prompt compression events
    # ------------------------------------------------------------------

    def on_prompt_compressed(
        self,
        run_id: str,
        before_tokens: int,
        after_tokens: int,
        messages_compressed: int,
    ) -> None:
        """Called when prompt compression reduces the context before an LLM call.

        ``before_tokens`` and ``after_tokens`` are estimates from
        :func:`~selectools.token_estimation.estimate_run_tokens`.
        ``messages_compressed`` is the count of messages replaced by the summary.
        """

    # ------------------------------------------------------------------
    # Eval events
    # ------------------------------------------------------------------

    def on_eval_start(
        self,
        suite_name: str,
        total_cases: int,
        model: str,
    ) -> None:
        """Called when an eval suite starts running."""

    def on_eval_case_end(
        self,
        suite_name: str,
        case_name: str,
        verdict: str,
        latency_ms: float,
        failures: int,
    ) -> None:
        """Called after each eval case completes."""

    def on_eval_end(
        self,
        suite_name: str,
        accuracy: float,
        total_cases: int,
        pass_count: int,
        fail_count: int,
        total_cost: float,
        duration_ms: float,
    ) -> None:
        """Called when an eval suite finishes."""


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

    def on_session_load(self, run_id: str, session_id: str, message_count: int) -> None:
        self._emit("session_load", run_id, session_id=session_id, message_count=message_count)

    def on_session_save(self, run_id: str, session_id: str, message_count: int) -> None:
        self._emit("session_save", run_id, session_id=session_id, message_count=message_count)

    def on_memory_summarize(self, run_id: str, summary: str) -> None:
        self._emit("memory_summarize", run_id, summary_length=len(summary))

    def on_entity_extraction(self, run_id: str, entities_extracted: int) -> None:
        self._emit("entity_extraction", run_id, entities_extracted=entities_extracted)

    def on_kg_extraction(self, run_id: str, triples_extracted: int) -> None:
        self._emit("kg_extraction", run_id, triples_extracted=triples_extracted)

    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._emit("error", run_id, error=str(error), error_type=type(error).__name__)

    def on_budget_exceeded(
        self, run_id: str, reason: str, tokens_used: int, cost_used: float
    ) -> None:
        self._emit(
            "budget_exceeded",
            run_id,
            reason=reason,
            tokens_used=tokens_used,
            cost_used=round(cost_used, 6),
        )

    def on_cancelled(self, run_id: str, iteration: int, reason: str) -> None:
        self._emit("cancelled", run_id, iteration=iteration, reason=reason)

    def on_model_switch(self, run_id: str, iteration: int, old_model: str, new_model: str) -> None:
        self._emit(
            "model_switch",
            run_id,
            iteration=iteration,
            old_model=old_model,
            new_model=new_model,
        )

    def on_prompt_compressed(
        self, run_id: str, before_tokens: int, after_tokens: int, messages_compressed: int
    ) -> None:
        self._emit(
            "prompt_compressed",
            run_id,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            messages_compressed=messages_compressed,
            tokens_saved=before_tokens - after_tokens,
        )

    def on_eval_start(self, suite_name: str, total_cases: int, model: str) -> None:
        self._emit("eval_start", "", suite_name=suite_name, total_cases=total_cases, model=model)

    def on_eval_case_end(
        self, suite_name: str, case_name: str, verdict: str, latency_ms: float, failures: int
    ) -> None:
        self._emit(
            "eval_case_end",
            "",
            suite_name=suite_name,
            case_name=case_name,
            verdict=verdict,
            latency_ms=round(latency_ms, 1),
            failures=failures,
        )

    def on_eval_end(
        self,
        suite_name: str,
        accuracy: float,
        total_cases: int,
        pass_count: int,
        fail_count: int,
        total_cost: float,
        duration_ms: float,
    ) -> None:
        self._emit(
            "eval_end",
            "",
            suite_name=suite_name,
            accuracy=round(accuracy, 4),
            total_cases=total_cases,
            pass_count=pass_count,
            fail_count=fail_count,
            total_cost=round(total_cost, 6),
            duration_ms=round(duration_ms, 1),
        )


class AsyncAgentObserver(AgentObserver):
    """Base class for async agent lifecycle observers.

    Override any subset of the ``a_on_*`` methods to receive notifications
    asynchronously.  Each method mirrors the sync ``AgentObserver`` counterpart.

    The ``blocking`` attribute controls execution behavior:

    - ``blocking=False`` (default): ``asyncio.ensure_future()`` — runs concurrently,
      never slows the agent loop.  Good for webhooks, logging, audit trails.
    - ``blocking=True``: Inline ``await`` — must complete before the loop continues.
      Good for DB writes, rate limiting, result enrichment.

    In sync ``run()``, blocking async observers are called via ``asyncio.run()``.
    Non-blocking async observers are skipped in sync context.
    """

    blocking: bool = False

    # ------------------------------------------------------------------
    # Run-level events
    # ------------------------------------------------------------------

    async def a_on_run_start(
        self,
        run_id: str,
        messages: List[Message],
        system_prompt: str,
    ) -> None:
        """Async counterpart of :meth:`on_run_start`."""

    async def a_on_run_end(
        self,
        run_id: str,
        result: "AgentResult",
    ) -> None:
        """Async counterpart of :meth:`on_run_end`."""

    # ------------------------------------------------------------------
    # LLM-level events
    # ------------------------------------------------------------------

    async def a_on_llm_start(
        self,
        run_id: str,
        messages: List[Message],
        model: str,
        system_prompt: str,
    ) -> None:
        """Async counterpart of :meth:`on_llm_start`."""

    async def a_on_llm_end(
        self,
        run_id: str,
        response: str,
        usage: Optional["UsageStats"],
    ) -> None:
        """Async counterpart of :meth:`on_llm_end`."""

    async def a_on_cache_hit(
        self,
        run_id: str,
        model: str,
        response: str,
    ) -> None:
        """Async counterpart of :meth:`on_cache_hit`."""

    async def a_on_usage(
        self,
        run_id: str,
        usage: "UsageStats",
    ) -> None:
        """Async counterpart of :meth:`on_usage`."""

    # ------------------------------------------------------------------
    # Tool-level events
    # ------------------------------------------------------------------

    async def a_on_tool_start(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Async counterpart of :meth:`on_tool_start`."""

    async def a_on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        """Async counterpart of :meth:`on_tool_end`."""

    async def a_on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Async counterpart of :meth:`on_tool_error`."""

    async def a_on_tool_chunk(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        chunk: str,
    ) -> None:
        """Async counterpart of :meth:`on_tool_chunk`."""

    # ------------------------------------------------------------------
    # Policy events
    # ------------------------------------------------------------------

    async def a_on_policy_decision(
        self,
        run_id: str,
        tool_name: str,
        decision: str,
        reason: str,
        tool_args: Dict[str, Any],
    ) -> None:
        """Async counterpart of :meth:`on_policy_decision`."""

    # ------------------------------------------------------------------
    # Structured output events
    # ------------------------------------------------------------------

    async def a_on_structured_validate(
        self,
        run_id: str,
        success: bool,
        attempt: int,
        error: Optional[str] = None,
    ) -> None:
        """Async counterpart of :meth:`on_structured_validate`."""

    # ------------------------------------------------------------------
    # Iteration events
    # ------------------------------------------------------------------

    async def a_on_iteration_start(
        self,
        run_id: str,
        iteration: int,
        messages: List[Message],
    ) -> None:
        """Async counterpart of :meth:`on_iteration_start`."""

    async def a_on_iteration_end(
        self,
        run_id: str,
        iteration: int,
        response: str,
    ) -> None:
        """Async counterpart of :meth:`on_iteration_end`."""

    # ------------------------------------------------------------------
    # Batch events
    # ------------------------------------------------------------------

    async def a_on_batch_start(
        self,
        batch_id: str,
        prompts_count: int,
    ) -> None:
        """Async counterpart of :meth:`on_batch_start`."""

    async def a_on_batch_end(
        self,
        batch_id: str,
        results_count: int,
        errors_count: int,
        total_duration_ms: float,
    ) -> None:
        """Async counterpart of :meth:`on_batch_end`."""

    # ------------------------------------------------------------------
    # Provider fallback events
    # ------------------------------------------------------------------

    async def a_on_provider_fallback(
        self,
        run_id: str,
        failed_provider: str,
        next_provider: str,
        error: Exception,
    ) -> None:
        """Async counterpart of :meth:`on_provider_fallback`."""

    # ------------------------------------------------------------------
    # LLM retry events
    # ------------------------------------------------------------------

    async def a_on_llm_retry(
        self,
        run_id: str,
        attempt: int,
        max_retries: int,
        error: Exception,
        backoff_seconds: float,
    ) -> None:
        """Async counterpart of :meth:`on_llm_retry`."""

    # ------------------------------------------------------------------
    # Memory events
    # ------------------------------------------------------------------

    async def a_on_memory_trim(
        self,
        run_id: str,
        messages_removed: int,
        messages_remaining: int,
        reason: str,
    ) -> None:
        """Async counterpart of :meth:`on_memory_trim`."""

    # ------------------------------------------------------------------
    # Session events
    # ------------------------------------------------------------------

    async def a_on_session_load(
        self,
        run_id: str,
        session_id: str,
        message_count: int,
    ) -> None:
        """Async counterpart of :meth:`on_session_load`."""

    async def a_on_session_save(
        self,
        run_id: str,
        session_id: str,
        message_count: int,
    ) -> None:
        """Async counterpart of :meth:`on_session_save`."""

    # ------------------------------------------------------------------
    # Memory summarization events
    # ------------------------------------------------------------------

    async def a_on_memory_summarize(
        self,
        run_id: str,
        summary: str,
    ) -> None:
        """Async counterpart of :meth:`on_memory_summarize`."""

    # ------------------------------------------------------------------
    # Entity extraction events
    # ------------------------------------------------------------------

    async def a_on_entity_extraction(
        self,
        run_id: str,
        entities_extracted: int,
    ) -> None:
        """Async counterpart of :meth:`on_entity_extraction`."""

    # ------------------------------------------------------------------
    # Knowledge graph extraction events
    # ------------------------------------------------------------------

    async def a_on_kg_extraction(
        self,
        run_id: str,
        triples_extracted: int,
    ) -> None:
        """Async counterpart of :meth:`on_kg_extraction`."""

    # ------------------------------------------------------------------
    # Error events
    # ------------------------------------------------------------------

    async def a_on_error(
        self,
        run_id: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> None:
        """Async counterpart of :meth:`on_error`."""

    # ------------------------------------------------------------------
    # Budget events
    # ------------------------------------------------------------------

    async def a_on_budget_exceeded(
        self,
        run_id: str,
        reason: str,
        tokens_used: int,
        cost_used: float,
    ) -> None:
        """Async counterpart of :meth:`on_budget_exceeded`."""

    # ------------------------------------------------------------------
    # Cancellation events
    # ------------------------------------------------------------------

    async def a_on_cancelled(
        self,
        run_id: str,
        iteration: int,
        reason: str,
    ) -> None:
        """Async counterpart of :meth:`on_cancelled`."""

    # ------------------------------------------------------------------
    # Model switching events
    # ------------------------------------------------------------------

    async def a_on_model_switch(
        self,
        run_id: str,
        iteration: int,
        old_model: str,
        new_model: str,
    ) -> None:
        """Async counterpart of :meth:`on_model_switch`."""

    # ------------------------------------------------------------------
    # Prompt compression events
    # ------------------------------------------------------------------

    async def a_on_prompt_compressed(
        self,
        run_id: str,
        before_tokens: int,
        after_tokens: int,
        messages_compressed: int,
    ) -> None:
        """Async counterpart of :meth:`on_prompt_compressed`."""


# ======================================================================
# Convenience observers
# ======================================================================


class SimpleStepObserver(AgentObserver):
    """Observer that routes all lifecycle events to a single callback.

    Instead of subclassing :class:`AgentObserver` and overriding many methods,
    pass a single function that receives every event::

        def on_event(event: str, run_id: str, **data):
            print(f"[{event}] run={run_id} {data}")

        config = AgentConfig(observers=[SimpleStepObserver(on_event)])

    The callback signature is ``(event_name: str, run_id: str, **kwargs)``.
    Event names match the method names without the ``on_`` prefix
    (e.g. ``"run_start"``, ``"tool_end"``, ``"usage"``).
    """

    def __init__(self, callback: Callable[..., None]) -> None:
        self._cb = callback

    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self._cb("run_start", run_id, message_count=len(messages), system_prompt=system_prompt)

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        self._cb("run_end", run_id, result=result)

    def on_llm_start(
        self, run_id: str, messages: List[Message], model: str, system_prompt: str
    ) -> None:
        self._cb("llm_start", run_id, model=model, message_count=len(messages))

    def on_llm_end(self, run_id: str, response: str, usage: Optional[UsageStats]) -> None:
        self._cb("llm_end", run_id, response=response, usage=usage)

    def on_cache_hit(self, run_id: str, model: str, response: str) -> None:
        self._cb("cache_hit", run_id, model=model, response=response)

    def on_usage(self, run_id: str, usage: UsageStats) -> None:
        self._cb("usage", run_id, usage=usage)

    def on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self._cb("tool_start", run_id, call_id=call_id, tool_name=tool_name, tool_args=tool_args)

    def on_tool_end(
        self, run_id: str, call_id: str, tool_name: str, result: str, duration_ms: float
    ) -> None:
        self._cb(
            "tool_end",
            run_id,
            call_id=call_id,
            tool_name=tool_name,
            result=result,
            duration_ms=duration_ms,
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
        self._cb(
            "tool_error",
            run_id,
            call_id=call_id,
            tool_name=tool_name,
            error=error,
            tool_args=tool_args,
            duration_ms=duration_ms,
        )

    def on_tool_chunk(self, run_id: str, call_id: str, tool_name: str, chunk: str) -> None:
        self._cb("tool_chunk", run_id, call_id=call_id, tool_name=tool_name, chunk=chunk)

    def on_policy_decision(
        self, run_id: str, tool_name: str, decision: str, reason: str, tool_args: Dict[str, Any]
    ) -> None:
        self._cb(
            "policy_decision",
            run_id,
            tool_name=tool_name,
            decision=decision,
            reason=reason,
            tool_args=tool_args,
        )

    def on_structured_validate(
        self, run_id: str, success: bool, attempt: int, error: Optional[str] = None
    ) -> None:
        self._cb("structured_validate", run_id, success=success, attempt=attempt, error=error)

    def on_iteration_start(self, run_id: str, iteration: int, messages: List[Message]) -> None:
        self._cb("iteration_start", run_id, iteration=iteration, message_count=len(messages))

    def on_iteration_end(self, run_id: str, iteration: int, response: str) -> None:
        self._cb("iteration_end", run_id, iteration=iteration, response=response)

    def on_batch_start(self, batch_id: str, prompts_count: int) -> None:
        self._cb("batch_start", batch_id, prompts_count=prompts_count)

    def on_batch_end(
        self, batch_id: str, results_count: int, errors_count: int, total_duration_ms: float
    ) -> None:
        self._cb(
            "batch_end",
            batch_id,
            results_count=results_count,
            errors_count=errors_count,
            total_duration_ms=total_duration_ms,
        )

    def on_provider_fallback(
        self, run_id: str, failed_provider: str, next_provider: str, error: Exception
    ) -> None:
        self._cb(
            "provider_fallback",
            run_id,
            failed_provider=failed_provider,
            next_provider=next_provider,
            error=error,
        )

    def on_llm_retry(
        self,
        run_id: str,
        attempt: int,
        max_retries: int,
        error: Exception,
        backoff_seconds: float,
    ) -> None:
        self._cb(
            "llm_retry",
            run_id,
            attempt=attempt,
            max_retries=max_retries,
            error=error,
            backoff_seconds=backoff_seconds,
        )

    def on_memory_trim(
        self, run_id: str, messages_removed: int, messages_remaining: int, reason: str
    ) -> None:
        self._cb(
            "memory_trim",
            run_id,
            messages_removed=messages_removed,
            messages_remaining=messages_remaining,
            reason=reason,
        )

    def on_session_load(self, run_id: str, session_id: str, message_count: int) -> None:
        self._cb("session_load", run_id, session_id=session_id, message_count=message_count)

    def on_session_save(self, run_id: str, session_id: str, message_count: int) -> None:
        self._cb("session_save", run_id, session_id=session_id, message_count=message_count)

    def on_memory_summarize(self, run_id: str, summary: str) -> None:
        self._cb("memory_summarize", run_id, summary=summary)

    def on_entity_extraction(self, run_id: str, entities_extracted: int) -> None:
        self._cb("entity_extraction", run_id, entities_extracted=entities_extracted)

    def on_kg_extraction(self, run_id: str, triples_extracted: int) -> None:
        self._cb("kg_extraction", run_id, triples_extracted=triples_extracted)

    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._cb("error", run_id, error=error, context=context)

    def on_budget_exceeded(
        self, run_id: str, reason: str, tokens_used: int, cost_used: float
    ) -> None:
        self._cb(
            "budget_exceeded",
            run_id,
            reason=reason,
            tokens_used=tokens_used,
            cost_used=cost_used,
        )

    def on_cancelled(self, run_id: str, iteration: int, reason: str) -> None:
        self._cb("cancelled", run_id, iteration=iteration, reason=reason)

    def on_model_switch(self, run_id: str, iteration: int, old_model: str, new_model: str) -> None:
        self._cb(
            "model_switch",
            run_id,
            iteration=iteration,
            old_model=old_model,
            new_model=new_model,
        )

    def on_prompt_compressed(
        self, run_id: str, before_tokens: int, after_tokens: int, messages_compressed: int
    ) -> None:
        self._cb(
            "prompt_compressed",
            run_id,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            messages_compressed=messages_compressed,
        )

    def on_eval_start(self, suite_name: str, total_cases: int, model: str) -> None:
        self._cb("eval_start", "", suite_name=suite_name, total_cases=total_cases, model=model)

    def on_eval_case_end(
        self, suite_name: str, case_name: str, verdict: str, latency_ms: float, failures: int
    ) -> None:
        self._cb(
            "eval_case_end",
            "",
            suite_name=suite_name,
            case_name=case_name,
            verdict=verdict,
            latency_ms=latency_ms,
            failures=failures,
        )

    def on_eval_end(
        self,
        suite_name: str,
        accuracy: float,
        total_cases: int,
        pass_count: int,
        fail_count: int,
        total_cost: float,
        duration_ms: float,
    ) -> None:
        self._cb(
            "eval_end",
            "",
            suite_name=suite_name,
            accuracy=accuracy,
            total_cases=total_cases,
            pass_count=pass_count,
            fail_count=fail_count,
            total_cost=total_cost,
            duration_ms=duration_ms,
        )


# ======================================================================
# Hooks compatibility adapter (internal)
# ======================================================================


class _HooksAdapter(AgentObserver):
    """Internal adapter that wraps a legacy ``hooks`` dict as an :class:`AgentObserver`.

    This allows the hooks dict to be routed through the single observer
    notification path, eliminating the need for parallel ``_call_hook()``
    calls throughout the agent loop.

    Not part of the public API — do **not** add to ``__all__``.
    """

    def __init__(self, hooks: Dict[str, Any]) -> None:
        self._hooks = hooks

    def _call(self, hook_name: str, *args: Any) -> None:
        fn = self._hooks.get(hook_name)
        if fn:
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 # nosec B110
                pass

    # -- Run-level events --------------------------------------------------

    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self._call("on_agent_start", messages)

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        self._call("on_agent_end", result.message, result.usage)

    # -- LLM-level events --------------------------------------------------

    def on_llm_start(
        self, run_id: str, messages: List[Message], model: str, system_prompt: str
    ) -> None:
        self._call("on_llm_start", messages, model)

    def on_llm_end(self, run_id: str, response: str, usage: Optional[UsageStats]) -> None:
        self._call("on_llm_end", response, usage)

    # -- Tool-level events -------------------------------------------------

    def on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self._call("on_tool_start", tool_name, tool_args)

    def on_tool_end(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
    ) -> None:
        self._call("on_tool_end", tool_name, result, duration_ms / 1000)

    def on_tool_error(
        self,
        run_id: str,
        call_id: str,
        tool_name: str,
        error: Exception,
        tool_args: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        self._call("on_tool_error", tool_name, error, tool_args)

    def on_tool_chunk(self, run_id: str, call_id: str, tool_name: str, chunk: str) -> None:
        self._call("on_tool_chunk", tool_name, chunk)

    # -- Iteration-level events --------------------------------------------

    def on_iteration_start(self, run_id: str, iteration: int, messages: List[Message]) -> None:
        self._call("on_iteration_start", iteration, messages)

    def on_iteration_end(self, run_id: str, iteration: int, response: str) -> None:
        self._call("on_iteration_end", iteration, response)

    # -- Error events ------------------------------------------------------

    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._call("on_error", error, context)


__all__ = ["AgentObserver", "AsyncAgentObserver", "LoggingObserver", "SimpleStepObserver"]
