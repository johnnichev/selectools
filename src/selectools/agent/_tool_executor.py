"""Mixin providing tool execution methods for the Agent class."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import inspect
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level singleton for tool timeout execution — avoids spawning a new
# thread pool on every tool call (see pitfall #20).
_tool_timeout_executor: Optional[ThreadPoolExecutor] = None
_tool_timeout_executor_lock = threading.Lock()

# Separate executor for the outer parallel dispatch layer.  Using the same
# pool for both the outer _run_one submissions and the inner timeout
# submissions causes a thread-pool deadlock when the number of concurrent tool
# calls is >= (max_workers / 2 + 1): outer workers block on inner futures that
# can never start because all pool slots are taken.
_parallel_dispatch_executor: Optional[ThreadPoolExecutor] = None
_parallel_dispatch_executor_lock = threading.Lock()


def _get_tool_timeout_executor() -> ThreadPoolExecutor:
    """Return the shared ThreadPoolExecutor for tool timeout enforcement."""
    global _tool_timeout_executor
    if _tool_timeout_executor is None:
        with _tool_timeout_executor_lock:
            if _tool_timeout_executor is None:
                _tool_timeout_executor = ThreadPoolExecutor(
                    max_workers=16, thread_name_prefix="selectools_tool_timeout"
                )
    return _tool_timeout_executor


def _get_parallel_dispatch_executor() -> ThreadPoolExecutor:
    """Return the shared ThreadPoolExecutor for parallel tool dispatch."""
    global _parallel_dispatch_executor
    if _parallel_dispatch_executor is None:
        with _parallel_dispatch_executor_lock:
            if _parallel_dispatch_executor is None:
                _parallel_dispatch_executor = ThreadPoolExecutor(
                    max_workers=32, thread_name_prefix="selectools_parallel_dispatch"
                )
    return _parallel_dispatch_executor


from .._async_utils import run_in_executor_copyctx
from ..coherence import CoherenceResult, acheck_coherence, check_coherence
from ..policy import ApprovalRequest, PolicyDecision, PolicyResult, ToolPolicy
from ..security import screen_output as screen_tool_output
from ..trace import StepType, TraceStep
from ..types import Message, Role

if TYPE_CHECKING:
    from ..tools import Tool
    from ..trace import AgentTrace
    from ..types import ToolCall
    from .core import _RunContext


# Marker prefix applied by _finish_compression to successful summaries.
# Used to distinguish real compression output from the truncation fallback
# when deciding whether to cache the compressed form alongside the raw result.
_COMPRESSED_MARKER_PREFIX = "[compressed from "

_COMPRESS_SYSTEM_PROMPT = (
    "You compress verbose tool outputs for an AI agent. Produce a faithful, dense "
    "summary of the tool output. Preserve exactly, verbatim: all numbers, "
    "identifiers, URLs, file paths, error messages, and key data values. Never "
    "invent or add information that is not in the input. Output only the summary."
)


class _ToolExecutorMixin:
    """Mixin that provides tool execution methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, _history, usage,
    analytics, _tools_by_name, etc.) which are expected to be provided by the
    Agent class that inherits from this mixin.
    """

    # ------------------------------------------------------------------
    # Tool result compression (ROADMAP P2)
    # ------------------------------------------------------------------

    def _tool_compression_candidate(self, tool: Optional["Tool"], result: str) -> bool:
        """Cheap gate deciding whether tool-result compression should be attempted.

        Returns False (with zero extra work) when compression is disabled, the
        result is within the threshold, or the tool is terminal (its result
        becomes the agent's final answer and must stay verbatim).
        """
        tcfg = self.config.tool
        if tcfg is None or not getattr(tcfg, "compress_results", False):
            return False
        if tool is None or getattr(tool, "terminal", False):
            return False
        return bool(len(result) > tcfg.compress_threshold)

    def _stop_condition_hit(self, tool_name: str, result: str) -> bool:
        """True if config.stop_condition fires for this result (terminal path)."""
        return bool(self.config.stop_condition and self.config.stop_condition(tool_name, result))

    def _compression_fallback(self, result: str) -> str:
        """Truncation-with-marker fallback used when summarization fails."""
        threshold = self.config.tool.compress_threshold
        return f"[truncated from {len(result)} chars; compression failed] {result[:threshold]}"

    def _warn_compression_fallback(
        self, tool_name: str, exc: Exception, run_ctx: Optional["_RunContext"]
    ) -> None:
        """Log a loud warning when compression degrades to truncation.

        Logged at most once per run (via a flag on the run context) so a
        persistently broken compress_provider/compress_model does not spam
        one warning per oversized tool result.
        """
        if run_ctx is not None:
            if run_ctx.compression_fallback_warned:
                return
            run_ctx.compression_fallback_warned = True
        logger.warning(
            "Tool result compression for '%s' fell back to truncation "
            "(%s: %s). Check compress_provider/compress_model. Further "
            "compression fallbacks in this run will not be logged.",
            tool_name,
            type(exc).__name__,
            exc,
            exc_info=exc,
        )

    def _build_compression_request(self, tool_name: str, result: str) -> Message:
        """Build the one-shot user message for the compression call."""
        return Message(
            role=Role.USER,
            content=(
                f"Summarize the following output from tool '{tool_name}'. Preserve "
                "all numbers, IDs, URLs, file paths, and error text verbatim:\n\n" + result
            ),
        )

    def _finish_compression(
        self,
        completion: Any,
        original: str,
        tool_name: str,
        trace: Optional["AgentTrace"],
    ) -> str:
        """Shared tail of the sync/async compression paths.

        Unpacks the provider response, records usage, applies the fidelity
        marker, and never returns something longer than the original.

        A summary that hit the max-token cap is treated as a failure (it is
        cut off mid-sentence and may have silently dropped data): detected via
        the response's finish/stop reason when the provider exposes one, or
        via ``completion_tokens`` reaching the requested budget. Raising here
        routes the caller to the truncation fallback.
        """
        msg = completion[0] if isinstance(completion, tuple) else completion
        usage = completion[1] if isinstance(completion, tuple) else None
        if usage is not None:
            self.usage.add_usage(usage)
        finish = getattr(msg, "finish_reason", None) or getattr(msg, "stop_reason", None)
        if isinstance(finish, str) and finish.lower() in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }:
            raise ValueError(f"compression summary hit the max-token cap (finish={finish})")
        if usage is not None and usage.completion_tokens >= self._compression_max_tokens():
            raise ValueError(
                "compression summary hit the max-token cap "
                f"(completion_tokens={usage.completion_tokens})"
            )
        summary = (msg.content or "").strip()
        if not summary:
            raise ValueError("compression returned empty summary")
        compressed = f"{_COMPRESSED_MARKER_PREFIX}{len(original)} chars] {summary}"
        if len(compressed) >= len(original):
            return original  # compression made it worse — keep the raw result
        if trace is not None:
            trace.add(
                TraceStep(
                    type=StepType.PROMPT_COMPRESSED,
                    tool_name=tool_name,
                    summary=(f"Tool result compressed: {len(original)}→{len(compressed)} chars"),
                )
            )
        return compressed

    def _compression_max_tokens(self) -> int:
        """Token budget for the summary: comfortably under the char threshold."""
        threshold = self.config.tool.compress_threshold
        return int(max(128, min(1000, threshold // 4)))

    def _compress_tool_result(
        self,
        tool_name: str,
        result: str,
        trace: Optional["AgentTrace"] = None,
        run_ctx: Optional["_RunContext"] = None,
    ) -> str:
        """Summarize an oversized tool result via a one-shot LLM call (sync).

        Falls back to truncation-with-marker on ANY failure — never crashes
        the tool loop. The fallback logs a warning (once per run).
        """
        tcfg = self.config.tool
        try:
            provider = tcfg.compress_provider or self.provider
            model = tcfg.compress_model or self._effective_model
            completion = provider.complete(
                model=model,
                system_prompt=_COMPRESS_SYSTEM_PROMPT,
                messages=[self._build_compression_request(tool_name, result)],
                max_tokens=self._compression_max_tokens(),
                timeout=self.config.request_timeout,
            )
            return self._finish_compression(completion, result, tool_name, trace)
        except Exception as exc:
            self._warn_compression_fallback(tool_name, exc, run_ctx)
            return self._compression_fallback(result)

    async def _acompress_tool_result(
        self,
        tool_name: str,
        result: str,
        trace: Optional["AgentTrace"] = None,
        run_ctx: Optional["_RunContext"] = None,
    ) -> str:
        """Async counterpart of :meth:`_compress_tool_result`."""
        tcfg = self.config.tool
        try:
            provider = tcfg.compress_provider or self.provider
            model = tcfg.compress_model or self._effective_model
            completion = await provider.acomplete(
                model=model,
                system_prompt=_COMPRESS_SYSTEM_PROMPT,
                messages=[self._build_compression_request(tool_name, result)],
                max_tokens=self._compression_max_tokens(),
                timeout=self.config.request_timeout,
            )
            return self._finish_compression(completion, result, tool_name, trace)
        except Exception as exc:
            self._warn_compression_fallback(tool_name, exc, run_ctx)
            return self._compression_fallback(result)

    def _screen_tool_result(self, tool_name: str, result: str) -> str:
        """Screen a tool result for prompt injection if the tool or config requires it."""
        tool = self._tools_by_name.get(tool_name)
        should_screen = self.config.screen_tool_output or (
            tool is not None and getattr(tool, "screen_output", False)
        )
        if not should_screen:
            return result
        screening = screen_tool_output(
            result,
            extra_patterns=self.config.output_screening_patterns,
        )
        return screening.content

    @staticmethod
    def _build_tool_cache_key(tool_name: str, params: Dict[str, Any]) -> str:
        """Build a deterministic cache key for a tool call."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"tool_result:{tool_name}:{params_hash}"

    def _check_tool_cache(
        self, tool: "Tool", params: Dict[str, Any]
    ) -> Optional[Tuple[str, Optional[str]]]:
        """Return ``(raw_result, compressed_or_None)`` on cache hit, None on miss.

        Hit paths must run stop_condition/terminal checks on the RAW result
        but append the stored compressed text (when present) to history, so
        repeated cache hits don't re-flood the context with the full blob.
        Backward compatible with older entry formats: a plain string or a
        ``(raw, None)`` tuple both yield ``compressed=None`` (behaves as if
        compression never happened).
        """
        if not self.config.cache or not getattr(tool, "cacheable", False):
            return None
        key = self._build_tool_cache_key(tool.name, params)
        cached = self.config.cache.get(key)
        if cached is None:
            return None
        if isinstance(cached, str):
            return (cached, None)  # legacy plain-string entry
        raw = str(cached[0])
        compressed: Optional[str] = None
        if len(cached) > 1 and cached[1] is not None:
            compressed = str(cached[1])
        return (raw, compressed)

    def _store_tool_cache(
        self,
        tool: "Tool",
        params: Dict[str, Any],
        result: str,
        compressed: Optional[str] = None,
    ) -> None:
        """Store a tool result (and its compressed form, when any) in the cache."""
        if not self.config.cache or not getattr(tool, "cacheable", False):
            return
        key = self._build_tool_cache_key(tool.name, params)
        ttl = getattr(tool, "cache_ttl", 300)
        self.config.cache.set(key, (result, compressed), ttl=ttl)

    def _store_compressed_in_cache(
        self, tool: Optional["Tool"], params: Dict[str, Any], raw: str, message_content: str
    ) -> None:
        """Re-store a cache entry with its compressed text after compression.

        Only successful summaries (marked ``[compressed from ...]``) are
        cached; the truncation fallback reflects a transient summarizer
        failure and must not be frozen into the cache.
        """
        if tool is None or message_content == raw:
            return
        if message_content.startswith(_COMPRESSED_MARKER_PREFIX):
            self._store_tool_cache(tool, params, raw, compressed=message_content)

    def _check_coherence(
        self,
        user_message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Sync coherence check.  Returns error string or None."""
        if not self.config.coherence_check:
            return None
        provider = self.config.coherence_provider or self.provider
        model = self.config.coherence_model or self._effective_model
        result = check_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
            fail_closed=getattr(self.config, "coherence_fail_closed", False),
        )
        if result.usage:
            self.usage.add_usage(result.usage)
        if not result.coherent:
            return (
                f"Coherence check failed for tool '{tool_name}': "
                f"{result.explanation or 'Tool call does not match user intent'}"
            )
        return None

    async def _acheck_coherence(
        self,
        user_message: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Async coherence check.  Returns error string or None."""
        if not self.config.coherence_check:
            return None
        provider = self.config.coherence_provider or self.provider
        model = self.config.coherence_model or self._effective_model
        result = await acheck_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
            fail_closed=getattr(self.config, "coherence_fail_closed", False),
        )
        if result.usage:
            self.usage.add_usage(result.usage)
        if not result.coherent:
            return (
                f"Coherence check failed for tool '{tool_name}': "
                f"{result.explanation or 'Tool call does not match user intent'}"
            )
        return None

    # ------------------------------------------------------------------
    # Agent-level human-in-the-loop (ROADMAP P2)
    # ------------------------------------------------------------------

    def _config_requires_approval(self, tool_name: str) -> bool:
        """True if ToolConfig.require_approval gates this tool ("*" gates all)."""
        tcfg = self.config.tool
        gated = getattr(tcfg, "require_approval", None)
        if not gated:
            return False
        if isinstance(gated, str):
            return gated == "*" or gated == tool_name
        return "*" in gated or tool_name in gated

    @staticmethod
    def _build_approval_request(
        tool_name: str, tool_args: Dict[str, Any], reason: str
    ) -> ApprovalRequest:
        """Build the structured request passed to the approval handler.

        ``tool_args`` is deep-copied so a handler that mutates the request
        (e.g. redacting a field before forwarding it to a pager) can never
        change the parameters the tool actually executes with.
        """
        try:
            args_copy = copy.deepcopy(tool_args)
        except Exception:  # unpicklable values — fall back to a shallow copy
            args_copy = dict(tool_args)
        args_repr = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        preview = f"{tool_name}({args_repr})"
        if len(preview) > 200:
            preview = preview[:197] + "..."
        return ApprovalRequest(
            tool_name=tool_name, tool_args=args_copy, reason=reason, preview=preview
        )

    @staticmethod
    def _is_async_approval_handler(handler: Callable[..., Any]) -> bool:
        """True for coroutine functions AND instances with ``async def __call__``.

        ``inspect.iscoroutinefunction(instance)`` is False for a class
        instance whose ``__call__`` is async — without this check the
        un-awaited coroutine object would be truth-tested (always truthy)
        and a deny-all handler would silently APPROVE every call.
        """
        # B004 suppressed: we are not testing callability here — we need the
        # bound __call__ attribute itself to detect `async def __call__`
        # instances.
        call_attr = getattr(handler, "__call__", handler)  # noqa: B004
        return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(call_attr)

    @staticmethod
    def _approval_memo_key(tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Digest key for per-run denial memoization."""
        args_str = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}:{hashlib.sha256(args_str.encode()).hexdigest()[:16]}"

    @staticmethod
    def _approval_verdict(tool_name: str, approved: Any, reason: str) -> Optional[str]:
        """Map a handler return value to an error string (None = approved).

        Fails CLOSED on anything that is not a plain bool: coroutines,
        generators, mocks, and other accidentally-truthy objects must never
        approve a gated call.
        """
        if not isinstance(approved, bool):
            return (
                f"Tool '{tool_name}' denied: approval handler returned "
                f"{type(approved).__name__} instead of bool — failing closed. "
                "Return True to approve or False to deny."
            )
        if not approved:
            return f"Tool '{tool_name}' denied by approval handler: {reason}"
        return None

    def _run_approval_handler(
        self,
        handler: Callable[..., Any],
        tool_name: str,
        tool_args: Dict[str, Any],
        reason: str,
    ) -> Optional[str]:
        """Invoke the approval handler from the sync path. Returns error string or None.

        Async handlers (coroutine functions or instances with ``async def
        __call__``) are executed via ``asyncio.run`` on the shared worker
        pool so they work from plain ``run()`` (even when the caller itself
        is inside an event loop).

        NOTE: while waiting for the human, the handler occupies one slot of
        the shared 16-worker ``selectools_tool_timeout`` pool (also used for
        tool-timeout enforcement and ``confirm_action``). Many concurrent
        blocking approvals can exhaust the pool and stall other runs — keep
        handlers prompt and ``approval_timeout`` deliberate.
        """
        request = self._build_approval_request(tool_name, tool_args, reason)
        try:
            # Reuse the shared module-level executor — never create a new
            # ThreadPoolExecutor per call (pitfall #20).
            executor = _get_tool_timeout_executor()
            ctx_copy = contextvars.copy_context()
            if self._is_async_approval_handler(handler):
                future = executor.submit(ctx_copy.run, asyncio.run, handler(request))
            else:

                def _invoke() -> Any:
                    result = handler(request)
                    # Belt-and-suspenders: a handler classified as sync can
                    # still return an awaitable — run it on this worker
                    # instead of truth-testing the awaitable (fail-open).
                    if inspect.isawaitable(result):

                        async def _await_it() -> Any:
                            return await result

                        return asyncio.run(_await_it())
                    return result

                future = executor.submit(ctx_copy.run, _invoke)
            try:
                approved = future.result(timeout=self.config.approval_timeout)
            except FuturesTimeoutError:
                # Cancel so a still-QUEUED handler can never fire after the
                # call was already denied (and never pages a human for a
                # decision that no longer matters).
                future.cancel()
                return (
                    f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
                )
        except Exception as exc:
            return f"Tool '{tool_name}' approval failed: {exc}"
        return self._approval_verdict(tool_name, approved, reason)

    async def _arun_approval_handler(
        self,
        handler: Callable[..., Any],
        tool_name: str,
        tool_args: Dict[str, Any],
        reason: str,
    ) -> Optional[str]:
        """Async counterpart of :meth:`_run_approval_handler`."""
        request = self._build_approval_request(tool_name, tool_args, reason)
        try:
            if self._is_async_approval_handler(handler):
                approved = await asyncio.wait_for(
                    handler(request), timeout=self.config.approval_timeout
                )
            else:
                # BUG-32: propagate caller contextvars (OTel / Langfuse)
                # into the handler worker thread.
                loop = asyncio.get_running_loop()
                approved = await asyncio.wait_for(
                    run_in_executor_copyctx(loop, None, lambda: handler(request)),
                    timeout=self.config.approval_timeout,
                )
                if inspect.isawaitable(approved):
                    # Belt-and-suspenders: sync-classified handler returned an
                    # awaitable — await it instead of truth-testing it.
                    approved = await asyncio.wait_for(
                        approved, timeout=self.config.approval_timeout
                    )
        except asyncio.TimeoutError:
            return f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
        except Exception as exc:
            return f"Tool '{tool_name}' approval failed: {exc}"
        return self._approval_verdict(tool_name, approved, reason)

    def _check_policy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        run_id: str = "",
        run_ctx: Optional["_RunContext"] = None,
    ) -> Optional[str]:
        """Evaluate tool policy and confirm_action. Returns error string or None.

        When ``run_ctx`` is provided, approval-handler denials are memoized
        per ``(tool_name, args digest)`` for the duration of the run so a
        human is not re-paged when the model retries the identical denied
        call on a later iteration. Approvals are NOT memoized.
        """
        # Check per-tool requires_approval flag and the agent-level
        # ToolConfig.require_approval gate even without a ToolPolicy
        tool_obj = self._tools_by_name.get(tool_name) if hasattr(self, "_tools_by_name") else None
        tool_flag = bool(tool_obj and getattr(tool_obj, "requires_approval", False))
        config_gated = self._config_requires_approval(tool_name)
        tool_requires_approval = tool_flag or config_gated

        if not self.config.tool_policy and not tool_requires_approval:
            return None

        if self.config.tool_policy:
            result = self.config.tool_policy.evaluate(tool_name, tool_args)
        else:
            result = PolicyResult(
                decision=PolicyDecision.ALLOW, reason="no policy", matched_rule=""
            )
        decision_str = result.decision.value

        # Override ALLOW to REVIEW if the tool itself requires approval
        if result.decision == PolicyDecision.ALLOW and tool_requires_approval:
            result = PolicyResult(
                decision=PolicyDecision.REVIEW,
                reason=f"Tool '{tool_name}' requires approval",
                matched_rule=(
                    "tool.requires_approval" if tool_flag else "tool_config.require_approval"
                ),
            )
            decision_str = result.decision.value

        if run_id:
            self._notify_observers(
                "on_policy_decision",
                run_id,
                tool_name,
                decision_str,
                result.reason,
                tool_args,
            )

        if result.decision == PolicyDecision.ALLOW:
            return None

        if result.decision == PolicyDecision.DENY:
            return f"Tool '{tool_name}' denied by policy: {result.reason}"

        if result.decision == PolicyDecision.REVIEW:
            memo_key = self._approval_memo_key(tool_name, tool_args)
            if run_ctx is not None:
                memoized_denial = run_ctx.denied_approvals.get(memo_key)
                if memoized_denial is not None:
                    return memoized_denial
            approval_handler = getattr(self.config.tool, "approval_handler", None)
            if approval_handler is not None:
                error = self._run_approval_handler(
                    approval_handler, tool_name, tool_args, result.reason
                )
                if error is not None and run_ctx is not None:
                    run_ctx.denied_approvals[memo_key] = error
                return error
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            if inspect.iscoroutinefunction(self.config.confirm_action):
                return (
                    f"Tool '{tool_name}' requires approval but confirm_action is async. "
                    f"Use arun() or astream() instead of run() for async callbacks."
                )
            try:
                # Reuse the shared module-level executor — never create a new
                # ThreadPoolExecutor per call (pitfall #20).
                executor = _get_tool_timeout_executor()
                future = executor.submit(
                    self.config.confirm_action, tool_name, tool_args, result.reason
                )
                try:
                    approved = future.result(timeout=self.config.approval_timeout)
                except FuturesTimeoutError:
                    # Same gap as the approval-handler path (sweep): cancel so
                    # a still-queued confirm_action can never fire after the
                    # call was already denied.
                    future.cancel()
                    return (
                        f"Tool '{tool_name}' approval timed out "
                        f"after {self.config.approval_timeout}s"
                    )
                if not approved:
                    return f"Tool '{tool_name}' rejected by reviewer: {result.reason}"
            except FuturesTimeoutError:
                return (
                    f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
                )
            except Exception as exc:
                return f"Tool '{tool_name}' approval failed: {exc}"

        return None

    async def _acheck_policy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        run_id: str = "",
        run_ctx: Optional["_RunContext"] = None,
    ) -> Optional[str]:
        """Async version of _check_policy (same denial memoization semantics)."""
        # Check per-tool requires_approval flag and the agent-level
        # ToolConfig.require_approval gate even without a ToolPolicy
        tool_obj = self._tools_by_name.get(tool_name) if hasattr(self, "_tools_by_name") else None
        tool_flag = bool(tool_obj and getattr(tool_obj, "requires_approval", False))
        config_gated = self._config_requires_approval(tool_name)
        tool_requires_approval = tool_flag or config_gated

        if not self.config.tool_policy and not tool_requires_approval:
            return None

        if self.config.tool_policy:
            result = self.config.tool_policy.evaluate(tool_name, tool_args)
        else:
            result = PolicyResult(
                decision=PolicyDecision.ALLOW, reason="no policy", matched_rule=""
            )
        decision_str = result.decision.value

        # Override ALLOW to REVIEW if the tool itself requires approval
        if result.decision == PolicyDecision.ALLOW and tool_requires_approval:
            result = PolicyResult(
                decision=PolicyDecision.REVIEW,
                reason=f"Tool '{tool_name}' requires approval",
                matched_rule=(
                    "tool.requires_approval" if tool_flag else "tool_config.require_approval"
                ),
            )
            decision_str = result.decision.value

        if run_id:
            self._notify_observers(
                "on_policy_decision",
                run_id,
                tool_name,
                decision_str,
                result.reason,
                tool_args,
            )
            await self._anotify_observers(
                "on_policy_decision",
                run_id,
                tool_name,
                decision_str,
                result.reason,
                tool_args,
            )

        if result.decision == PolicyDecision.ALLOW:
            return None

        if result.decision == PolicyDecision.DENY:
            return f"Tool '{tool_name}' denied by policy: {result.reason}"

        if result.decision == PolicyDecision.REVIEW:
            memo_key = self._approval_memo_key(tool_name, tool_args)
            if run_ctx is not None:
                memoized_denial = run_ctx.denied_approvals.get(memo_key)
                if memoized_denial is not None:
                    return memoized_denial
            approval_handler = getattr(self.config.tool, "approval_handler", None)
            if approval_handler is not None:
                error = await self._arun_approval_handler(
                    approval_handler, tool_name, tool_args, result.reason
                )
                if error is not None and run_ctx is not None:
                    run_ctx.denied_approvals[memo_key] = error
                return error
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            try:
                if inspect.iscoroutinefunction(self.config.confirm_action):
                    approved = await asyncio.wait_for(
                        self.config.confirm_action(tool_name, tool_args, result.reason),
                        timeout=self.config.approval_timeout,
                    )
                else:
                    # BUG-32: propagate caller contextvars (OTel / Langfuse)
                    # into the confirm_action worker thread.
                    loop = asyncio.get_running_loop()
                    confirm_fn = self.config.confirm_action
                    approved = await asyncio.wait_for(
                        run_in_executor_copyctx(
                            loop,
                            None,
                            lambda: confirm_fn(tool_name, tool_args, result.reason),
                        ),
                        timeout=self.config.approval_timeout,
                    )
                if not approved:
                    return f"Tool '{tool_name}' rejected by reviewer: {result.reason}"
            except asyncio.TimeoutError:
                return (
                    f"Tool '{tool_name}' approval timed out after {self.config.approval_timeout}s"
                )
            except Exception as exc:
                return f"Tool '{tool_name}' approval failed: {exc}"

        return None

    def _append_tool_result(
        self,
        tool_content: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tool_result: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Update history with tool output."""
        tool_msg = Message(
            role=Role.TOOL,
            content=tool_content,
            tool_name=tool_name,
            tool_result=tool_result,
            tool_call_id=tool_call_id,
        )
        self._history.append(tool_msg)
        self._memory_add(tool_msg, run_id or "")

    # ------------------------------------------------------------------
    # Parallel tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
        user_text_for_coherence: str = "",
        all_tool_results: Optional[List[str]] = None,
        run_ctx: Optional["_RunContext"] = None,
    ) -> tuple:
        """Execute multiple tool calls concurrently using ThreadPoolExecutor.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int
            error_step_type: "StepType" = StepType.ERROR
            cached: bool = False
            cached_compressed: Optional[str] = None

        def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            policy_error = self._check_policy(tool_name, parameters, run_id or "", run_ctx=run_ctx)
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            coherence_error = self._check_coherence(user_text_for_coherence, tool_name, parameters)
            if coherence_error:
                return _Result(
                    tc,
                    coherence_error,
                    True,
                    0.0,
                    tool,
                    0,
                    error_step_type=StepType.COHERENCE_CHECK,
                )

            # Tool result cache check
            cached_entry = self._check_tool_cache(tool, parameters)
            if cached_entry is not None:
                raw, compressed = cached_entry
                return _Result(
                    tc, raw, False, 0.0, tool, 0, cached=True, cached_compressed=compressed
                )

            call_id = tc.id or ""
            start = time.time()
            if run_id:
                self._notify_observers("on_tool_start", run_id, call_id, tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                if run_id:
                    self._notify_observers(
                        "on_tool_chunk",
                        run_id,
                        call_id,
                        tool_name,
                        chunk,
                    )

            try:
                result = self._execute_tool_with_timeout(tool, parameters, chunk_cb)
                result = self._screen_tool_result(tool_name, result)
                self._store_tool_cache(tool, parameters, result)
                dur = time.time() - start
                if run_id:
                    self._notify_observers(
                        "on_tool_end",
                        run_id,
                        call_id,
                        tool_name,
                        result,
                        dur * 1000,
                    )
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                if run_id:
                    self._notify_observers(
                        "on_tool_error",
                        run_id,
                        call_id,
                        tool_name,
                        exc,
                        parameters,
                        dur * 1000,
                    )
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        # Submit outer _run_one tasks to the dedicated parallel-dispatch pool
        # (NOT the tool-timeout pool).  Using the same pool for both levels
        # causes a thread-pool deadlock when len(tool_calls_to_execute) >=
        # max_workers/2+1: every dispatch-worker blocks on an inner timeout
        # submission that can never start because all slots are taken.
        pool = _get_parallel_dispatch_executor()
        # Propagate caller contextvars into dispatch workers (pitfall #28 /
        # BUG-32) so emit_artifact() reaches the per-run collector. Each
        # submission needs its OWN context copy: Context.run raises if the
        # same Context is entered concurrently.
        futures = [
            pool.submit(contextvars.copy_context().run, _run_one, tc)
            for tc in tool_calls_to_execute
        ]
        results = [f.result() for f in futures]  # preserves order

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}
        terminal_result: Optional[str] = None

        # tool_tokens attribution: capture the PARENT iteration's usage once,
        # BEFORE compression appends its own usage entries below — otherwise
        # the second result in the batch would be attributed the compression
        # call's tokens instead of the iteration that requested the tools.
        parent_iteration_usage = self.usage.iterations[-1] if self.usage.iterations else None

        for r in results:
            all_tool_calls.append(r.tool_call)
            if all_tool_results is not None:
                all_tool_results.append(r.result)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )
            if not r.is_error and parent_iteration_usage is not None:
                self.usage.tool_usage[r.tool_call.tool_name] = (
                    self.usage.tool_usage.get(r.tool_call.tool_name, 0) + 1
                )
                self.usage.tool_tokens[r.tool_call.tool_name] = (
                    self.usage.tool_tokens.get(r.tool_call.tool_name, 0)
                    + parent_iteration_usage.total_tokens
                )

            if trace is not None:
                step_type = r.error_step_type if r.is_error else StepType.TOOL_EXECUTION
                trace.add(
                    TraceStep(
                        type=step_type,
                        duration_ms=r.duration * 1000,
                        tool_name=r.tool_call.tool_name,
                        tool_args=r.tool_call.parameters,
                        tool_result=(
                            self._truncate_tool_result(r.result) if not r.is_error else None
                        ),
                        error=r.result if r.is_error else None,
                        summary=f"{r.tool_call.tool_name} → {'error' if r.is_error else f'{len(r.result)} chars'}",
                    )
                )

            # Evaluate stop_condition exactly once per result (always on the
            # RAW result) and reuse the boolean for both the compression gate
            # and the terminal check. A raising predicate propagates.
            stop_hit = not r.is_error and self._stop_condition_hit(r.tool_call.tool_name, r.result)

            if r.is_error:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    run_id=run_id,
                )
            else:
                # Tool result compression runs here in the dispatching thread
                # (after workers returned), never inside the worker pool.
                # NOTE: this sync path compresses SEQUENTIALLY by design — the
                # shared thread pools serve every concurrent agent run, so a
                # batch with N oversized results costs N back-to-back
                # summarizer round-trips (documented in docs/modules/TOOLS.md).
                # The async path uses an asyncio.gather pre-pass instead.
                message_content = r.result
                if r.cached:
                    # Cache hit: append the stored compressed text when present;
                    # terminal/stop checks still use the RAW result above/below.
                    if r.cached_compressed is not None:
                        message_content = r.cached_compressed
                elif not stop_hit and self._tool_compression_candidate(r.tool, r.result):
                    message_content = self._compress_tool_result(
                        r.tool_call.tool_name, r.result, trace, run_ctx=run_ctx
                    )
                    self._store_compressed_in_cache(
                        r.tool, r.tool_call.parameters, r.result, message_content
                    )
                self._append_tool_result(
                    message_content,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=message_content,
                    run_id=run_id,
                )

            # Check terminal tool
            if not r.is_error and r.tool and terminal_result is None:
                if getattr(r.tool, "terminal", False) or stop_hit:
                    terminal_result = r.result

        return last_tool_name, last_tool_args, terminal_result

    async def _aexecute_tools_parallel(
        self,
        tool_calls_to_execute: List[ToolCall],
        all_tool_calls: List[ToolCall],
        iteration: int,
        response_text: str,
        trace: Optional[AgentTrace] = None,
        run_id: Optional[str] = None,
        user_text_for_coherence: str = "",
        all_tool_results: Optional[List[str]] = None,
        run_ctx: Optional["_RunContext"] = None,
    ) -> tuple:
        """Execute multiple tool calls concurrently using asyncio.gather.

        Returns (last_tool_name, last_tool_args) from the batch.
        Results are appended to history in the original request order.
        """

        @dataclass
        class _Result:
            tool_call: ToolCall
            result: str
            is_error: bool
            duration: float
            tool: Optional[Tool]
            chunk_count: int
            error_step_type: "StepType" = StepType.ERROR
            cached: bool = False
            cached_compressed: Optional[str] = None

        async def _run_one(tc: ToolCall) -> _Result:
            tool_name = tc.tool_name
            parameters = tc.parameters
            call_id = tc.id or ""
            tool = self._tools_by_name.get(tool_name)

            if not tool:
                error_msg = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {', '.join(self._tools_by_name.keys())}"
                )
                return _Result(tc, error_msg, True, 0.0, None, 0)

            policy_error = await self._acheck_policy(
                tool_name, parameters, run_id or "", run_ctx=run_ctx
            )
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            coherence_error = await self._acheck_coherence(
                user_text_for_coherence, tool_name, parameters
            )
            if coherence_error:
                return _Result(
                    tc,
                    coherence_error,
                    True,
                    0.0,
                    tool,
                    0,
                    error_step_type=StepType.COHERENCE_CHECK,
                )

            # Tool result cache check
            cached_entry = self._check_tool_cache(tool, parameters)
            if cached_entry is not None:
                raw, compressed = cached_entry
                return _Result(
                    tc, raw, False, 0.0, tool, 0, cached=True, cached_compressed=compressed
                )

            start = time.time()
            if run_id:
                self._notify_observers("on_tool_start", run_id, call_id, tool_name, parameters)
                await self._anotify_observers(
                    "on_tool_start", run_id, call_id, tool_name, parameters
                )

            chunk_counter = {"count": 0}

            def chunk_cb(chunk: str) -> None:
                chunk_counter["count"] += 1
                if run_id:
                    self._notify_observers(
                        "on_tool_chunk",
                        run_id,
                        call_id,
                        tool_name,
                        chunk,
                    )

            try:
                result = await self._aexecute_tool_with_timeout(tool, parameters, chunk_cb)
                result = self._screen_tool_result(tool_name, result)
                self._store_tool_cache(tool, parameters, result)
                dur = time.time() - start
                if run_id:
                    self._notify_observers(
                        "on_tool_end",
                        run_id,
                        call_id,
                        tool_name,
                        result,
                        dur * 1000,
                    )
                    await self._anotify_observers(
                        "on_tool_end",
                        run_id,
                        call_id,
                        tool_name,
                        result,
                        dur * 1000,
                    )
                return _Result(tc, result, False, dur, tool, chunk_counter["count"])
            except Exception as exc:
                dur = time.time() - start
                if run_id:
                    self._notify_observers(
                        "on_tool_error",
                        run_id,
                        call_id,
                        tool_name,
                        exc,
                        parameters,
                        dur * 1000,
                    )
                    await self._anotify_observers(
                        "on_tool_error",
                        run_id,
                        call_id,
                        tool_name,
                        exc,
                        parameters,
                        dur * 1000,
                    )
                error_msg = f"Error executing tool '{tool_name}': {exc}"
                return _Result(tc, error_msg, True, dur, tool, 0)

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls_to_execute])

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}
        terminal_result: Optional[str] = None

        # tool_tokens attribution: capture the PARENT iteration's usage once,
        # BEFORE the compression pre-pass appends its own usage entries —
        # otherwise per-result attribution would pick up compression-call
        # tokens instead of the iteration that requested the tools.
        parent_iteration_usage = self.usage.iterations[-1] if self.usage.iterations else None

        # Evaluate stop_condition exactly once per result (always on the RAW
        # result); the booleans gate compression below and feed the terminal
        # check in the result loop. A raising predicate propagates.
        stop_hits = [
            (not r.is_error) and self._stop_condition_hit(r.tool_call.tool_name, r.result)
            for r in results
        ]

        # Compression pre-pass: summarize every oversized candidate
        # CONCURRENTLY in a single asyncio.gather instead of one sequential
        # round-trip per result inside the loop below.
        compressed_by_index: Dict[int, str] = {}
        candidate_indices = [
            i
            for i, r in enumerate(results)
            if not r.is_error
            and not r.cached
            and not stop_hits[i]
            and self._tool_compression_candidate(r.tool, r.result)
        ]
        if candidate_indices:
            compressed_texts = await asyncio.gather(
                *[
                    self._acompress_tool_result(
                        results[i].tool_call.tool_name, results[i].result, trace, run_ctx=run_ctx
                    )
                    for i in candidate_indices
                ]
            )
            for i, text in zip(candidate_indices, compressed_texts):
                compressed_by_index[i] = text
                self._store_compressed_in_cache(
                    results[i].tool, results[i].tool_call.parameters, results[i].result, text
                )

        for idx, r in enumerate(results):
            all_tool_calls.append(r.tool_call)
            if all_tool_results is not None:
                all_tool_results.append(r.result)
            last_tool_name = r.tool_call.tool_name
            last_tool_args = r.tool_call.parameters

            if self.config.verbose:
                status = "OK" if not r.is_error else "ERR"
                print(
                    f"[agent] Iteration {iteration}: tool={r.tool_call.tool_name} "
                    f"[{status}] {r.duration:.3f}s"
                )

            # Record analytics
            if self.analytics and r.tool:
                self.analytics.record_tool_call(
                    tool_name=r.tool.name,
                    success=not r.is_error,
                    duration=r.duration,
                    params=r.tool_call.parameters,
                    cost=0.0,
                    chunk_count=r.chunk_count,
                )
            if not r.is_error and parent_iteration_usage is not None:
                self.usage.tool_usage[r.tool_call.tool_name] = (
                    self.usage.tool_usage.get(r.tool_call.tool_name, 0) + 1
                )
                self.usage.tool_tokens[r.tool_call.tool_name] = (
                    self.usage.tool_tokens.get(r.tool_call.tool_name, 0)
                    + parent_iteration_usage.total_tokens
                )

            if trace is not None:
                step_type = r.error_step_type if r.is_error else StepType.TOOL_EXECUTION
                trace.add(
                    TraceStep(
                        type=step_type,
                        duration_ms=r.duration * 1000,
                        tool_name=r.tool_call.tool_name,
                        tool_args=r.tool_call.parameters,
                        tool_result=(
                            self._truncate_tool_result(r.result) if not r.is_error else None
                        ),
                        error=r.result if r.is_error else None,
                        summary=f"{r.tool_call.tool_name} → {'error' if r.is_error else f'{len(r.result)} chars'}",
                    )
                )

            if r.is_error:
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    run_id=run_id,
                )
            else:
                # Compressed text comes from the gather pre-pass above (or the
                # cached compressed form on cache hits); never compressed
                # inside tool coroutines.
                message_content = r.result
                if r.cached:
                    if r.cached_compressed is not None:
                        message_content = r.cached_compressed
                elif idx in compressed_by_index:
                    message_content = compressed_by_index[idx]
                self._append_tool_result(
                    message_content,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=message_content,
                    run_id=run_id,
                )

            # Check terminal tool
            if not r.is_error and r.tool and terminal_result is None:
                if getattr(r.tool, "terminal", False) or stop_hits[idx]:
                    terminal_result = r.result

        return last_tool_name, last_tool_args, terminal_result

    def _execute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run tool.execute with optional timeout and chunk callback."""
        if self.config.tool_timeout_seconds is None:
            return tool.execute(parameters, chunk_callback=chunk_callback)

        executor = _get_tool_timeout_executor()
        # Propagate caller contextvars into the timeout worker (pitfall #28 /
        # BUG-32) so emit_artifact() reaches the per-run collector.
        ctx_copy = contextvars.copy_context()
        future = executor.submit(ctx_copy.run, tool.execute, parameters, chunk_callback)
        try:
            return future.result(timeout=self.config.tool_timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )

    async def _aexecute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Async version of _execute_tool_with_timeout."""
        if self.config.tool_timeout_seconds is None:
            return await tool.aexecute(parameters, chunk_callback=chunk_callback)

        try:
            return await asyncio.wait_for(
                tool.aexecute(parameters, chunk_callback=chunk_callback),
                timeout=self.config.tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )

    # ------------------------------------------------------------------
    # Single tool execution helpers (used by the sequential tool loop)
    # ------------------------------------------------------------------

    def _execute_single_tool(self, ctx: _RunContext, tool_call: ToolCall) -> bool:
        """Execute a single tool call end-to-end (sync).

        Handles: lookup, policy, coherence, execute with timeout, screen,
        trace, analytics, usage, observers, error handling, and appending
        the result to history.

        Returns True if the agent loop should stop (terminal tool).
        """
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        call_id = tool_call.id or ""

        ctx.all_tool_calls.append(tool_call)
        ctx.last_tool_name = tool_name
        ctx.last_tool_args = parameters

        if self.config.verbose:
            print(f"[agent] Iteration {ctx.iteration}: tool={tool_name} params={parameters}")

        # --- Malformed tool-call arguments (BUG-31 / Pydantic AI #4609) ---
        # Providers surface parse errors on the ToolCall so the LLM learns
        # its JSON was the problem instead of silently retrying with the
        # same malformed call.
        if tool_call.parse_error is not None:
            error_message = (
                f"Tool call for '{tool_name}' had malformed arguments: "
                f"{tool_call.parse_error}. Retry with properly escaped JSON."
            )
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=error_message,
                    summary=f"Malformed arguments for {tool_name}",
                )
            )
            return False

        # --- Tool lookup ---
        tool = self._tools_by_name.get(tool_name)
        if not tool:
            error_message = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(self._tools_by_name.keys())}"
            )
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=error_message,
                    summary=f"Unknown tool {tool_name}",
                )
            )
            return False

        # --- Policy check ---
        policy_error = self._check_policy(tool_name, parameters, ctx.run_id, run_ctx=ctx)
        if policy_error:
            ctx.all_tool_results.append(policy_error)
            self._append_tool_result(policy_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=policy_error,
                    summary=f"Policy denied {tool_name}",
                )
            )
            return False

        # --- Coherence check ---
        coherence_error = self._check_coherence(ctx.user_text_for_coherence, tool_name, parameters)
        if coherence_error:
            ctx.all_tool_results.append(coherence_error)
            self._append_tool_result(coherence_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.COHERENCE_CHECK,
                    tool_name=tool_name,
                    error=coherence_error,
                    summary=f"Coherence check failed for {tool_name}",
                )
            )
            return False

        # --- Tool result cache check ---
        cached_entry = self._check_tool_cache(tool, parameters)
        if cached_entry is not None:
            cached_raw, cached_compressed = cached_entry
            ctx.trace.add(
                TraceStep(
                    type=StepType.CACHE_HIT,
                    tool_name=tool_name,
                    tool_args=parameters,
                    tool_result=self._truncate_tool_result(cached_raw),
                    summary=f"{tool_name} → cached",
                )
            )
            # History gets the stored compressed text when present; loop
            # detection and terminal checks always see the RAW result.
            cached_content = cached_compressed if cached_compressed is not None else cached_raw
            ctx.all_tool_results.append(cached_raw)
            self._append_tool_result(
                cached_content,
                tool_name,
                tool_call.id,
                tool_result=cached_content,
                run_id=ctx.run_id,
            )
            if getattr(tool, "terminal", False) or self._stop_condition_hit(tool_name, cached_raw):
                ctx.terminal_tool_result = cached_raw
                return True
            return False

        # --- Execute ---
        try:
            start_time = time.time()
            self._notify_observers("on_tool_start", ctx.run_id, call_id, tool_name, parameters)

            chunk_counter = {"count": 0}

            def chunk_callback(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._notify_observers("on_tool_chunk", ctx.run_id, call_id, tool_name, chunk)

            result = self._execute_tool_with_timeout(tool, parameters, chunk_callback)
            result = self._screen_tool_result(tool_name, result)
            duration = time.time() - start_time

            self._notify_observers(
                "on_tool_end", ctx.run_id, call_id, tool_name, result, duration * 1000
            )

            # Store in tool result cache
            self._store_tool_cache(tool, parameters, result)

            ctx.trace.add(
                TraceStep(
                    type=StepType.TOOL_EXECUTION,
                    duration_ms=duration * 1000,
                    tool_name=tool_name,
                    tool_args=parameters,
                    tool_result=self._truncate_tool_result(result),
                    summary=f"{tool_name} → {len(result)} chars",
                )
            )

            if self.analytics:
                self.analytics.record_tool_call(
                    tool_name=tool.name,
                    success=True,
                    duration=duration,
                    params=parameters,
                    cost=0.0,
                    chunk_count=chunk_counter["count"],
                )
            # tool_tokens attribution: prefer the parent-iteration usage
            # captured before this turn's tool batch — a previous tool's
            # compression call may have appended its own usage entry since.
            parent_usage = ctx.parent_iteration_usage or (
                self.usage.iterations[-1] if self.usage.iterations else None
            )
            if parent_usage is not None:
                self.usage.tool_usage[tool.name] = self.usage.tool_usage.get(tool.name, 0) + 1
                self.usage.tool_tokens[tool.name] = (
                    self.usage.tool_tokens.get(tool.name, 0) + parent_usage.total_tokens
                )

        except Exception as exc:
            duration = time.time() - start_time
            self._notify_observers(
                "on_tool_error",
                ctx.run_id,
                call_id,
                tool_name,
                exc,
                parameters,
                duration * 1000,
            )
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    duration_ms=duration * 1000,
                    tool_name=tool_name,
                    error=str(exc),
                    summary=f"{tool_name} failed: {exc}",
                )
            )
            if self.analytics:
                self.analytics.record_tool_call(
                    tool_name=tool.name,
                    success=False,
                    duration=duration,
                    params=parameters,
                    cost=0.0,
                    chunk_count=0,
                )

            error_message = f"Error executing tool '{tool_name}': {exc}"
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            return False

        # Evaluate stop_condition exactly once, OUTSIDE the execute try: a
        # raising predicate propagates to the caller instead of converting a
        # successful tool result into a tool error.
        stop_hit = self._stop_condition_hit(tool_name, result)

        # Tool result compression: the model sees the compressed text, but
        # ctx.all_tool_results keeps the raw result (loop detection needs
        # deterministic outputs) and terminal checks run on the raw result.
        message_content = result
        if not stop_hit and self._tool_compression_candidate(tool, result):
            message_content = self._compress_tool_result(tool_name, result, ctx.trace, run_ctx=ctx)
            self._store_compressed_in_cache(tool, parameters, result, message_content)

        ctx.all_tool_results.append(result)
        self._append_tool_result(
            message_content,
            tool_name,
            tool_call.id,
            tool_result=message_content,
            run_id=ctx.run_id,
        )

        # --- Terminal tool check ---
        if getattr(tool, "terminal", False) or stop_hit:
            ctx.terminal_tool_result = result
            return True

        return False

    async def _aexecute_single_tool(self, ctx: _RunContext, tool_call: ToolCall) -> bool:
        """Execute a single tool call end-to-end (async).

        Async counterpart of :meth:`_execute_single_tool`.  Uses async
        policy/coherence checks, async tool execution, and fires async
        observer notifications alongside sync ones.

        Returns True if the agent loop should stop (terminal tool).
        """
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        call_id = tool_call.id or ""

        ctx.all_tool_calls.append(tool_call)
        ctx.last_tool_name = tool_name
        ctx.last_tool_args = parameters

        if self.config.verbose:
            print(f"[agent] Iteration {ctx.iteration}: tool={tool_name} params={parameters}")

        # --- Malformed tool-call arguments (BUG-31 / Pydantic AI #4609) ---
        if tool_call.parse_error is not None:
            error_message = (
                f"Tool call for '{tool_name}' had malformed arguments: "
                f"{tool_call.parse_error}. Retry with properly escaped JSON."
            )
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=error_message,
                    summary=f"Malformed arguments for {tool_name}",
                )
            )
            return False

        # --- Tool lookup ---
        tool = self._tools_by_name.get(tool_name)
        if not tool:
            error_message = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(self._tools_by_name.keys())}"
            )
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=error_message,
                    summary=f"Unknown tool {tool_name}",
                )
            )
            return False

        # --- Policy check (async) ---
        policy_error = await self._acheck_policy(tool_name, parameters, ctx.run_id, run_ctx=ctx)
        if policy_error:
            ctx.all_tool_results.append(policy_error)
            self._append_tool_result(policy_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=policy_error,
                    summary=f"Policy denied {tool_name}",
                )
            )
            return False

        # --- Coherence check (async) ---
        coherence_error = await self._acheck_coherence(
            ctx.user_text_for_coherence, tool_name, parameters
        )
        if coherence_error:
            ctx.all_tool_results.append(coherence_error)
            self._append_tool_result(coherence_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.COHERENCE_CHECK,
                    tool_name=tool_name,
                    error=coherence_error,
                    summary=f"Coherence check failed for {tool_name}",
                )
            )
            return False

        # --- Tool result cache check ---
        cached_entry = self._check_tool_cache(tool, parameters)
        if cached_entry is not None:
            cached_raw, cached_compressed = cached_entry
            ctx.trace.add(
                TraceStep(
                    type=StepType.CACHE_HIT,
                    tool_name=tool_name,
                    tool_args=parameters,
                    tool_result=self._truncate_tool_result(cached_raw),
                    summary=f"{tool_name} → cached",
                )
            )
            # History gets the stored compressed text when present; loop
            # detection and terminal checks always see the RAW result.
            cached_content = cached_compressed if cached_compressed is not None else cached_raw
            ctx.all_tool_results.append(cached_raw)
            self._append_tool_result(
                cached_content,
                tool_name,
                tool_call.id,
                tool_result=cached_content,
                run_id=ctx.run_id,
            )
            if getattr(tool, "terminal", False) or self._stop_condition_hit(tool_name, cached_raw):
                ctx.terminal_tool_result = cached_raw
                return True
            return False

        # --- Execute (async) ---
        try:
            start_time = time.time()
            self._notify_observers("on_tool_start", ctx.run_id, call_id, tool_name, parameters)
            await self._anotify_observers(
                "on_tool_start", ctx.run_id, call_id, tool_name, parameters
            )

            chunk_counter = {"count": 0}

            def chunk_callback(chunk: str) -> None:
                chunk_counter["count"] += 1
                self._notify_observers("on_tool_chunk", ctx.run_id, call_id, tool_name, chunk)

            result = await self._aexecute_tool_with_timeout(tool, parameters, chunk_callback)
            result = self._screen_tool_result(tool_name, result)
            duration = time.time() - start_time

            self._notify_observers(
                "on_tool_end", ctx.run_id, call_id, tool_name, result, duration * 1000
            )
            await self._anotify_observers(
                "on_tool_end", ctx.run_id, call_id, tool_name, result, duration * 1000
            )

            # Store in tool result cache
            self._store_tool_cache(tool, parameters, result)

            ctx.trace.add(
                TraceStep(
                    type=StepType.TOOL_EXECUTION,
                    duration_ms=duration * 1000,
                    tool_name=tool_name,
                    tool_args=parameters,
                    tool_result=self._truncate_tool_result(result),
                    summary=f"{tool_name} → {len(result)} chars",
                )
            )

            if self.analytics:
                self.analytics.record_tool_call(
                    tool_name=tool.name,
                    success=True,
                    duration=duration,
                    params=parameters,
                    cost=0.0,
                    chunk_count=chunk_counter["count"],
                )
            # tool_tokens attribution: prefer the parent-iteration usage
            # captured before this turn's tool batch — a previous tool's
            # compression call may have appended its own usage entry since.
            parent_usage = ctx.parent_iteration_usage or (
                self.usage.iterations[-1] if self.usage.iterations else None
            )
            if parent_usage is not None:
                self.usage.tool_usage[tool.name] = self.usage.tool_usage.get(tool.name, 0) + 1
                self.usage.tool_tokens[tool.name] = (
                    self.usage.tool_tokens.get(tool.name, 0) + parent_usage.total_tokens
                )

        except Exception as exc:
            duration = time.time() - start_time
            self._notify_observers(
                "on_tool_error",
                ctx.run_id,
                call_id,
                tool_name,
                exc,
                parameters,
                duration * 1000,
            )
            await self._anotify_observers(
                "on_tool_error",
                ctx.run_id,
                call_id,
                tool_name,
                exc,
                parameters,
                duration * 1000,
            )
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    duration_ms=duration * 1000,
                    tool_name=tool_name,
                    error=str(exc),
                    summary=f"{tool_name} failed: {exc}",
                )
            )
            if self.analytics:
                self.analytics.record_tool_call(
                    tool_name=tool.name,
                    success=False,
                    duration=duration,
                    params=parameters,
                    cost=0.0,
                    chunk_count=0,
                )

            error_message = f"Error executing tool '{tool_name}': {exc}"
            ctx.all_tool_results.append(error_message)
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            return False

        # Evaluate stop_condition exactly once, OUTSIDE the execute try: a
        # raising predicate propagates to the caller instead of converting a
        # successful tool result into a tool error.
        stop_hit = self._stop_condition_hit(tool_name, result)

        # Tool result compression (see sync counterpart for invariants).
        message_content = result
        if not stop_hit and self._tool_compression_candidate(tool, result):
            message_content = await self._acompress_tool_result(
                tool_name, result, ctx.trace, run_ctx=ctx
            )
            self._store_compressed_in_cache(tool, parameters, result, message_content)

        ctx.all_tool_results.append(result)
        self._append_tool_result(
            message_content,
            tool_name,
            tool_call.id,
            tool_result=message_content,
            run_id=ctx.run_id,
        )

        # --- Terminal tool check ---
        if getattr(tool, "terminal", False) or stop_hit:
            ctx.terminal_tool_result = result
            return True

        return False
