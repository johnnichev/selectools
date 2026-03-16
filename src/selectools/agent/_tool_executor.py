"""Mixin providing tool execution methods for the Agent class."""

from __future__ import annotations

import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..coherence import CoherenceResult, acheck_coherence, check_coherence
from ..policy import PolicyDecision, ToolPolicy
from ..security import screen_output as screen_tool_output
from ..trace import StepType, TraceStep
from ..types import Message, Role

if TYPE_CHECKING:
    from ..tools import Tool
    from ..trace import AgentTrace
    from ..types import ToolCall
    from .core import _RunContext


class _ToolExecutorMixin:
    """Mixin that provides tool execution methods for the Agent class.

    All methods access ``self.*`` attributes (config, provider, _history, usage,
    analytics, _tools_by_name, etc.) which are expected to be provided by the
    Agent class that inherits from this mixin.
    """

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
        model = self.config.coherence_model or self.config.model
        result = check_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
        )
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
        model = self.config.coherence_model or self.config.model
        result = await acheck_coherence(
            provider=provider,
            model=model,
            user_message=user_message,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=list(self._tools_by_name.keys()),
            timeout=self.config.request_timeout,
        )
        if not result.coherent:
            return (
                f"Coherence check failed for tool '{tool_name}': "
                f"{result.explanation or 'Tool call does not match user intent'}"
            )
        return None

    def _check_policy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        run_id: str = "",
    ) -> Optional[str]:
        """Evaluate tool policy and confirm_action. Returns error string or None."""
        if not self.config.tool_policy:
            return None

        result = self.config.tool_policy.evaluate(tool_name, tool_args)
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
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.config.confirm_action, tool_name, tool_args, result.reason
                    )
                    try:
                        approved = future.result(timeout=self.config.approval_timeout)
                    except FuturesTimeoutError:
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
    ) -> Optional[str]:
        """Async version of _check_policy."""
        if not self.config.tool_policy:
            return None

        result = self.config.tool_policy.evaluate(tool_name, tool_args)
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
            if self.config.confirm_action is None:
                return f"Tool '{tool_name}' requires approval but no confirm_action configured: {result.reason}"
            try:
                if inspect.iscoroutinefunction(self.config.confirm_action):
                    approved = await asyncio.wait_for(
                        self.config.confirm_action(tool_name, tool_args, result.reason),
                        timeout=self.config.approval_timeout,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    approved = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            self.config.confirm_action,
                            tool_name,
                            tool_args,
                            result.reason,
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

            policy_error = self._check_policy(tool_name, parameters, run_id or "")
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            coherence_error = self._check_coherence(user_text_for_coherence, tool_name, parameters)
            if coherence_error:
                return _Result(tc, coherence_error, True, 0.0, tool, 0)

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

        # Submit all tool calls to the thread pool
        with ThreadPoolExecutor(max_workers=len(tool_calls_to_execute)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls_to_execute]
            results = [f.result() for f in futures]  # preserves order

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}
        terminal_result: Optional[str] = None

        for r in results:
            all_tool_calls.append(r.tool_call)
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
            if not r.is_error and self.usage.iterations:
                self.usage.tool_usage[r.tool_call.tool_name] = (
                    self.usage.tool_usage.get(r.tool_call.tool_name, 0) + 1
                )
                self.usage.tool_tokens[r.tool_call.tool_name] = (
                    self.usage.tool_tokens.get(r.tool_call.tool_name, 0)
                    + self.usage.iterations[-1].total_tokens
                )

            if trace is not None:
                step_type = StepType.ERROR if r.is_error else StepType.TOOL_EXECUTION
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
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=r.result,
                    run_id=run_id,
                )

            # Check terminal tool
            if not r.is_error and r.tool and terminal_result is None:
                if getattr(r.tool, "terminal", False) or (
                    self.config.stop_condition
                    and self.config.stop_condition(r.tool_call.tool_name, r.result)
                ):
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

            policy_error = await self._acheck_policy(tool_name, parameters, run_id or "")
            if policy_error:
                return _Result(tc, policy_error, True, 0.0, tool, 0)

            coherence_error = await self._acheck_coherence(
                user_text_for_coherence, tool_name, parameters
            )
            if coherence_error:
                return _Result(tc, coherence_error, True, 0.0, tool, 0)

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
                result = await self._aexecute_tool_with_timeout(tool, parameters, chunk_cb)
                result = self._screen_tool_result(tool_name, result)
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

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls_to_execute])

        last_tool_name: Optional[str] = None
        last_tool_args: Dict[str, Any] = {}
        terminal_result: Optional[str] = None

        for r in results:
            all_tool_calls.append(r.tool_call)
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

            if trace is not None:
                step_type = StepType.ERROR if r.is_error else StepType.TOOL_EXECUTION
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
                self._append_tool_result(
                    r.result,
                    r.tool_call.tool_name,
                    r.tool_call.id,
                    tool_result=r.result,
                    run_id=run_id,
                )

            # Check terminal tool
            if not r.is_error and r.tool and terminal_result is None:
                if getattr(r.tool, "terminal", False) or (
                    self.config.stop_condition
                    and self.config.stop_condition(r.tool_call.tool_name, r.result)
                ):
                    terminal_result = r.result

        return last_tool_name, last_tool_args, terminal_result

    def _execute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Run tool.execute with optional timeout and chunk callback."""
        if not self.config.tool_timeout_seconds:
            return tool.execute(parameters, chunk_callback=chunk_callback)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.execute, parameters, chunk_callback)
        try:
            return future.result(timeout=self.config.tool_timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError(
                f"Tool '{tool.name}' timed out after {self.config.tool_timeout_seconds} seconds"
            )
        finally:
            executor.shutdown(wait=False)

    async def _aexecute_tool_with_timeout(
        self, tool: Tool, parameters: dict, chunk_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Async version of _execute_tool_with_timeout."""
        if not self.config.tool_timeout_seconds:
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

        # --- Tool lookup ---
        tool = self._tools_by_name.get(tool_name)
        if not tool:
            error_message = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(self._tools_by_name.keys())}"
            )
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
        policy_error = self._check_policy(tool_name, parameters, ctx.run_id)
        if policy_error:
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
            self._append_tool_result(coherence_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=coherence_error,
                    summary=f"Coherence check failed for {tool_name}",
                )
            )
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
            if self.usage.iterations:
                self.usage.tool_usage[tool.name] = self.usage.tool_usage.get(tool.name, 0) + 1
                self.usage.tool_tokens[tool.name] = (
                    self.usage.tool_tokens.get(tool.name, 0)
                    + self.usage.iterations[-1].total_tokens
                )

            self._append_tool_result(
                result, tool_name, tool_call.id, tool_result=result, run_id=ctx.run_id
            )

            # --- Terminal tool check ---
            if getattr(tool, "terminal", False) or (
                self.config.stop_condition and self.config.stop_condition(tool_name, result)
            ):
                ctx.terminal_tool_result = result
                return True

            return False

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
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
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

        # --- Tool lookup ---
        tool = self._tools_by_name.get(tool_name)
        if not tool:
            error_message = (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(self._tools_by_name.keys())}"
            )
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
        policy_error = await self._acheck_policy(tool_name, parameters, ctx.run_id)
        if policy_error:
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
            self._append_tool_result(coherence_error, tool_name, tool_call.id, run_id=ctx.run_id)
            ctx.trace.add(
                TraceStep(
                    type=StepType.ERROR,
                    tool_name=tool_name,
                    error=coherence_error,
                    summary=f"Coherence check failed for {tool_name}",
                )
            )
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
            if self.usage.iterations:
                self.usage.tool_usage[tool.name] = self.usage.tool_usage.get(tool.name, 0) + 1
                self.usage.tool_tokens[tool.name] = (
                    self.usage.tool_tokens.get(tool.name, 0)
                    + self.usage.iterations[-1].total_tokens
                )

            self._append_tool_result(
                result, tool_name, tool_call.id, tool_result=result, run_id=ctx.run_id
            )

            # --- Terminal tool check ---
            if getattr(tool, "terminal", False) or (
                self.config.stop_condition and self.config.stop_condition(tool_name, result)
            ):
                ctx.terminal_tool_result = result
                return True

            return False

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
            self._append_tool_result(error_message, tool_name, tool_call.id, run_id=ctx.run_id)
            return False
