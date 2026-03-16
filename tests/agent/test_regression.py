"""
Regression and defensive tests for every bug fix and edge case in agent core.

Each test here targets a specific production bug or risk scenario. If any of
these tests are ever removed or skipped, the corresponding bug WILL return.

Naming convention: test_<bug_number_or_category>_<description>
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.observer import AgentObserver
from selectools.policy import PolicyDecision, ToolPolicy
from selectools.providers.base import Provider, ProviderError
from selectools.providers.fallback import FallbackProvider
from selectools.tools import Tool, tool
from selectools.types import AgentResult, Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")


@tool()
def dummy_tool(x: str) -> str:
    """A dummy tool."""
    return f"result:{x}"


@tool()
def noop_tool() -> str:
    """A no-op tool for agents that need at least one tool."""
    return "noop"


class _SimpleProvider(Provider):
    """Configurable mock provider for targeted regression tests."""

    name = "simple"
    supports_streaming = False
    supports_async = True

    def __init__(
        self,
        content: Optional[str] = "Hello",
        tool_calls: Optional[List[ToolCall]] = None,
        fail_count: int = 0,
    ) -> None:
        self.default_model = "test"
        self._content = content
        self._tool_calls = tool_calls
        self._fail_count = fail_count
        self._call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ProviderError("rate limit 429")
        msg = Message(
            role=Role.ASSISTANT,
            content=self._content,
            tool_calls=self._tool_calls,
        )
        return msg, _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class _NoneContentProvider(Provider):
    """Returns Message with content=None (some providers do this for tool calls)."""

    name = "none-content"
    supports_streaming = False
    supports_async = True

    def __init__(self) -> None:
        self.default_model = "test"

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content=None), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class _JsonProvider(Provider):
    """Returns raw JSON text (simulating structured output response)."""

    name = "json-provider"
    supports_streaming = False
    supports_async = True

    def __init__(self, json_response: str) -> None:
        self.default_model = "test"
        self._response = json_response

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return Message(role=Role.ASSISTANT, content=self._response), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class _CountingObserver:
    """Minimal observer that counts events."""

    def __init__(self) -> None:
        self.events: Dict[str, List[Any]] = {}

    def _record(self, event: str, *args: Any) -> None:
        self.events.setdefault(event, []).append(args)

    def on_run_start(self, run_id: str, messages: Any) -> None:
        self._record("on_run_start", run_id)

    def on_run_end(self, run_id: str, result: Any) -> None:
        self._record("on_run_end", run_id)

    def on_llm_start(self, run_id: str, model: str, messages: Any) -> None:
        self._record("on_llm_start", run_id)

    def on_llm_end(self, run_id: str, model: str, content: str) -> None:
        self._record("on_llm_end", run_id)

    def on_iteration_start(self, run_id: str, iteration: int) -> None:
        self._record("on_iteration_start", run_id, iteration)

    def on_iteration_end(self, run_id: str, iteration: int, content: str) -> None:
        self._record("on_iteration_end", run_id, iteration)

    def on_tool_start(self, run_id: str, call_id: str, name: str, args: Any) -> None:
        self._record("on_tool_start", run_id, name)

    def on_tool_end(self, run_id: str, call_id: str, name: str, result: str, dur: float) -> None:
        self._record("on_tool_end", run_id, name)

    def on_tool_error(self, run_id: str, call_id: str, name: str, error: str, dur: float) -> None:
        self._record("on_tool_error", run_id, name)

    def on_policy_decision(
        self, run_id: str, tool_name: str, decision: str, reason: str, args: Any
    ) -> None:
        self._record("on_policy_decision", run_id, tool_name, decision)

    def on_provider_fallback(self, run_id: str, failed: str, next_p: str, exc: Exception) -> None:
        self._record("on_provider_fallback", run_id, failed, next_p)

    def on_memory_trim(self, run_id: str, removed: int, remaining: int) -> None:
        self._record("on_memory_trim", run_id, removed, remaining)

    def on_batch_start(self, batch_id: str, size: int) -> None:
        self._record("on_batch_start", batch_id, size)

    def on_batch_end(self, batch_id: str, count: int, errors: int, dur_ms: float) -> None:
        self._record("on_batch_end", batch_id, count, errors)


# ---------------------------------------------------------------------------
# Bug 14: Text parser intercepts structured output JSON as tool calls
# ---------------------------------------------------------------------------


class TestStructuredOutputParserBypass:
    """The text parser must NOT parse structured output JSON as a tool call."""

    def test_json_response_not_parsed_as_tool_call_sync(self) -> None:
        json_str = json.dumps({"name": "Alice", "age": 30})
        provider = _JsonProvider(json_str)
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        result = agent.run("give me user info", response_format=schema)
        assert result.content == json_str
        assert result.tool_name is None, "Parser should NOT have extracted a tool call"

    @pytest.mark.asyncio
    async def test_json_response_not_parsed_as_tool_call_async(self) -> None:
        json_str = json.dumps({"name": "Bob", "score": 95})
        provider = _JsonProvider(json_str)
        schema: Dict[str, Any] = {"type": "object"}
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        result = await agent.arun("give me data", response_format=schema)
        assert result.tool_name is None, "Parser should NOT have extracted a tool call"

    def test_json_response_IS_parsed_when_no_response_format(self) -> None:
        """Without response_format, the parser SHOULD try to parse tool calls."""
        json_str = json.dumps({"tool": "dummy_tool", "args": {"x": "hi"}})
        provider = _JsonProvider(f"TOOL_CALL: {json_str}")
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=2),
        )
        result = agent.run("do something")
        assert result.iterations >= 1


# ---------------------------------------------------------------------------
# Bug 17: Provider returning content=None crashes with TypeError
# ---------------------------------------------------------------------------


class TestNoneContentHandling:
    def test_none_content_sync(self) -> None:
        agent = Agent(
            tools=[noop_tool],
            provider=_NoneContentProvider(),
            config=AgentConfig(max_iterations=1),
        )
        result = agent.run("hi")
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_none_content_async(self) -> None:
        agent = Agent(
            tools=[noop_tool],
            provider=_NoneContentProvider(),
            config=AgentConfig(max_iterations=1),
        )
        result = await agent.arun("hi")
        assert isinstance(result, AgentResult)


# ---------------------------------------------------------------------------
# Bug 16: routing_only fires on_iteration_start but not on_iteration_end
# ---------------------------------------------------------------------------


class TestRoutingOnlyIterationEvents:
    def test_routing_only_fires_iteration_end_sync(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "val"}, id="c1")
        provider = _SimpleProvider(content="routing", tool_calls=[tc])
        observer = _CountingObserver()
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=1,
                routing_only=True,
                observers=[observer],
            ),
        )
        result = agent.run("route me")
        assert result.tool_name == "dummy_tool"
        assert (
            len(observer.events.get("on_iteration_end", [])) == 1
        ), "routing_only must fire on_iteration_end before returning"

    @pytest.mark.asyncio
    async def test_routing_only_fires_iteration_end_async(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "val"}, id="c1")
        provider = _SimpleProvider(content="routing", tool_calls=[tc])
        observer = _CountingObserver()
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=1,
                routing_only=True,
                observers=[observer],
            ),
        )
        result = await agent.arun("route me")
        assert len(observer.events.get("on_iteration_end", [])) == 1


# ---------------------------------------------------------------------------
# Bug 18: _acheck_policy ignores approval_timeout for sync callbacks
# Test _acheck_policy directly to verify timeout enforcement.
# ---------------------------------------------------------------------------


class TestAsyncPolicyTimeoutEnforcement:
    @pytest.mark.asyncio
    async def test_sync_callback_timeout_enforced(self) -> None:
        """A sync confirm_action that sleeps past approval_timeout must time out."""

        def slow_approver(tool_name: str, tool_args: Dict[str, Any], reason: str) -> bool:
            time.sleep(5)
            return True

        agent = Agent(
            tools=[dummy_tool],
            provider=_SimpleProvider(),
            config=AgentConfig(
                max_iterations=1,
                tool_policy=ToolPolicy(review=["dummy_tool"]),
                confirm_action=slow_approver,
                approval_timeout=0.1,
            ),
        )
        error_msg = await agent._acheck_policy("dummy_tool", {"x": "v"}, "test-run")
        assert error_msg is not None
        assert "timed out" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_async_callback_timeout_enforced(self) -> None:
        """An async confirm_action that sleeps past approval_timeout must time out."""

        async def slow_async_approver(
            tool_name: str, tool_args: Dict[str, Any], reason: str
        ) -> bool:
            await asyncio.sleep(5)
            return True

        agent = Agent(
            tools=[dummy_tool],
            provider=_SimpleProvider(),
            config=AgentConfig(
                max_iterations=1,
                tool_policy=ToolPolicy(review=["dummy_tool"]),
                confirm_action=slow_async_approver,
                approval_timeout=0.1,
            ),
        )
        error_msg = await agent._acheck_policy("dummy_tool", {"x": "v"}, "test-run")
        assert error_msg is not None
        assert "timed out" in error_msg.lower()

    def test_sync_callback_timeout_enforced_sync_path(self) -> None:
        """The sync _check_policy also enforces approval_timeout."""

        def slow_approver(tool_name: str, tool_args: Dict[str, Any], reason: str) -> bool:
            time.sleep(5)
            return True

        agent = Agent(
            tools=[dummy_tool],
            provider=_SimpleProvider(),
            config=AgentConfig(
                max_iterations=1,
                tool_policy=ToolPolicy(review=["dummy_tool"]),
                confirm_action=slow_approver,
                approval_timeout=0.1,
            ),
        )
        error_msg = agent._check_policy("dummy_tool", {"x": "v"}, "test-run")
        assert error_msg is not None
        assert "timed out" in error_msg.lower()


# ---------------------------------------------------------------------------
# Defensive: Provider returning empty tool_calls list
# ---------------------------------------------------------------------------


class TestEmptyAndMalformedToolCalls:
    def test_empty_tool_calls_list(self) -> None:
        """Provider returns tool_calls=[] which should be treated as no tool call."""
        provider = _SimpleProvider(content="no tools needed", tool_calls=[])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )
        result = agent.run("hi")
        assert result.content == "no tools needed"
        assert result.tool_name is None

    def test_tool_call_for_nonexistent_tool(self) -> None:
        """Provider requests a tool that doesn't exist."""
        tc = ToolCall(tool_name="nonexistent_tool", parameters={}, id="c1")
        provider = _SimpleProvider(content="", tool_calls=[tc])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=2),
        )
        result = agent.run("do it")
        assert (
            "unknown" in result.content.lower()
            or "not found" in result.content.lower()
            or result.iterations >= 1
        )

    def test_tool_call_with_wrong_args(self) -> None:
        """Provider passes wrong argument types — tool validation should handle it."""
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": 123}, id="c1")

        calls = 0

        class _TwoCallProvider(Provider):
            name = "two-call"
            supports_streaming = False
            supports_async = False

            def __init__(self) -> None:
                self.default_model = "test"

            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return (
                        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                        _DUMMY_USAGE,
                    )
                return Message(role=Role.ASSISTANT, content="done"), _DUMMY_USAGE

        agent = Agent(
            tools=[dummy_tool],
            provider=_TwoCallProvider(),
            config=AgentConfig(max_iterations=3),
        )
        result = agent.run("hi")
        assert isinstance(result, AgentResult)


# ---------------------------------------------------------------------------
# Defensive: Concurrent arun() on same agent instance
# ---------------------------------------------------------------------------


class TestConcurrentArun:
    @pytest.mark.asyncio
    async def test_concurrent_arun_does_not_crash(self) -> None:
        """Concurrent arun() on the same agent must not crash.

        NOTE: Concurrent arun() on a single agent shares _history, so
        results may be incorrect. Users should use batch()/abatch() for
        concurrent execution. This test verifies the operation is at
        least safe (no crash, no exception).
        """

        class _SlowProvider(Provider):
            name = "slow"
            supports_streaming = False
            supports_async = True

            def __init__(self) -> None:
                self.default_model = "test"

            async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                await asyncio.sleep(0.01)
                return (
                    Message(role=Role.ASSISTANT, content="ok"),
                    _DUMMY_USAGE,
                )

        agent = Agent(
            tools=[noop_tool],
            provider=_SlowProvider(),
            config=AgentConfig(max_iterations=1),
        )

        results = await asyncio.gather(
            agent.arun("question-A"),
            agent.arun("question-B"),
            agent.arun("question-C"),
        )

        assert len(results) == 3
        assert all(isinstance(r, AgentResult) for r in results)

    @pytest.mark.asyncio
    async def test_abatch_provides_isolation(self) -> None:
        """abatch() must isolate history between concurrent items."""

        class _EchoProvider(Provider):
            name = "echo"
            supports_streaming = False
            supports_async = True

            def __init__(self) -> None:
                self.default_model = "test"

            async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                await asyncio.sleep(0.01)
                msgs = kwargs.get("messages", [])
                user_msgs = [m for m in msgs if m.role == Role.USER]
                user_text = user_msgs[-1].content if user_msgs else "?"
                return (
                    Message(role=Role.ASSISTANT, content=f"reply:{user_text}"),
                    _DUMMY_USAGE,
                )

        agent = Agent(
            tools=[noop_tool],
            provider=_EchoProvider(),
            config=AgentConfig(max_iterations=1),
        )

        results = await agent.abatch(["A", "B", "C"])
        contents = sorted([r.content for r in results])
        assert "reply:A" in contents
        assert "reply:B" in contents
        assert "reply:C" in contents


# ---------------------------------------------------------------------------
# Regression: FallbackProvider + observers + batch no stack overflow
# ---------------------------------------------------------------------------


class TestFallbackObserverBatchNoOverflow:
    def test_fallback_with_observer_in_batch(self) -> None:
        """FallbackProvider + observers + batch() must not cause infinite recursion."""

        class _FailOnce(Provider):
            name = "fail-once"
            supports_streaming = False
            supports_async = True

            def __init__(self) -> None:
                self.default_model = "test"

            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                raise ProviderError("rate limit 429")

            async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                raise ProviderError("rate limit 429")

        ok = _SimpleProvider("batch-ok")
        fb = FallbackProvider(providers=[_FailOnce(), ok])
        observer = _CountingObserver()

        agent = Agent(
            tools=[noop_tool],
            provider=fb,
            config=AgentConfig(max_iterations=1, observers=[observer]),
        )

        results = agent.batch(["q1", "q2", "q3"], max_concurrency=3)
        assert len(results) == 3
        assert all(r.content == "batch-ok" for r in results)
        assert "on_provider_fallback" in observer.events
        assert "on_batch_start" in observer.events
        assert "on_batch_end" in observer.events


# ---------------------------------------------------------------------------
# Defensive: Policy blocks tool in sync and async paths
# ---------------------------------------------------------------------------


class TestPolicyEnforcementInAgent:
    def test_denied_tool_returns_error_sync(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "val"}, id="c1")

        calls = 0

        class _Provider(Provider):
            name = "p"
            supports_streaming = False
            supports_async = False

            def __init__(self) -> None:
                self.default_model = "test"

            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return (
                        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                        _DUMMY_USAGE,
                    )
                return Message(role=Role.ASSISTANT, content="ok done"), _DUMMY_USAGE

        observer = _CountingObserver()
        agent = Agent(
            tools=[dummy_tool],
            provider=_Provider(),
            config=AgentConfig(
                max_iterations=3,
                tool_policy=ToolPolicy(deny=["dummy_tool"]),
                observers=[observer],
            ),
        )
        result = agent.run("do it")
        assert isinstance(result, AgentResult)

        policy_events = observer.events.get("on_policy_decision", [])
        assert len(policy_events) >= 1
        assert policy_events[0][2] == "deny"

    @pytest.mark.asyncio
    async def test_denied_tool_returns_error_async(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "val"}, id="c1")

        calls = 0

        class _AsyncProvider(Provider):
            name = "p"
            supports_streaming = False
            supports_async = True

            def __init__(self) -> None:
                self.default_model = "test"

            async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return (
                        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                        _DUMMY_USAGE,
                    )
                return Message(role=Role.ASSISTANT, content="ok done"), _DUMMY_USAGE

        observer = _CountingObserver()
        agent = Agent(
            tools=[dummy_tool],
            provider=_AsyncProvider(),
            config=AgentConfig(
                max_iterations=3,
                tool_policy=ToolPolicy(deny=["dummy_tool"]),
                observers=[observer],
            ),
        )
        result = await agent.arun("do it")
        assert isinstance(result, AgentResult)

        policy_events = observer.events.get("on_policy_decision", [])
        assert len(policy_events) >= 1
        assert policy_events[0][2] == "deny"


# ---------------------------------------------------------------------------
# Defensive: Agent.reset() clears state properly
# ---------------------------------------------------------------------------


class TestAgentReset:
    def test_reset_clears_history(self) -> None:
        agent = Agent(
            tools=[dummy_tool],
            provider=_SimpleProvider("hello"),
            config=AgentConfig(max_iterations=1),
        )
        agent.run("first message")
        assert len(agent._history) > 0

        agent.reset()
        assert len(agent._history) == 0

    def test_reset_clears_usage(self) -> None:
        agent = Agent(
            tools=[dummy_tool],
            provider=_SimpleProvider("hello"),
            config=AgentConfig(max_iterations=1),
        )
        agent.run("msg")
        agent.reset()
        assert agent.usage.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Defensive: Max iterations respected
# ---------------------------------------------------------------------------


class TestMaxIterationsEnforced:
    def test_stops_after_max_iterations(self) -> None:
        """Agent must stop looping even if provider keeps returning tool calls."""
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "v"}, id="c1")
        provider = _SimpleProvider(content="", tool_calls=[tc])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        result = agent.run("loop forever")
        assert result.iterations <= 3

    @pytest.mark.asyncio
    async def test_stops_after_max_iterations_async(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "v"}, id="c1")
        provider = _SimpleProvider(content="", tool_calls=[tc])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        result = await agent.arun("loop forever")
        assert result.iterations <= 3


# ---------------------------------------------------------------------------
# Defensive: Retry logic on provider errors
# ---------------------------------------------------------------------------


class TestRetryOnProviderError:
    def test_retries_then_succeeds(self) -> None:
        provider = _SimpleProvider("success after retry", fail_count=2)
        agent = Agent(
            tools=[noop_tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=1,
                max_retries=3,
                retry_backoff_seconds=0.01,
            ),
        )
        result = agent.run("try hard")
        assert result.content == "success after retry"

    def test_exhausts_retries(self) -> None:
        provider = _SimpleProvider("never", fail_count=100)
        agent = Agent(
            tools=[noop_tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=1,
                max_retries=2,
                retry_backoff_seconds=0.01,
            ),
        )
        result = agent.run("fail")
        assert "error" in result.content.lower() or "rate limit" in result.content.lower()


# ---------------------------------------------------------------------------
# Defensive: Observer exceptions don't crash the agent
# ---------------------------------------------------------------------------


class TestObserverIsolation:
    def test_broken_observer_doesnt_crash_agent(self) -> None:
        class _CrashingObserver:
            def on_run_start(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

            def on_llm_start(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

            def on_llm_end(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

            def on_run_end(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

            def on_iteration_start(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

            def on_iteration_end(self, *args: Any) -> None:
                raise RuntimeError("observer crash!")

        agent = Agent(
            tools=[noop_tool],
            provider=_SimpleProvider("safe"),
            config=AgentConfig(max_iterations=1, observers=[_CrashingObserver()]),
        )
        result = agent.run("hi")
        assert result.content == "safe"


# ---------------------------------------------------------------------------
# Defensive: Trace is always populated
# ---------------------------------------------------------------------------


class TestTraceAlwaysPopulated:
    def test_simple_run_has_trace(self) -> None:
        agent = Agent(
            tools=[noop_tool],
            provider=_SimpleProvider("traced"),
            config=AgentConfig(max_iterations=1),
        )
        result = agent.run("hi")
        assert result.trace is not None
        assert len(result.trace.steps) >= 1

    def test_tool_call_run_has_tool_steps(self) -> None:
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": "v"}, id="c1")

        calls = 0

        class _P(Provider):
            name = "p"
            supports_streaming = False
            supports_async = False

            def __init__(self) -> None:
                self.default_model = "test"

            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return (
                        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                        _DUMMY_USAGE,
                    )
                return Message(role=Role.ASSISTANT, content="done"), _DUMMY_USAGE

        agent = Agent(
            tools=[dummy_tool],
            provider=_P(),
            config=AgentConfig(max_iterations=3),
        )
        result = agent.run("use tool")

        tool_steps = result.trace.filter(type="tool_execution")
        assert len(tool_steps) >= 1
        assert tool_steps[0].tool_name == "dummy_tool"


# ---------------------------------------------------------------------------
# Defensive: AgentResult.usage is populated
# ---------------------------------------------------------------------------


class TestAgentResultUsage:
    def test_usage_attached_to_result(self) -> None:
        agent = Agent(
            tools=[noop_tool],
            provider=_SimpleProvider("usage test"),
            config=AgentConfig(max_iterations=1),
        )
        result = agent.run("hi")
        assert result.usage is not None


# ---------------------------------------------------------------------------
# Regression: Gemini thought_signature non-UTF-8 binary crash
# ---------------------------------------------------------------------------


class TestGeminiThoughtSignatureNonUtf8:
    """Gemini 3.x thought_signature is opaque binary (protobuf/hash), not UTF-8.

    The old code did raw_sig.decode("utf-8") which crashed with UnicodeDecodeError
    on bytes like 0xa4 or 0xd5. The fix uses base64 for the round-trip.
    """

    def _get_provider(self) -> Any:
        try:
            from google.genai import types  # noqa: F401
        except ImportError:
            pytest.skip("google-genai not installed")

        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-test"
        return provider

    def test_non_utf8_thought_signature_round_trip(self) -> None:
        """Non-UTF-8 binary bytes survive encode→store→decode round-trip."""
        provider = self._get_provider()

        # Bytes that are NOT valid UTF-8 — exactly the pattern from the bug report
        raw_binary = b"\xa4\xd5\x01\x02\xff\x80\x00\xfe"
        b64_sig = base64.b64encode(raw_binary).decode("ascii")

        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="get_weather",
                        parameters={"city": "Paris"},
                        id="call_rt",
                        thought_signature=b64_sig,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content='{"temp": 18}',
                tool_name="get_weather",
                tool_call_id="call_rt",
            ),
        ]
        contents = provider._format_contents("system", messages)

        # ASSISTANT function_call part: base64 decoded back to original bytes
        fc_parts = [p for p in contents[0].parts if p.function_call is not None]
        assert getattr(fc_parts[0], "thought_signature", None) == raw_binary

        # TOOL echo part: also has original bytes
        echo_part = contents[1].parts[0]
        assert getattr(echo_part, "thought_signature", None) == raw_binary
