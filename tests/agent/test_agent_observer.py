"""
Tests for the AgentObserver protocol and all observer events.

Covers:
- AgentObserver base class wiring into Agent
- Run-level events: on_run_start, on_run_end
- LLM events: on_llm_start, on_llm_end, on_cache_hit, on_usage
- Iteration events: on_iteration_start, on_iteration_end
- Tool events: on_tool_start, on_tool_end, on_tool_error, on_tool_chunk
- Batch events: on_batch_start, on_batch_end
- Retry events: on_llm_retry
- Fallback events: on_provider_fallback
- Memory events: on_memory_trim
- Policy events: on_policy_decision
- Structured output events: on_structured_validate
- Error events: on_error
- LoggingObserver built-in
- AgentResult.usage field
- Observer error isolation (observer exceptions don't crash the agent)
- Async execution paths (arun, astream, abatch)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

from selectools import Agent, AgentConfig, AgentResult, Message, Role, ToolCall, tool
from selectools.cache import InMemoryCache
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver, LoggingObserver
from selectools.policy import ToolPolicy
from selectools.providers.base import ProviderError
from selectools.providers.fallback import FallbackProvider
from selectools.usage import UsageStats

# =============================================================================
# Test helpers
# =============================================================================


@dataclass
class ObserverEvent:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


class RecordingObserver(AgentObserver):
    """Records all observer events for assertion."""

    def __init__(self) -> None:
        self.events: List[ObserverEvent] = []

    def _record(self, name: str, **kwargs: Any) -> None:
        self.events.append(ObserverEvent(name=name, args=kwargs))

    def get(self, name: str) -> List[ObserverEvent]:
        return [e for e in self.events if e.name == name]

    def names(self) -> List[str]:
        return [e.name for e in self.events]

    # -- Run --
    def on_run_start(self, run_id: str, messages: List[Message], system_prompt: str) -> None:
        self._record("run_start", run_id=run_id, message_count=len(messages))

    def on_run_end(self, run_id: str, result: AgentResult) -> None:
        self._record("run_end", run_id=run_id, content=result.content)

    # -- LLM --
    def on_llm_start(
        self, run_id: str, messages: List[Message], model: str, system_prompt: str
    ) -> None:
        self._record("llm_start", run_id=run_id, model=model)

    def on_llm_end(self, run_id: str, response: str, usage: Optional[UsageStats]) -> None:
        self._record("llm_end", run_id=run_id, response_len=len(response) if response else 0)

    def on_cache_hit(self, run_id: str, model: str, response: str) -> None:
        self._record("cache_hit", run_id=run_id, model=model)

    def on_usage(self, run_id: str, usage: UsageStats) -> None:
        self._record("usage", run_id=run_id, total_tokens=usage.total_tokens)

    # -- Iteration --
    def on_iteration_start(self, run_id: str, iteration: int, messages: List[Message]) -> None:
        self._record("iteration_start", run_id=run_id, iteration=iteration)

    def on_iteration_end(self, run_id: str, iteration: int, response: str) -> None:
        self._record("iteration_end", run_id=run_id, iteration=iteration)

    # -- Tool --
    def on_tool_start(
        self, run_id: str, call_id: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        self._record("tool_start", run_id=run_id, call_id=call_id, tool_name=tool_name)

    def on_tool_end(
        self, run_id: str, call_id: str, tool_name: str, result: str, duration_ms: float
    ) -> None:
        self._record(
            "tool_end", run_id=run_id, call_id=call_id, tool_name=tool_name, duration_ms=duration_ms
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
        self._record("tool_error", run_id=run_id, tool_name=tool_name, error=str(error))

    def on_tool_chunk(self, run_id: str, call_id: str, tool_name: str, chunk: str) -> None:
        self._record("tool_chunk", run_id=run_id, tool_name=tool_name, chunk=chunk)

    # -- Batch --
    def on_batch_start(self, batch_id: str, prompts_count: int) -> None:
        self._record("batch_start", batch_id=batch_id, prompts_count=prompts_count)

    def on_batch_end(
        self, batch_id: str, results_count: int, errors_count: int, total_duration_ms: float
    ) -> None:
        self._record(
            "batch_end", batch_id=batch_id, results_count=results_count, errors_count=errors_count
        )

    # -- Retry / Fallback --
    def on_llm_retry(
        self, run_id: str, attempt: int, max_retries: int, error: Exception, backoff_seconds: float
    ) -> None:
        self._record("llm_retry", run_id=run_id, attempt=attempt, max_retries=max_retries)

    def on_provider_fallback(
        self, run_id: str, failed_provider: str, next_provider: str, error: Exception
    ) -> None:
        self._record(
            "provider_fallback",
            run_id=run_id,
            failed_provider=failed_provider,
            next_provider=next_provider,
        )

    # -- Memory --
    def on_memory_trim(
        self, run_id: str, messages_removed: int, messages_remaining: int, reason: str
    ) -> None:
        self._record(
            "memory_trim",
            run_id=run_id,
            messages_removed=messages_removed,
            messages_remaining=messages_remaining,
            reason=reason,
        )

    # -- Policy --
    def on_policy_decision(
        self, run_id: str, tool_name: str, decision: str, reason: str, tool_args: Dict[str, Any]
    ) -> None:
        self._record("policy_decision", run_id=run_id, tool_name=tool_name, decision=decision)

    # -- Structured --
    def on_structured_validate(
        self, run_id: str, success: bool, attempt: int, error: Optional[str] = None
    ) -> None:
        self._record("structured_validate", run_id=run_id, success=success, attempt=attempt)

    # -- Error --
    def on_error(self, run_id: str, error: Exception, context: Dict[str, Any]) -> None:
        self._record("error", run_id=run_id, error=str(error))


class SimpleProvider:
    """Returns a simple text response."""

    name = "simple"
    supports_streaming = False

    def complete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        msg = Message(role=Role.ASSISTANT, content="Hello from simple provider!")
        stats = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, model=model)
        return msg, stats

    async def acomplete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        return self.complete(
            messages=messages, model=model, system_prompt=system_prompt, tools=tools, **kwargs
        )


class ToolCallProvider:
    """Returns a tool call on first call, text on second."""

    name = "tool-caller"
    supports_streaming = False

    def __init__(
        self, tool_name: str = "greet", tool_args: Optional[Dict[str, Any]] = None
    ) -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args or {"name": "World"}
        self._call_count = 0

    def complete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        stats = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, model=model)
        if self._call_count == 1:
            msg = Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(tool_name=self._tool_name, parameters=self._tool_args, id="tc1")
                ],
            )
        else:
            msg = Message(role=Role.ASSISTANT, content="Done!")
        return msg, stats

    async def acomplete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        return self.complete(
            messages=messages, model=model, system_prompt=system_prompt, tools=tools, **kwargs
        )


class FailingToolCallProvider:
    """Returns a tool call to a tool that will raise."""

    name = "fail-tool-caller"
    supports_streaming = False

    def __init__(self) -> None:
        self._call_count = 0

    def complete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        stats = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, model=model)
        if self._call_count == 1:
            msg = Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(tool_name="failing_tool", parameters={}, id="tc_fail")],
            )
        else:
            msg = Message(role=Role.ASSISTANT, content="Handled the error.")
        return msg, stats


class RetryableProvider:
    """Fails N times then succeeds."""

    name = "retryable"
    supports_streaming = False

    def __init__(self, fail_count: int = 2) -> None:
        self._calls = 0
        self._fail_count = fail_count

    def complete(
        self,
        *,
        messages: List[Message],
        model: str,
        system_prompt: str = "",
        tools: Any = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageStats]:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise ProviderError("transient error")
        msg = Message(role=Role.ASSISTANT, content="Recovered!")
        stats = UsageStats(prompt_tokens=5, completion_tokens=5, total_tokens=10, model=model)
        return msg, stats


class RetrievableFailProvider:
    """Always fails with a retriable error."""

    name = "retriable-fail"
    supports_streaming = False

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        raise ProviderError("rate limit exceeded")


@tool()
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


@tool()
def failing_tool() -> str:
    """Always fails."""
    raise ValueError("Tool exploded!")


@tool()
def noop() -> str:
    """Do nothing."""
    return "ok"


def _agent(
    provider: Any = None,
    observer: Optional[RecordingObserver] = None,
    tools: Optional[list] = None,
    **config_kw: Any,
) -> Tuple[Agent, RecordingObserver]:
    obs = observer or RecordingObserver()
    config_kw.setdefault("observers", [obs])
    return (
        Agent(
            tools=tools or [greet],
            provider=provider or SimpleProvider(),
            config=AgentConfig(**config_kw),
        ),
        obs,
    )


# =============================================================================
# Run-level events
# =============================================================================


class TestRunEvents:
    def test_run_start_and_end_fire(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        assert len(obs.get("run_start")) == 1
        assert len(obs.get("run_end")) == 1

    def test_run_start_has_run_id(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        start = obs.get("run_start")[0]
        end = obs.get("run_end")[0]
        assert start.args["run_id"] == end.args["run_id"]
        assert len(start.args["run_id"]) > 0

    def test_run_end_has_content(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        end = obs.get("run_end")[0]
        assert "Hello from simple provider" in end.args["content"]

    @pytest.mark.asyncio
    async def test_arun_fires_run_events(self) -> None:
        agent, obs = _agent()
        await agent.arun("hello")
        assert len(obs.get("run_start")) == 1
        assert len(obs.get("run_end")) == 1

    @pytest.mark.asyncio
    async def test_astream_fires_run_events(self) -> None:
        agent, obs = _agent()
        async for _ in agent.astream("hello"):
            pass
        assert len(obs.get("run_start")) == 1
        assert len(obs.get("run_end")) == 1


# =============================================================================
# LLM events
# =============================================================================


class TestLLMEvents:
    def test_llm_start_and_end_fire(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        assert len(obs.get("llm_start")) >= 1
        assert len(obs.get("llm_end")) >= 1

    def test_usage_event_fires(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        usage_events = obs.get("usage")
        assert len(usage_events) >= 1
        assert usage_events[0].args["total_tokens"] == 15

    def test_cache_hit_fires_on_second_call(self) -> None:
        cache = InMemoryCache(max_size=10, default_ttl=60)
        obs = RecordingObserver()
        agent1 = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(cache=cache, observers=[obs]),
        )
        agent1.run("hello")
        agent2 = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(cache=cache, observers=[obs]),
        )
        agent2.run("hello")
        cache_hits = obs.get("cache_hit")
        assert len(cache_hits) >= 1

    @pytest.mark.asyncio
    async def test_async_usage_event_fires(self) -> None:
        agent, obs = _agent()
        await agent.arun("hello")
        usage_events = obs.get("usage")
        assert len(usage_events) >= 1


# =============================================================================
# Iteration events
# =============================================================================


class TestIterationEvents:
    def test_iteration_start_and_end_fire(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        starts = obs.get("iteration_start")
        ends = obs.get("iteration_end")
        assert len(starts) >= 1
        assert len(ends) >= 1
        assert starts[0].args["iteration"] == 1

    def test_multi_iteration_with_tool_call(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        agent.run("greet someone")
        starts = obs.get("iteration_start")
        ends = obs.get("iteration_end")
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].args["iteration"] == 1
        assert starts[1].args["iteration"] == 2

    @pytest.mark.asyncio
    async def test_arun_iteration_events(self) -> None:
        agent, obs = _agent()
        await agent.arun("hello")
        assert len(obs.get("iteration_start")) >= 1
        assert len(obs.get("iteration_end")) >= 1

    @pytest.mark.asyncio
    async def test_astream_iteration_events(self) -> None:
        agent, obs = _agent()
        async for _ in agent.astream("hello"):
            pass
        assert len(obs.get("iteration_start")) >= 1
        assert len(obs.get("iteration_end")) >= 1


# =============================================================================
# Tool events
# =============================================================================


class TestToolEvents:
    def test_tool_start_and_end_fire(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        agent.run("greet someone")
        assert len(obs.get("tool_start")) == 1
        assert len(obs.get("tool_end")) == 1
        start = obs.get("tool_start")[0]
        end = obs.get("tool_end")[0]
        assert start.args["tool_name"] == "greet"
        assert end.args["tool_name"] == "greet"
        assert end.args["duration_ms"] >= 0

    def test_tool_start_end_share_call_id(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        agent.run("greet someone")
        start = obs.get("tool_start")[0]
        end = obs.get("tool_end")[0]
        assert start.args["call_id"] == end.args["call_id"]

    def test_tool_error_fires_on_failure(self) -> None:
        agent, obs = _agent(
            provider=FailingToolCallProvider(),
            tools=[failing_tool],
        )
        agent.run("do something")
        errors = obs.get("tool_error")
        assert len(errors) == 1
        assert "Tool exploded" in errors[0].args["error"]

    @pytest.mark.asyncio
    async def test_arun_tool_events(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        await agent.arun("greet someone")
        assert len(obs.get("tool_start")) >= 1
        assert len(obs.get("tool_end")) >= 1


# =============================================================================
# Batch events
# =============================================================================


class TestBatchEvents:
    def test_batch_start_and_end_fire(self) -> None:
        agent, obs = _agent()
        agent.batch(["q1", "q2", "q3"])
        starts = obs.get("batch_start")
        ends = obs.get("batch_end")
        assert len(starts) == 1
        assert len(ends) == 1
        assert starts[0].args["prompts_count"] == 3
        assert ends[0].args["results_count"] == 3
        assert ends[0].args["errors_count"] == 0

    def test_batch_start_end_share_batch_id(self) -> None:
        agent, obs = _agent()
        agent.batch(["q1", "q2"])
        start = obs.get("batch_start")[0]
        end = obs.get("batch_end")[0]
        assert start.args["batch_id"] == end.args["batch_id"]
        assert len(start.args["batch_id"]) > 0

    @pytest.mark.asyncio
    async def test_abatch_fires_batch_events(self) -> None:
        agent, obs = _agent()
        await agent.abatch(["q1", "q2"])
        starts = obs.get("batch_start")
        ends = obs.get("batch_end")
        assert len(starts) == 1
        assert len(ends) == 1
        assert starts[0].args["prompts_count"] == 2

    def test_batch_also_fires_per_item_run_events(self) -> None:
        agent, obs = _agent()
        agent.batch(["q1", "q2"])
        assert len(obs.get("run_start")) == 2
        assert len(obs.get("run_end")) == 2


# =============================================================================
# LLM retry events
# =============================================================================


class TestLLMRetryEvents:
    def test_retry_events_fire_on_provider_error(self) -> None:
        agent, obs = _agent(
            provider=RetryableProvider(fail_count=2),
            max_retries=3,
            retry_backoff_seconds=0.01,
        )
        agent.run("test")
        retries = obs.get("llm_retry")
        assert len(retries) == 2
        assert retries[0].args["attempt"] == 1
        assert retries[1].args["attempt"] == 2
        assert retries[0].args["max_retries"] == 3

    def test_no_retry_events_on_success(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        assert len(obs.get("llm_retry")) == 0


# =============================================================================
# Provider fallback events
# =============================================================================


class TestProviderFallbackEvents:
    def test_fallback_event_fires_when_provider_fails(self) -> None:
        fb = FallbackProvider(providers=[RetrievableFailProvider(), SimpleProvider()])
        agent, obs = _agent(provider=fb)
        agent.run("test")
        fallbacks = obs.get("provider_fallback")
        assert len(fallbacks) >= 1
        assert fallbacks[0].args["failed_provider"] == "retriable-fail"
        assert fallbacks[0].args["next_provider"] == "simple"

    def test_no_fallback_event_when_first_provider_works(self) -> None:
        fb = FallbackProvider(providers=[SimpleProvider(), RetrievableFailProvider()])
        agent, obs = _agent(provider=fb)
        agent.run("test")
        assert len(obs.get("provider_fallback")) == 0

    def test_fallback_observer_unwired_after_run(self) -> None:
        fb = FallbackProvider(providers=[RetrievableFailProvider(), SimpleProvider()])
        agent, obs = _agent(provider=fb)
        agent.run("test")
        assert not hasattr(fb, "_original_on_fallback")


# =============================================================================
# Memory trim events
# =============================================================================


class TestMemoryTrimEvents:
    def test_memory_trim_fires_when_limit_exceeded(self) -> None:
        mem = ConversationMemory(max_messages=3)
        obs = RecordingObserver()
        agent = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(observers=[obs]),
            memory=mem,
        )
        mem.add(Message(role=Role.USER, content="m1"))
        mem.add(Message(role=Role.USER, content="m2"))
        mem.add(Message(role=Role.USER, content="m3"))
        agent.run("m4")
        trims = obs.get("memory_trim")
        assert len(trims) >= 1
        assert trims[0].args["messages_removed"] > 0
        assert trims[0].args["reason"] == "enforce_limits"

    def test_no_trim_event_when_memory_not_full(self) -> None:
        mem = ConversationMemory(max_messages=100)
        obs = RecordingObserver()
        agent = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(observers=[obs]),
            memory=mem,
        )
        agent.run("hello")
        assert len(obs.get("memory_trim")) == 0


# =============================================================================
# Policy decision events
# =============================================================================


class TestPolicyDecisionEvents:
    def test_policy_decision_fires_for_allowed_tool(self) -> None:
        policy = ToolPolicy(allow=["greet"])
        agent, obs = _agent(
            provider=ToolCallProvider(),
            tool_policy=policy,
        )
        agent.run("greet someone")
        decisions = obs.get("policy_decision")
        assert len(decisions) >= 1
        assert decisions[0].args["tool_name"] == "greet"
        assert decisions[0].args["decision"] == "allow"

    def test_policy_decision_fires_for_denied_tool(self) -> None:
        policy = ToolPolicy(deny=["greet"])
        agent, obs = _agent(
            provider=ToolCallProvider(),
            tool_policy=policy,
        )
        agent.run("greet someone")
        decisions = obs.get("policy_decision")
        assert len(decisions) >= 1
        assert decisions[0].args["decision"] == "deny"


# =============================================================================
# Structured validate events
# =============================================================================


class TestStructuredValidateEvents:
    def test_structured_validate_fires_on_success(self) -> None:
        from pydantic import BaseModel

        class Response(BaseModel):
            answer: str

        class JSONProvider:
            name = "json-provider"
            supports_streaming = False

            def complete(
                self, *, messages: List[Message], model: str, **kwargs: Any
            ) -> Tuple[Message, UsageStats]:
                msg = Message(role=Role.ASSISTANT, content='{"answer": "42"}')
                stats = UsageStats(
                    prompt_tokens=5, completion_tokens=5, total_tokens=10, model=model
                )
                return msg, stats

        agent, obs = _agent(provider=JSONProvider())
        result = agent.run("what is the answer?", response_format=Response)
        validates = obs.get("structured_validate")
        assert len(validates) >= 1
        assert validates[0].args["success"] is True
        assert validates[0].args["attempt"] == 1


# =============================================================================
# Error events
# =============================================================================


class TestErrorEvents:
    def test_error_event_fires_on_unrecoverable_error(self) -> None:
        class CrashProvider:
            name = "crash"
            supports_streaming = False

            def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
                raise RuntimeError("Unexpected crash!")

        agent, obs = _agent(provider=CrashProvider(), max_retries=0)
        with pytest.raises(RuntimeError):
            agent.run("test")
        errors = obs.get("error")
        assert len(errors) == 1
        assert "Unexpected crash" in errors[0].args["error"]


# =============================================================================
# Observer error isolation
# =============================================================================


class TestObserverErrorIsolation:
    def test_observer_exception_does_not_crash_agent(self) -> None:
        class BrokenObserver(AgentObserver):
            def on_run_start(
                self, run_id: str, messages: List[Message], system_prompt: str
            ) -> None:
                raise RuntimeError("Observer broke!")

            def on_llm_start(
                self, run_id: str, messages: List[Message], model: str, system_prompt: str
            ) -> None:
                raise RuntimeError("Observer broke!")

            def on_iteration_start(
                self, run_id: str, iteration: int, messages: List[Message]
            ) -> None:
                raise RuntimeError("Observer broke!")

        agent = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(observers=[BrokenObserver()]),
        )
        result = agent.run("hello")
        assert result.content is not None


# =============================================================================
# AgentResult.usage field
# =============================================================================


class TestAgentResultUsage:
    def test_result_has_usage(self) -> None:
        agent, obs = _agent()
        result = agent.run("hello")
        assert result.usage is not None
        assert result.usage.total_tokens > 0

    def test_result_usage_is_snapshot(self) -> None:
        agent, _ = _agent()
        r1 = agent.run("hello")
        r2 = agent.run("hello again")
        assert r1.usage is not r2.usage
        assert r2.usage.total_tokens >= r1.usage.total_tokens

    @pytest.mark.asyncio
    async def test_arun_result_has_usage(self) -> None:
        agent, obs = _agent()
        result = await agent.arun("hello")
        assert result.usage is not None
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_astream_result_has_usage(self) -> None:
        agent, obs = _agent()
        result = None
        async for item in agent.astream("hello"):
            if isinstance(item, AgentResult):
                result = item
        assert result is not None
        assert result.usage is not None


# =============================================================================
# LoggingObserver
# =============================================================================


class TestLoggingObserver:
    def test_logging_observer_emits_json(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="selectools.observer"):
            lo = LoggingObserver()
            lo.on_run_start("test-run", [], "sys")
            lo.on_run_end(
                "test-run", AgentResult(message=Message(role=Role.ASSISTANT, content="hi"))
            )
        records = [r for r in caplog.records if r.name == "selectools.observer"]
        assert len(records) >= 2
        parsed = json.loads(records[0].message)
        assert parsed["event"] == "run_start"
        assert parsed["run_id"] == "test-run"

    def test_logging_observer_all_new_events_callable(self) -> None:
        lo = LoggingObserver(level=logging.DEBUG)
        lo.on_iteration_start("r", 1, [])
        lo.on_iteration_end("r", 1, "resp")
        lo.on_batch_start("b", 3)
        lo.on_batch_end("b", 3, 0, 100.5)
        lo.on_provider_fallback("r", "a", "b", Exception("e"))
        lo.on_llm_retry("r", 1, 3, Exception("e"), 0.5)
        lo.on_memory_trim("r", 2, 5, "enforce_limits")
        lo.on_tool_chunk("r", "c1", "tool_a", "chunk data")
        lo.on_cache_hit("r", "gpt-4", "cached response")
        lo.on_usage("r", UsageStats(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        lo.on_policy_decision("r", "tool_a", "allow", "matched rule", {})
        lo.on_structured_validate("r", True, 1)
        lo.on_error("r", Exception("err"), {})
        lo.on_prompt_compressed("r", 80_000, 5_000, 10)

    def test_logging_observer_integrates_with_agent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="selectools.observer"):
            agent = Agent(
                tools=[greet],
                provider=SimpleProvider(),
                config=AgentConfig(observers=[LoggingObserver()]),
            )
            agent.run("hello")
        records = [r for r in caplog.records if r.name == "selectools.observer"]
        event_names = [json.loads(r.message)["event"] for r in records]
        assert "run_start" in event_names
        assert "run_end" in event_names
        assert "llm_start" in event_names
        assert "llm_end" in event_names
        assert "iteration_start" in event_names
        assert "iteration_end" in event_names


# =============================================================================
# Multiple observers
# =============================================================================


class TestMultipleObservers:
    def test_multiple_observers_all_receive_events(self) -> None:
        obs1 = RecordingObserver()
        obs2 = RecordingObserver()
        agent = Agent(
            tools=[greet],
            provider=SimpleProvider(),
            config=AgentConfig(observers=[obs1, obs2]),
        )
        agent.run("hello")
        assert len(obs1.get("run_start")) == 1
        assert len(obs2.get("run_start")) == 1
        assert len(obs1.get("run_end")) == 1
        assert len(obs2.get("run_end")) == 1


# =============================================================================
# Event ordering
# =============================================================================


class TestEventOrdering:
    def test_event_order_simple_run(self) -> None:
        agent, obs = _agent()
        agent.run("hello")
        names = obs.names()
        assert names[0] == "run_start"
        assert names[-1] == "run_end"
        assert "iteration_start" in names
        assert "llm_start" in names
        assert names.index("iteration_start") < names.index("llm_start")
        assert names.index("llm_start") < names.index("llm_end")
        assert names.index("llm_end") < names.index("iteration_end")

    def test_event_order_with_tool_call(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        agent.run("greet someone")
        names = obs.names()
        assert names[0] == "run_start"
        assert names[-1] == "run_end"
        assert "tool_start" in names
        assert "tool_end" in names
        assert names.index("tool_start") < names.index("tool_end")


# =============================================================================
# Run ID correlation
# =============================================================================


class TestRunIdCorrelation:
    def test_all_events_in_a_run_share_run_id(self) -> None:
        agent, obs = _agent(provider=ToolCallProvider())
        agent.run("greet someone")
        run_id = obs.get("run_start")[0].args["run_id"]
        for event in obs.events:
            if "run_id" in event.args:
                assert event.args["run_id"] == run_id, (
                    f"Event {event.name} has run_id={event.args['run_id']}, expected {run_id}"
                )

    def test_different_runs_have_different_run_ids(self) -> None:
        agent, obs = _agent()
        agent.run("hello 1")
        agent.run("hello 2")
        starts = obs.get("run_start")
        assert len(starts) == 2
        assert starts[0].args["run_id"] != starts[1].args["run_id"]
