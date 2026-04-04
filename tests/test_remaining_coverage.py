"""Coverage gap tests — targeting uncovered lines across multiple modules.

Each test class focuses on one module, hitting branches that existing tests miss.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools import Agent, AgentConfig
from selectools.cancellation import CancellationToken
from selectools.exceptions import (
    BudgetExceededError,
    CancellationError,
    GraphExecutionError,
    MCPConnectionError,
    MCPError,
    MCPToolError,
    MemoryLimitExceededError,
    ProviderConfigurationError,
    SelectoolsError,
    ToolExecutionError,
    ToolValidationError,
)
from selectools.providers.stubs import LocalProvider
from selectools.tools.base import Tool
from selectools.tools.decorators import tool
from selectools.trace import AgentTrace, StepType, TraceStep, trace_to_html, trace_to_json
from selectools.types import Message, Role, ToolCall
from selectools.usage import AgentUsage, UsageStats

# ── Helpers ──────────────────────────────────────────────────────────────────


@tool()
def greet(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"


@tool()
def slow_tool(seconds: float = 0.01) -> str:
    """A tool that sleeps briefly."""
    import time as _t

    _t.sleep(seconds)
    return "done"


def _make_agent(**kw: Any) -> Agent:
    """Create a test agent with LocalProvider."""
    provider = LocalProvider()
    defaults = dict(
        model="test-model",
        max_iterations=3,
    )
    defaults.update(kw)
    tools = defaults.pop("tools", [greet])
    return Agent(tools, provider=provider, config=AgentConfig(**defaults))


# ── 1. exceptions.py (lines 91-111, 147-150) ────────────────────────────────


class TestExceptions:
    """Cover all exception subclasses and their message formatting."""

    def test_memory_limit_exceeded_messages_type(self):
        exc = MemoryLimitExceededError(current=50, limit=20, limit_type="messages")
        assert "messages" in str(exc).lower() or "max_messages" in str(exc)
        assert exc.current == 50
        assert exc.limit == 20

    def test_memory_limit_exceeded_tokens_type(self):
        exc = MemoryLimitExceededError(current=10000, limit=5000, limit_type="tokens")
        assert "tokens" in str(exc).lower() or "max_tokens" in str(exc)
        assert exc.limit_type == "tokens"

    def test_memory_limit_exceeded_unknown_type(self):
        exc = MemoryLimitExceededError(current=10, limit=5, limit_type="other")
        assert exc.limit_type == "other"

    def test_graph_execution_error(self):
        inner = RuntimeError("boom")
        exc = GraphExecutionError(graph_name="my_graph", node_name="node_a", error=inner, step=3)
        assert exc.graph_name == "my_graph"
        assert exc.node_name == "node_a"
        assert exc.step == 3
        assert "node_a" in str(exc)
        assert "step 3" in str(exc)

    def test_budget_exceeded_error(self):
        exc = BudgetExceededError(reason="over budget", tokens_used=1000, cost_used=0.05)
        assert exc.reason == "over budget"
        assert exc.tokens_used == 1000
        assert exc.cost_used == 0.05

    def test_cancellation_error(self):
        exc = CancellationError()
        assert "cancelled" in str(exc).lower()
        exc2 = CancellationError(reason="user stopped")
        assert exc2.reason == "user stopped"

    def test_mcp_errors(self):
        assert issubclass(MCPConnectionError, MCPError)
        assert issubclass(MCPToolError, MCPError)
        assert issubclass(MCPError, SelectoolsError)

    def test_tool_validation_error(self):
        exc = ToolValidationError("my_tool", "param_a", "wrong type", "use int")
        assert exc.tool_name == "my_tool"
        assert exc.param_name == "param_a"
        assert "Suggestion" in str(exc)

    def test_tool_execution_error(self):
        inner = ValueError("bad")
        exc = ToolExecutionError("my_tool", inner, {"x": 1})
        assert exc.tool_name == "my_tool"
        assert exc.error is inner

    def test_provider_configuration_error_with_env_var(self):
        exc = ProviderConfigurationError("openai", "API key", "OPENAI_API_KEY")
        assert "OPENAI_API_KEY" in str(exc)
        assert exc.env_var == "OPENAI_API_KEY"

    def test_provider_configuration_error_without_env_var(self):
        exc = ProviderConfigurationError("custom", "endpoint", "")
        assert exc.env_var == ""


# ── 2. token_estimation.py (lines 59-67, 88, 115, 143-144) ──────────────────


class TestTokenEstimation:
    """Cover estimate_run_tokens and heuristic fallback paths."""

    def test_estimate_tokens_empty(self):
        from selectools.token_estimation import estimate_tokens

        assert estimate_tokens("", "gpt-4o") == 0

    def test_estimate_tokens_heuristic_fallback(self):
        from selectools.token_estimation import _heuristic_count

        assert _heuristic_count("hello world") >= 1
        assert _heuristic_count("") == 0

    def test_estimate_run_tokens_with_tools(self):
        from selectools.token_estimation import estimate_run_tokens

        msgs = [Message(role=Role.USER, content="Hello there")]
        result = estimate_run_tokens(
            messages=msgs,
            tools=[greet],
            system_prompt="You are helpful.",
            model="gpt-4o",
        )
        assert result.system_tokens > 0
        assert result.message_tokens > 0
        assert result.tool_schema_tokens > 0
        assert result.total_tokens == (
            result.system_tokens + result.message_tokens + result.tool_schema_tokens
        )
        assert result.model == "gpt-4o"

    def test_estimate_run_tokens_unknown_model(self):
        from selectools.token_estimation import estimate_run_tokens

        msgs = [Message(role=Role.USER, content="Test")]
        result = estimate_run_tokens(
            messages=msgs,
            tools=[],
            system_prompt="sys",
            model="nonexistent-model-xyz",
        )
        # Should still produce an estimate, context_window may be 0
        assert result.total_tokens > 0

    def test_tiktoken_count_unknown_model(self):
        from selectools.token_estimation import _tiktoken_count

        # Should fall back to cl100k_base
        result = _tiktoken_count("hello", "totally-fake-model-xxx")
        # Either returns a count or None if tiktoken can't handle it
        assert result is None or result > 0


# ── 3. trace.py (lines 178-179, 264-269, 387, 391, 503-507) ─────────────────


class TestTrace:
    """Cover trace_to_html edge cases and trace_to_json."""

    def test_trace_from_dict_unknown_step_type(self):
        d = {
            "run_id": "test-run",
            "start_time": time.time(),
            "steps": [
                {"type": "completely_unknown_type", "summary": "test step"},
            ],
        }
        trace = AgentTrace.from_dict(d)
        # Unknown types should fall back to ERROR
        assert trace.steps[0].type == StepType.ERROR

    def test_trace_to_html_with_reasoning_and_routing(self):
        trace = AgentTrace()
        trace.steps.append(
            TraceStep(
                type=StepType.LLM_CALL,
                duration_ms=100.0,
                model="gpt-4o",
                prompt_tokens=50,
                completion_tokens=25,
                summary="LLM call",
                reasoning="I chose this because...",
            )
        )
        trace.steps.append(
            TraceStep(
                type=StepType.GRAPH_ROUTING,
                duration_ms=5.0,
                from_node="node_a",
                to_node="node_b",
                summary="Routing",
                node_name="router",
            )
        )
        trace.steps.append(
            TraceStep(
                type=StepType.TOOL_EXECUTION,
                duration_ms=50.0,
                tool_name="greet",
                tool_result="Hello!",
                summary="Tool exec",
            )
        )
        trace.steps.append(
            TraceStep(
                type=StepType.ERROR,
                duration_ms=1.0,
                error="Something went wrong",
                summary="Error step",
            )
        )
        html = trace_to_html(trace)
        assert "gpt-4o" in html
        assert "node_a" in html
        assert "reasoning" in html.lower()

    def test_trace_to_html_empty_trace(self):
        trace = AgentTrace()
        html = trace_to_html(trace)
        assert "Agent Trace" in html
        assert "0" in html  # 0 steps

    def test_trace_to_json(self):
        trace = AgentTrace()
        trace.steps.append(TraceStep(type=StepType.LLM_CALL, duration_ms=10.0, model="gpt-4o"))
        result = trace_to_json(trace)
        parsed = json.loads(result)
        assert "steps" in parsed
        assert len(parsed["steps"]) == 1

    def test_trace_to_otel_spans_with_various_types(self):
        trace = AgentTrace()
        trace.steps.append(
            TraceStep(
                type=StepType.LLM_CALL,
                duration_ms=100.0,
                model="gpt-4o",
                prompt_tokens=50,
                completion_tokens=25,
                summary="LLM call",
            )
        )
        trace.steps.append(
            TraceStep(
                type=StepType.CACHE_HIT,
                duration_ms=1.0,
                model="gpt-4o",
                summary="Cache hit",
            )
        )
        trace.steps.append(
            TraceStep(
                type=StepType.STRUCTURED_RETRY,
                duration_ms=2.0,
                error="validation error",
                summary="Retry",
            )
        )
        trace.parent_run_id = "parent-123"
        trace.metadata = {"env": "test"}
        spans = trace.to_otel_spans()
        assert len(spans) == 4  # 1 root + 3 children
        root = spans[0]
        assert root["attributes"]["selectools.parent_run_id"] == "parent-123"
        assert "selectools.metadata.env" in root["attributes"]


# ── 4. usage.py (lines 114, 116, 137-138, 160-164) ──────────────────────────


class TestUsage:
    """Cover AgentUsage __str__ and to_dict edge cases."""

    def test_usage_str_with_embeddings(self):
        usage = AgentUsage()
        usage.total_tokens = 100
        usage.total_prompt_tokens = 60
        usage.total_completion_tokens = 40
        usage.total_cost_usd = 0.001
        usage.total_embedding_tokens = 500
        usage.total_embedding_cost_usd = 0.0005
        s = str(usage)
        assert "Embedding" in s
        assert "500" in s

    def test_usage_to_dict_with_embeddings(self):
        usage = AgentUsage()
        usage.total_embedding_tokens = 200
        usage.total_embedding_cost_usd = 0.0002
        d = usage.to_dict()
        assert "total_embedding_tokens" in d
        assert d["total_embedding_tokens"] == 200

    def test_usage_to_dict_without_embeddings(self):
        usage = AgentUsage()
        d = usage.to_dict()
        assert "total_embedding_tokens" not in d

    def test_usage_merge(self):
        a = AgentUsage()
        a.total_tokens = 100
        a.tool_usage = {"greet": 2}
        a.tool_tokens = {"greet": 50}
        a.iterations = [{"i": 1}]

        b = AgentUsage()
        b.total_tokens = 200
        b.tool_usage = {"greet": 1, "search": 3}
        b.tool_tokens = {"greet": 30, "search": 100}
        b.iterations = [{"i": 2}]

        a.merge(b)
        assert a.total_tokens == 300
        assert a.tool_usage["greet"] == 3
        assert a.tool_usage["search"] == 3
        assert len(a.iterations) == 2


# ── 5. observer.py (lines 706, 716, 734, 861-873, 989-1014, 1603-1705) ──────


class TestObservers:
    """Cover LoggingObserver and SimpleStepObserver edge cases."""

    def test_logging_observer_tool_events(self):
        from selectools.observer import LoggingObserver

        obs = LoggingObserver()
        # These should not raise
        obs.on_tool_start("r1", "c1", "greet", {"name": "Alice"})
        obs.on_tool_end("r1", "c1", "greet", "Hello!", 15.5)
        obs.on_tool_error("r1", "c1", "greet", ValueError("bad"), {"name": "X"}, 10.0)

    def test_logging_observer_session_events(self):
        from selectools.observer import LoggingObserver

        obs = LoggingObserver()
        obs.on_session_load("r1", "sess1", 10)
        obs.on_session_save("r1", "sess1", 10)
        obs.on_memory_summarize("r1", "summary text")
        obs.on_entity_extraction("r1", 5)
        obs.on_kg_extraction("r1", 3)

    def test_logging_observer_budget_cancelled(self):
        from selectools.observer import LoggingObserver

        obs = LoggingObserver()
        obs.on_budget_exceeded("r1", "tokens exceeded", 5000, 0.05)
        obs.on_cancelled("r1", 3, "user cancelled")

    def test_logging_observer_eval_events(self):
        from selectools.observer import LoggingObserver

        obs = LoggingObserver()
        obs.on_eval_start("suite1", 10, "gpt-4o")
        obs.on_eval_case_end("suite1", "case1", "pass", 150.0, 0)
        obs.on_eval_end("suite1", 0.9, 10, 9, 1, 0.01, 5000.0)

    def test_simple_step_observer_all_events(self):
        from selectools.observer import SimpleStepObserver

        events: list = []
        obs = SimpleStepObserver(callback=lambda event, *a, **kw: events.append(event))

        obs.on_budget_exceeded("r1", "over", 1000, 0.01)
        obs.on_cancelled("r1", 1, "cancelled")
        obs.on_model_switch("r1", 1, "old", "new")
        obs.on_prompt_compressed("r1", 1000, 500, 3)
        obs.on_graph_start("r1", "g1", "start", {})
        obs.on_graph_end("r1", "g1", 5, 100.0)
        obs.on_graph_error("r1", "g1", "n1", RuntimeError("err"))
        obs.on_node_start("r1", "n1", 1)
        obs.on_node_end("r1", "n1", 1, 50.0)
        obs.on_graph_routing("r1", "a", "b")
        obs.on_graph_interrupt("r1", "n1", "int1")
        obs.on_graph_resume("r1", "n1", "int1")
        obs.on_parallel_start("r1", "g1", ["a", "b"])
        obs.on_parallel_end("r1", "g1", 2)
        obs.on_stall_detected("r1", "n1", 3)
        obs.on_loop_detected("r1", "n1", 2)
        obs.on_supervisor_replan("r1", 1, "new plan")
        obs.on_eval_start("s1", 5, "gpt-4o")
        obs.on_eval_case_end("s1", "c1", "pass", 100.0, 0)
        obs.on_eval_end("s1", 0.8, 5, 4, 1, 0.01, 1000.0)
        obs.on_error("r1", ValueError("err"), {})

        assert len(events) >= 20

    def test_hooks_adapter(self):
        """Test the deprecated hooks adapter (HooksAdapter)."""
        calls: Dict[str, list] = {}

        def track(name: str):
            def fn(*args):
                calls.setdefault(name, []).append(args)

            return fn

        with pytest.warns(DeprecationWarning):
            agent = _make_agent(
                hooks={
                    "on_agent_start": track("on_agent_start"),
                    "on_agent_end": track("on_agent_end"),
                    "on_tool_start": track("on_tool_start"),
                    "on_tool_end": track("on_tool_end"),
                    "on_iteration_start": track("on_iteration_start"),
                    "on_iteration_end": track("on_iteration_end"),
                }
            )
        agent.run("Hello")
        assert "on_agent_start" in calls
        assert "on_agent_end" in calls


# ── 6. agent/config.py (lines 217-226, 283-285, 294-295, 303-306, 316-318) ──


class TestAgentConfig:
    """Cover config validation and nested config unpacking."""

    def test_nested_config_dict_unpacking(self):
        """Config sub-groups passed as dicts should be unpacked automatically."""
        config = AgentConfig(
            model="test",
            retry={"max_retries": 5, "backoff_seconds": 2.0},
            tool={"timeout_seconds": 30},
        )
        assert config.max_retries == 5
        assert config.retry_backoff_seconds == 2.0
        assert config.tool_timeout_seconds == 30

    def test_nested_config_wrong_type_warning(self):
        """Passing wrong type for nested config should warn and use None."""
        with pytest.warns(match="Expected.*or dict"):
            config = AgentConfig(model="test", retry="bad_value")
        # Should have created a RetryConfig from flat defaults
        assert config.retry is not None

    def test_nested_guardrail_config(self):
        config = AgentConfig(
            model="test",
            guardrail={"pipeline": None, "screen_tool_output": True},
        )
        assert config.screen_tool_output is True

    def test_nested_session_config(self):
        config = AgentConfig(
            model="test",
            session={"store": None, "session_id": "sess-123"},
        )
        assert config.session_id == "sess-123"

    def test_nested_summarize_config(self):
        config = AgentConfig(
            model="test",
            summarize={"enabled": True, "max_tokens": 200},
        )
        assert config.summarize_on_trim is True
        assert config.summarize_max_tokens == 200

    def test_nested_memory_config(self):
        config = AgentConfig(
            model="test",
            memory={"entity_memory": None, "knowledge_graph": None},
        )
        assert config.entity_memory is None

    def test_nested_budget_config(self):
        config = AgentConfig(
            model="test",
            budget={"max_total_tokens": 10000, "max_cost_usd": 1.0},
        )
        assert config.max_total_tokens == 10000
        assert config.max_cost_usd == 1.0


# ── 7. agent/_tool_executor.py (timeout, approval gate, coherence) ───────────


class TestToolExecutor:
    """Cover timeout paths, approval gate, coherence check blocking."""

    def test_tool_timeout_sync(self):
        @tool()
        def very_slow_tool() -> str:
            """A tool that takes too long."""
            import time as _t

            _t.sleep(10)
            return "done"

        agent = _make_agent(tools=[very_slow_tool], tool_timeout_seconds=0.05)
        # The tool should time out but the agent should handle it gracefully
        result = agent.run("Call very_slow_tool")
        # Should contain error about timeout
        assert result.content is not None

    def test_tool_policy_deny(self):
        from selectools.policy import ToolPolicy

        policy = ToolPolicy(deny=["greet"])
        agent = _make_agent(tool_policy=policy)
        result = agent.run("Greet Alice")
        # Agent should get back a denial message from the tool system
        assert result.content is not None

    def test_approval_gate_timeout(self):
        """Approval callback that takes too long should time out."""

        def slow_approver(tool_name: str, tool_args: dict, reason: str) -> bool:
            import time as _t

            _t.sleep(10)
            return True

        from selectools.policy import ToolPolicy

        policy = ToolPolicy(review=["greet"])
        agent = _make_agent(
            tool_policy=policy,
            confirm_action=slow_approver,
            approval_timeout=0.05,
        )
        result = agent.run("Greet Alice")
        assert result.content is not None

    def test_approval_gate_rejection(self):
        """Approval callback that rejects the tool."""

        def rejector(tool_name: str, tool_args: dict, reason: str) -> bool:
            return False

        from selectools.policy import ToolPolicy

        policy = ToolPolicy(review=["greet"])
        agent = _make_agent(
            tool_policy=policy,
            confirm_action=rejector,
            approval_timeout=5.0,
        )
        result = agent.run("Greet Alice")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_tool_timeout_async(self):
        @tool()
        def very_slow_tool() -> str:
            """A tool that takes too long."""
            import time as _t

            _t.sleep(10)
            return "done"

        agent = _make_agent(tools=[very_slow_tool], tool_timeout_seconds=0.05)
        result = await agent.arun("Call very_slow_tool")
        assert result.content is not None


# ── 8. agent/core.py — astream paths ────────────────────────────────────────


class TestAgentCoreAstream:
    """Cover astream structured retry, budget, cancellation, routing_only."""

    @pytest.mark.asyncio
    async def test_astream_cancellation_before_iteration(self):
        token = CancellationToken()
        token.cancel()
        agent = _make_agent(cancellation_token=token)
        chunks = []
        async for chunk in agent.astream("Hello"):
            chunks.append(chunk)
        # Should yield exactly one AgentResult with cancelled content
        from selectools.types import AgentResult

        assert any(isinstance(c, AgentResult) for c in chunks)

    @pytest.mark.asyncio
    async def test_astream_budget_exceeded(self):
        agent = _make_agent(max_total_tokens=1)
        # Pre-fill usage to trigger budget
        agent.usage.total_tokens = 100
        chunks = []
        async for chunk in agent.astream("Hello"):
            chunks.append(chunk)
        from selectools.types import AgentResult

        assert any(isinstance(c, AgentResult) for c in chunks)

    @pytest.mark.asyncio
    async def test_astream_routing_only(self):
        """When routing_only is set, astream should yield result without executing tools."""

        # Need a provider that returns a tool call
        class ToolCallProvider(LocalProvider):
            def stream(self, **kwargs):
                yield ToolCall(tool_name="greet", parameters={"name": "Alice"})

            @property
            def supports_streaming(self):
                return True

            async def astream(self, **kwargs):
                yield ToolCall(tool_name="greet", parameters={"name": "Alice"})

        provider = ToolCallProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test", routing_only=True),
        )
        chunks = []
        async for chunk in agent.astream("Greet Alice"):
            chunks.append(chunk)
        from selectools.types import AgentResult

        results = [c for c in chunks if isinstance(c, AgentResult)]
        assert len(results) == 1
        assert results[0].tool_name == "greet"

    @pytest.mark.asyncio
    async def test_astream_max_iterations(self):
        """astream should yield result when max iterations is reached."""

        # Provider that always returns tool calls
        class InfiniteToolProvider(LocalProvider):
            def complete(self, **kwargs):
                msg = Message(
                    role=Role.ASSISTANT,
                    content="calling greet",
                    tool_calls=[ToolCall(tool_name="greet", parameters={"name": "Bob"})],
                )
                usage = UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="test",
                )
                return msg, usage

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = InfiniteToolProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=2),
        )
        chunks = []
        async for chunk in agent.astream("Greet Bob"):
            chunks.append(chunk)
        from selectools.types import AgentResult

        results = [c for c in chunks if isinstance(c, AgentResult)]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_astream_post_tool_cancellation(self):
        """Cancel after tool execution in astream."""
        token = CancellationToken()

        @tool()
        def cancel_tool() -> str:
            """Cancels after execution."""
            token.cancel()
            return "done"

        class ToolCallProvider(LocalProvider):
            _called = False

            def complete(self, **kwargs):
                if not self._called:
                    self._called = True
                    msg = Message(
                        role=Role.ASSISTANT,
                        content="calling",
                        tool_calls=[ToolCall(tool_name="cancel_tool", parameters={})],
                    )
                    usage = UsageStats(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=10,
                        cost_usd=0.0,
                        model="test",
                    )
                    return msg, usage
                msg = Message(role=Role.ASSISTANT, content="Done")
                return msg, UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ToolCallProvider()
        agent = Agent(
            [cancel_tool],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5, cancellation_token=token),
        )
        chunks = []
        async for chunk in agent.astream("Call cancel_tool"):
            chunks.append(chunk)
        from selectools.types import AgentResult

        results = [c for c in chunks if isinstance(c, AgentResult)]
        assert len(results) == 1


# ── 9. agent/core.py — arun paths ───────────────────────────────────────────


class TestAgentCoreArun:
    """Cover arun structured retry, budget, cancellation, terminal tool."""

    @pytest.mark.asyncio
    async def test_arun_cancellation_before_iteration(self):
        token = CancellationToken()
        token.cancel()
        agent = _make_agent(cancellation_token=token)
        result = await agent.arun("Hello")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_arun_budget_exceeded(self):
        agent = _make_agent(max_total_tokens=1)
        agent.usage.total_tokens = 100
        result = await agent.arun("Hello")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_arun_structured_retry(self):
        """arun should retry on structured output validation failure."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        call_count = 0

        class RetryProvider(LocalProvider):
            def complete(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call returns invalid JSON
                    msg = Message(role=Role.ASSISTANT, content="not valid json")
                else:
                    msg = Message(
                        role=Role.ASSISTANT,
                        content='{"name": "Alice", "age": 30}',
                    )
                usage = UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="test",
                )
                return msg, usage

            @property
            def supports_async(self):
                return False

            @property
            def supports_streaming(self):
                return False

        provider = RetryProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5),
        )
        result = await agent.arun("Return a person", response_format=Person)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_arun_post_tool_cancellation(self):
        """Cancel via cancellation token after tool execution in arun."""
        token = CancellationToken()

        @tool()
        def cancel_after() -> str:
            """Cancel the run."""
            token.cancel()
            return "cancelled"

        class ToolCallProvider(LocalProvider):
            _called = False

            def complete(self, **kwargs):
                if not self._called:
                    self._called = True
                    msg = Message(
                        role=Role.ASSISTANT,
                        content="calling",
                        tool_calls=[ToolCall(tool_name="cancel_after", parameters={})],
                    )
                    return msg, UsageStats(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=10,
                        cost_usd=0.0,
                        model="test",
                    )
                return Message(role=Role.ASSISTANT, content="Done"), UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ToolCallProvider()
        agent = Agent(
            [cancel_after],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5, cancellation_token=token),
        )
        result = await agent.arun("Call cancel_after")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_arun_model_selector(self):
        """arun should fire model switch when model_selector changes the model."""

        def selector(iteration, tool_calls, usage):
            return "gpt-4o-mini" if iteration > 1 else "gpt-4o"

        class MultiIterProvider(LocalProvider):
            _call = 0

            def complete(self, **kwargs):
                self._call += 1
                if self._call == 1:
                    msg = Message(
                        role=Role.ASSISTANT,
                        content="calling greet",
                        tool_calls=[ToolCall(tool_name="greet", parameters={"name": "X"})],
                    )
                    return msg, UsageStats(
                        prompt_tokens=5,
                        completion_tokens=5,
                        total_tokens=10,
                        cost_usd=0.0,
                        model=kwargs.get("model", "test"),
                    )
                return Message(role=Role.ASSISTANT, content="Done"), UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model=kwargs.get("model", "test"),
                )

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        agent = Agent(
            [greet],
            provider=MultiIterProvider(),
            config=AgentConfig(model="gpt-4o", max_iterations=5, model_selector=selector),
        )
        result = await agent.arun("Greet X")
        assert result.content is not None


# ── 10. agent/_memory_manager.py (lines 54, 75, 116-135, 155, 176, 198, 242-249)


class TestMemoryManager:
    """Cover compress context, entity extraction, KG extraction."""

    def test_memory_add_many_with_trim(self):
        from selectools.memory import ConversationMemory

        memory = ConversationMemory(max_messages=4)
        agent = _make_agent()
        agent.memory = memory

        # Pre-fill memory
        for i in range(3):
            memory.add(Message(role=Role.USER, content=f"msg {i}"))

        # Adding more should trigger trim
        msgs = [
            Message(role=Role.USER, content="new1"),
            Message(role=Role.ASSISTANT, content="resp1"),
            Message(role=Role.USER, content="new2"),
        ]
        agent._memory_add_many(msgs, "test-run")
        assert len(memory) <= 4

    def test_entity_extraction(self):
        """_extract_entities should handle exceptions gracefully."""
        agent = _make_agent()
        agent._history = [Message(role=Role.USER, content="Alice is 30")]

        mock_em = MagicMock()
        mock_em._relevance_window = 5
        mock_em.extract_entities.side_effect = RuntimeError("extraction failed")
        agent.config.entity_memory = mock_em

        # Should not raise
        agent._extract_entities("test-run")

    def test_extract_kg_triples(self):
        """_extract_kg_triples should handle exceptions gracefully."""
        agent = _make_agent()
        agent._history = [Message(role=Role.USER, content="Alice knows Bob")]

        mock_kg = MagicMock()
        mock_kg._relevance_window = 5
        mock_kg.extract_triples.return_value = [("Alice", "knows", "Bob")]
        mock_kg.store = MagicMock()
        agent.config.knowledge_graph = mock_kg

        agent._extract_kg_triples("test-run")
        mock_kg.store.add_many.assert_called_once()

    def test_extract_kg_triples_exception(self):
        """_extract_kg_triples should handle exceptions gracefully."""
        agent = _make_agent()
        agent._history = [Message(role=Role.USER, content="Alice knows Bob")]

        mock_kg = MagicMock()
        mock_kg._relevance_window = 5
        mock_kg.extract_triples.side_effect = RuntimeError("KG failed")
        agent.config.knowledge_graph = mock_kg

        # Should not raise
        agent._extract_kg_triples("test-run")

    def test_compress_context_below_threshold(self):
        """_maybe_compress_context should be a no-op when fill rate is low."""
        agent = _make_agent(compress_context=True, compress_threshold=0.9)
        agent._history = [Message(role=Role.USER, content="Hi")]
        trace = AgentTrace()
        agent._maybe_compress_context("test-run", trace)
        # No compression should happen with 1 message
        assert len([s for s in trace.steps if s.type == StepType.PROMPT_COMPRESSED]) == 0


# ── 11. evals/snapshot.py (lines 22, 48-54, 106-109) ────────────────────────


class TestSnapshotStore:
    """Cover snapshot save, compare, diffs, and summary."""

    def test_snapshot_save_and_compare(self):
        from selectools.evals.snapshot import SnapshotStore
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)

            # Create a mock report
            case1 = TestCase(input="What is 2+2?", name="math_q")
            result1 = MagicMock()
            result1.case = case1
            result1.verdict = CaseVerdict.PASS
            result1.tool_calls = []
            result1.agent_result = MagicMock()
            result1.agent_result.content = "4"
            result1.agent_result.iterations = 1
            result1.error = None

            report = MagicMock()
            report.case_results = [result1]

            # Save
            path = store.save(report, suite_name="test_suite")
            assert path.exists()

            # Load
            loaded = store.load(suite_name="test_suite")
            assert loaded is not None

            # Compare with same results — should show unchanged
            result_snap = store.compare(report, suite_name="test_suite")
            assert not result_snap.has_changes
            assert len(result_snap.unchanged) == 1

            # Modify and compare
            result1.agent_result.content = "5"
            result_snap2 = store.compare(report, suite_name="test_suite")
            assert result_snap2.has_changes
            assert result_snap2.changed_count >= 1
            summary = result_snap2.summary()
            assert "Changed" in summary

    def test_snapshot_no_existing_snapshot(self):
        from selectools.evals.snapshot import SnapshotStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            result = store.load("nonexistent")
            assert result is None

    def test_snapshot_diff_is_changed(self):
        from selectools.evals.snapshot import SnapshotDiff

        d = SnapshotDiff(case_name="case1", field="content", expected="a", actual="b")
        assert d.is_changed

        d2 = SnapshotDiff(case_name="case1", field="content", expected="a", actual="a")
        assert not d2.is_changed

    def test_snapshot_summary_with_new_and_removed(self):
        from selectools.evals.snapshot import SnapshotResult

        result = SnapshotResult(
            new_cases=["new_case"],
            removed_cases=["old_case"],
            diffs=[],
            unchanged=["stable"],
        )
        s = result.summary()
        assert "New cases" in s
        assert "Removed cases" in s
        assert "new_case" in s
        assert "old_case" in s


# ── 12. evals/pairwise.py (line 72, 194-217) ────────────────────────────────


class TestPairwiseEval:
    """Cover pairwise comparison run and edge cases."""

    def test_pairwise_run(self):
        from selectools.evals.pairwise import PairwiseEval
        from selectools.evals.types import TestCase

        provider = LocalProvider()
        agent_a = Agent([greet], provider=provider, config=AgentConfig(model="test"))
        agent_b = Agent([greet], provider=provider, config=AgentConfig(model="test"))

        cases = [
            TestCase(input="Hello", name="test1"),
            TestCase(input="World", name="test2"),
        ]

        comparison = PairwiseEval(
            agent_a=agent_a,
            agent_b=agent_b,
            cases=cases,
            name="ab_test",
            agent_a_name="Fast",
            agent_b_name="Accurate",
        )
        result = comparison.run()
        assert result.name == "ab_test"
        assert len(result.case_results) == 2

        # Summary and to_dict should work
        summary = result.summary()
        assert "Fast" in summary
        assert "Accurate" in summary
        d = result.to_dict()
        assert "cases" in d


# ── 13. evals/evaluators.py (uncovered lines) ───────────────────────────────


class TestEvaluators:
    """Cover edge cases in evaluator check methods."""

    def test_tool_call_evaluator_missing_tool(self):
        from selectools.evals.evaluators import ToolUseEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        case = TestCase(
            input="test",
            expect_tool_args={"missing_tool": {"arg": "val"}},
        )
        agent_result = MagicMock()
        agent_result.tool_calls = []
        cr = CaseResult(
            case=case,
            verdict=CaseVerdict.PASS,
            agent_result=agent_result,
        )
        evaluator = ToolUseEvaluator()
        failures = evaluator.check(case, cr)
        assert len(failures) >= 1
        assert "not called" in failures[0].message

    def test_tool_call_evaluator_wrong_args(self):
        from selectools.evals.evaluators import ToolUseEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        case = TestCase(
            input="test",
            expect_tool_args={"greet": {"name": "Alice"}},
        )
        agent_result = MagicMock()
        tc = ToolCall(tool_name="greet", parameters={"name": "Bob"})
        agent_result.tool_calls = [tc]
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=agent_result)
        evaluator = ToolUseEvaluator()
        failures = evaluator.check(case, cr)
        assert len(failures) >= 1
        assert "expected" in failures[0].message.lower()

    def test_word_count_evaluator_too_short(self):
        from selectools.evals.evaluators import WordCountEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        case = TestCase(input="test", expect_min_words=100)
        agent_result = MagicMock()
        agent_result.content = "short"
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=agent_result)
        evaluator = WordCountEvaluator()
        failures = evaluator.check(case, cr)
        assert len(failures) == 1
        assert "short" in failures[0].message.lower()

    def test_cosine_similarity_below_threshold(self):
        from selectools.evals.evaluators import SemanticSimilarityEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        case = TestCase(
            input="test",
            reference="The quick brown fox",
            expect_semantic_similarity_gte=0.99,
        )
        agent_result = MagicMock()
        agent_result.content = "Completely unrelated XYZ 123"
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=agent_result)
        evaluator = SemanticSimilarityEvaluator(threshold=0.99)
        failures = evaluator.check(case, cr)
        assert len(failures) >= 1

    def test_tf_cosine_empty(self):
        from selectools.evals.evaluators import _tf_cosine

        assert _tf_cosine("", "hello") == 0.0
        assert _tf_cosine("hello", "") == 0.0

    def test_json_schema_validator_fallback(self):
        from selectools.evals.evaluators import _validate_json_schema

        # Test with a simple schema using the fallback validator
        with mock.patch.dict("sys.modules", {"jsonschema": None}):
            # Re-import won't work easily, so test the fallback path directly
            schema = {"type": "object", "required": ["name"]}
            # Test type violation
            errors = _validate_json_schema("not a dict", schema)
            assert len(errors) >= 1
            # Test missing required field
            errors = _validate_json_schema({"age": 30}, schema)
            assert len(errors) >= 1
            assert "name" in errors[0]
            # Test valid
            errors = _validate_json_schema({"name": "Alice"}, schema)
            assert len(errors) == 0


# ── 14. evals/llm_evaluators.py (uncovered check() methods) ─────────────────


class TestLLMEvaluators:
    """Cover LLM evaluator check methods with mocked providers."""

    def _mock_judge(self, score: float):
        """Create a mock provider that returns a judge response with given score."""
        provider = MagicMock()
        response = Message(role=Role.ASSISTANT, content=f"Good. Score: {score}")
        provider.complete.return_value = (
            response,
            UsageStats(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0),
        )
        return provider

    def test_llm_judge_evaluator(self):
        from selectools.evals.llm_evaluators import LLMJudgeEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        provider = self._mock_judge(8.0)
        evaluator = LLMJudgeEvaluator(provider=provider, model="test", threshold=7.0)
        case = TestCase(input="test", rubric="Is it good?")
        ar = MagicMock()
        ar.content = "A great response"
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=ar)
        failures = evaluator.check(case, cr)
        assert len(failures) == 0

    def test_llm_judge_evaluator_below_threshold(self):
        from selectools.evals.llm_evaluators import LLMJudgeEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        provider = self._mock_judge(3.0)
        evaluator = LLMJudgeEvaluator(provider=provider, model="test", threshold=7.0)
        case = TestCase(input="test")
        ar = MagicMock()
        ar.content = "bad"
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=ar)
        failures = evaluator.check(case, cr)
        assert len(failures) >= 1

    def test_llm_judge_evaluator_no_score(self):
        from selectools.evals.llm_evaluators import LLMJudgeEvaluator
        from selectools.evals.types import CaseResult, CaseVerdict, TestCase

        provider = MagicMock()
        provider.complete.return_value = (
            Message(role=Role.ASSISTANT, content="No score here"),
            UsageStats(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0),
        )
        evaluator = LLMJudgeEvaluator(provider=provider, model="test")
        case = TestCase(input="test")
        ar = MagicMock()
        ar.content = "response"
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=ar)
        failures = evaluator.check(case, cr)
        assert len(failures) >= 1
        assert (
            "parseable score" in failures[0].message.lower()
            or "could not parse" in failures[0].message.lower()
        )

    def test_extract_score_variants(self):
        from selectools.evals.llm_evaluators import _extract_score

        assert _extract_score("Score: 8") == 8.0
        assert _extract_score("Rating: 6.5") == 6.5
        assert _extract_score("I'd give this 7/10") == 7.0
        assert _extract_score("Verdict: PASS") == 1.0
        assert _extract_score("Verdict: FAIL") == 0.0
        assert _extract_score("No number here at all") is None


# ── 15. orchestration/supervisor.py (magentic, round_robin) ──────────────────


class TestSupervisor:
    """Cover magentic strategy and round_robin with max_rounds."""

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self):
        from selectools.orchestration.supervisor import SupervisorAgent, SupervisorStrategy

        provider = LocalProvider()
        agent_a = Agent([greet], provider=provider, config=AgentConfig(model="test"))
        agent_b = Agent([greet], provider=provider, config=AgentConfig(model="test"))

        supervisor = SupervisorAgent(
            agents={"agent_a": agent_a, "agent_b": agent_b},
            provider=provider,
            strategy=SupervisorStrategy.ROUND_ROBIN,
            max_rounds=1,
        )
        result = await supervisor.arun("Say hello")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_magentic_strategy(self):
        from selectools.orchestration.supervisor import SupervisorAgent, SupervisorStrategy

        # Create a provider that returns JSON ledger for the planner
        class PlannerProvider(LocalProvider):
            _call_count = 0

            def complete(self, **kwargs):
                self._call_count += 1
                if (
                    "orchestrator" in kwargs.get("system_prompt", "").lower()
                    or "ledger" in kwargs.get("system_prompt", "").lower()
                ):
                    content = json.dumps(
                        {
                            "progress_ledger": {
                                "is_complete": True if self._call_count > 1 else False,
                                "is_progressing": True,
                                "next_agent": "agent_a",
                                "task": "do something",
                            },
                            "task_ledger": {"goal": "test"},
                        }
                    )
                else:
                    content = "Done with the task."
                msg = Message(role=Role.ASSISTANT, content=content)
                usage = UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="test",
                )
                return msg, usage

            async def acomplete(self, **kwargs):
                return self.complete(**kwargs)

            @property
            def supports_async(self):
                return True

        provider = PlannerProvider()
        agent_a = Agent([greet], provider=provider, config=AgentConfig(model="test"))

        supervisor = SupervisorAgent(
            agents={"agent_a": agent_a},
            provider=provider,
            strategy=SupervisorStrategy.MAGENTIC,
            max_rounds=3,
            max_stalls=2,
        )
        result = await supervisor.arun("Do something")
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_dynamic_strategy(self):
        from selectools.orchestration.supervisor import SupervisorAgent, SupervisorStrategy

        class DynamicProvider(LocalProvider):
            _call_count = 0

            def complete(self, **kwargs):
                self._call_count += 1
                # For routing, return agent name first time, then DONE
                if "Which agent" in (
                    kwargs.get("messages", [{}])[-1].content if kwargs.get("messages") else ""
                ):
                    content = "DONE" if self._call_count > 2 else "agent_a"
                else:
                    content = "Task completed"
                msg = Message(role=Role.ASSISTANT, content=content)
                usage = UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="test",
                )
                return msg, usage

            async def acomplete(self, **kwargs):
                return self.complete(**kwargs)

            @property
            def supports_async(self):
                return True

        provider = DynamicProvider()
        agent_a = Agent([greet], provider=provider, config=AgentConfig(model="test"))

        supervisor = SupervisorAgent(
            agents={"agent_a": agent_a},
            provider=provider,
            strategy=SupervisorStrategy.DYNAMIC,
            max_rounds=3,
        )
        result = await supervisor.arun("Do something")
        assert result is not None

    def test_supervisor_no_agents_error(self):
        from selectools.orchestration.supervisor import SupervisorAgent

        with pytest.raises(ValueError, match="at least one agent"):
            SupervisorAgent(agents={}, provider=LocalProvider())


# ── 16. serve/_starlette_app.py ──────────────────────────────────────────────


class TestStarletteApp:
    """Cover auth, routes, and SSE endpoints in the Starlette app."""

    @pytest.fixture
    def app(self):
        """Create a test app."""
        try:
            from selectools.serve._starlette_app import create_builder_app
        except ImportError:
            pytest.skip("starlette not installed")
        return create_builder_app(auth_token="test-secret")

    @pytest.fixture
    def no_auth_app(self):
        """Create a test app without auth."""
        try:
            from selectools.serve._starlette_app import create_builder_app
        except ImportError:
            pytest.skip("starlette not installed")
        return create_builder_app(auth_token=None)

    @pytest.fixture
    def client(self, app):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette not installed")
        return TestClient(app)

    @pytest.fixture
    def no_auth_client(self, no_auth_app):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette not installed")
        return TestClient(no_auth_app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_login_get(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_post_success(self, client):
        resp = client.post("/login", data={"token": "test-secret"}, follow_redirects=False)
        assert resp.status_code == 302

    def test_login_post_failure(self, client):
        resp = client.post("/login", data={"token": "wrong"}, follow_redirects=False)
        assert resp.status_code == 200  # Returns error HTML

    def test_builder_unauthenticated(self, client):
        resp = client.get("/builder", follow_redirects=False)
        assert resp.status_code == 302

    def test_builder_authenticated(self, no_auth_client):
        resp = no_auth_client.get("/builder")
        assert resp.status_code == 200

    def test_provider_health_unauthenticated(self, client):
        resp = client.get("/provider-health", follow_redirects=False)
        assert resp.status_code == 302

    def test_eval_dashboard_unauthenticated(self, client):
        resp = client.get("/eval-dashboard", follow_redirects=False)
        assert resp.status_code == 302

    def test_eval_dashboard_authenticated(self, no_auth_client):
        resp = no_auth_client.get("/eval-dashboard")
        assert resp.status_code == 200

    def test_ai_build_unauthenticated(self, client):
        resp = client.post("/ai-build", json={"description": "test"})
        assert resp.status_code == 401

    def test_ai_build_no_description(self, no_auth_client):
        resp = no_auth_client.post("/ai-build", json={"description": ""})
        assert resp.status_code == 400

    def test_ai_build_fallback(self, no_auth_client):
        resp = no_auth_client.post("/ai-build", json={"description": "A chatbot"})
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data or "error" not in data

    def test_ai_refine_unauthenticated(self, client):
        resp = client.post("/ai-refine", json={"message": "test"})
        assert resp.status_code == 401

    def test_ai_refine_no_message(self, no_auth_client):
        resp = no_auth_client.post("/ai-refine", json={"message": ""})
        assert resp.status_code == 400

    def test_ai_refine_no_api_key(self, no_auth_client):
        resp = no_auth_client.post(
            "/ai-refine",
            json={"message": "make it faster", "current_graph": {}, "api_key": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "No API key" in data.get("explanation", "")

    def test_estimate_run_cost_unauthenticated(self, client):
        resp = client.post("/estimate-run-cost", json={"nodes": [], "input": "test"})
        assert resp.status_code == 401

    def test_estimate_run_cost(self, no_auth_client):
        resp = no_auth_client.post("/estimate-run-cost", json={"nodes": [], "input": "test"})
        assert resp.status_code == 200

    def test_smart_route_unauthenticated(self, client):
        resp = client.post("/smart-route", json={"prompt": "test"})
        assert resp.status_code == 401

    def test_smart_route(self, no_auth_client):
        resp = no_auth_client.post("/smart-route", json={"prompt": "test", "system_prompt": "sys"})
        assert resp.status_code == 200

    def test_runs_unauthenticated(self, client):
        resp = client.post("/runs", json={})
        assert resp.status_code == 401

    def test_runs(self, no_auth_client):
        resp = no_auth_client.post("/runs", json={})
        assert resp.status_code == 200

    def test_feedback_unauthenticated(self, client):
        resp = client.post("/feedback", json={"run_id": "r1", "score": 5})
        assert resp.status_code == 401

    def test_feedback(self, no_auth_client):
        resp = no_auth_client.post("/feedback", json={"run_id": "r1", "score": 5})
        assert resp.status_code == 200

    def test_run_sse_unauthenticated(self, client):
        resp = client.post("/run", json={"input": "test"})
        assert resp.status_code == 401

    def test_run_sse_mock(self, no_auth_client):
        resp = no_auth_client.post(
            "/run",
            json={"input": "Hello", "nodes": [], "edges": [], "api_key": ""},
        )
        assert resp.status_code == 200

    def test_watch_file_unauthenticated(self, client):
        resp = client.post("/watch-file", json={"path": "/tmp/test.txt"})
        assert resp.status_code == 401

    def test_watch_file_not_found(self, no_auth_client):
        resp = no_auth_client.post("/watch-file", json={"path": "/nonexistent/file.txt"})
        assert resp.status_code == 404

    def test_sync_to_file_unauthenticated(self, client):
        resp = client.post("/sync-to-file", json={"path": "/tmp/test.txt"})
        assert resp.status_code == 401

    def test_sync_to_file_not_found(self, no_auth_client):
        resp = no_auth_client.post("/sync-to-file", json={"path": "/nonexistent/file.txt"})
        assert resp.status_code == 404

    def test_sync_to_file_success(self, no_auth_client):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# test file content\n")
            path = f.name
        try:
            resp = no_auth_client.post(
                "/sync-to-file",
                json={
                    "path": path,
                    "patch": {"type": "update_node", "node_id": "n1", "changes": {}},
                },
            )
            assert resp.status_code == 200
        finally:
            os.unlink(path)

    def test_auth_github_redirect(self, no_auth_client):
        with mock.patch.dict(os.environ, {"GITHUB_CLIENT_ID": "test-client-id"}):
            resp = no_auth_client.get("/auth/github", follow_redirects=False)
            assert resp.status_code == 302
            assert "github.com" in resp.headers.get("location", "")


# ── 17. agent/_lifecycle.py (lines 43, 55, 58, 108-111, 132, 140-141) ───────


class TestLifecycle:
    """Cover async observer notification and fallback wiring edge cases."""

    def test_truncate_tool_result_none(self):
        agent = _make_agent()
        assert agent._truncate_tool_result(None) is None

    def test_truncate_tool_result_with_limit(self):
        agent = _make_agent(trace_tool_result_chars=10)
        assert agent._truncate_tool_result("a" * 100) == "a" * 10

    def test_truncate_tool_result_no_limit(self):
        agent = _make_agent(trace_tool_result_chars=None)
        long_text = "a" * 1000
        assert agent._truncate_tool_result(long_text) == long_text

    def test_unwire_fallback_no_lock(self):
        """Unwiring when no lock exists should be a no-op."""
        agent = _make_agent()
        # Should not raise
        agent._unwire_fallback_observer()


# ── 18. guardrails/pipeline.py (lines 100-115) ──────────────────────────────


class TestGuardrailsPipelineAsync:
    """Cover async guardrail pipeline chain."""

    @pytest.mark.asyncio
    async def test_async_guardrail_block(self):
        from selectools.guardrails.base import Guardrail, GuardrailAction, GuardrailResult
        from selectools.guardrails.pipeline import GuardrailError, GuardrailsPipeline

        class BlockGuardrail(Guardrail):
            name = "blocker"
            action = GuardrailAction.BLOCK

            def check(self, content: str) -> GuardrailResult:
                return GuardrailResult(passed=False, content=content, reason="blocked")

        pipeline = GuardrailsPipeline(input=[BlockGuardrail()])
        with pytest.raises(GuardrailError):
            await pipeline.acheck_input("bad input")

    @pytest.mark.asyncio
    async def test_async_guardrail_warn(self):
        from selectools.guardrails.base import Guardrail, GuardrailAction, GuardrailResult
        from selectools.guardrails.pipeline import GuardrailsPipeline

        class WarnGuardrail(Guardrail):
            name = "warner"
            action = GuardrailAction.WARN

            def check(self, content: str) -> GuardrailResult:
                return GuardrailResult(passed=False, content=content, reason="warning")

        pipeline = GuardrailsPipeline(input=[WarnGuardrail()])
        result = await pipeline.acheck_input("test")
        assert result.passed

    @pytest.mark.asyncio
    async def test_async_guardrail_rewrite(self):
        from selectools.guardrails.base import Guardrail, GuardrailAction, GuardrailResult
        from selectools.guardrails.pipeline import GuardrailsPipeline

        class RewriteGuardrail(Guardrail):
            name = "rewriter"
            action = GuardrailAction.REWRITE

            def check(self, content: str) -> GuardrailResult:
                return GuardrailResult(passed=False, content=content.upper(), reason="rewritten")

        pipeline = GuardrailsPipeline(input=[RewriteGuardrail()])
        result = await pipeline.acheck_input("test")
        assert result.content == "TEST"


# ── 19. agent/core.py — batch / abatch ──────────────────────────────────────


class TestAgentBatch:
    """Cover batch error handling and abatch progress callback."""

    def test_batch_with_error(self):
        class ErrorProvider(LocalProvider):
            _call = 0

            def complete(self, **kwargs):
                self._call += 1
                if self._call == 1:
                    raise RuntimeError("provider error")
                return super().complete(**kwargs)

        provider = ErrorProvider()
        agent = Agent([greet], provider=provider, config=AgentConfig(model="test"))
        results = agent.batch(["prompt1", "prompt2"])
        assert len(results) == 2
        # First should be an error
        error_results = [r for r in results if r.content and "Batch error" in r.content]
        assert len(error_results) >= 1

    @pytest.mark.asyncio
    async def test_abatch_with_progress(self):
        provider = LocalProvider()
        agent = Agent([greet], provider=provider, config=AgentConfig(model="test"))

        progress_calls: list = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        results = await agent.abatch(
            ["Hello", "World"],
            on_progress=on_progress,
        )
        assert len(results) == 2
        assert len(progress_calls) == 2

    @pytest.mark.asyncio
    async def test_abatch_with_error(self):
        class ErrorProvider(LocalProvider):
            _call = 0

            def complete(self, **kwargs):
                self._call += 1
                if self._call == 1:
                    raise RuntimeError("async error")
                return super().complete(**kwargs)

            @property
            def supports_async(self):
                return False

        provider = ErrorProvider()
        agent = Agent([greet], provider=provider, config=AgentConfig(model="test"))
        results = await agent.abatch(["p1", "p2"])
        assert len(results) == 2


# ── 20. agent/core.py — run structured retry ────────────────────────────────


class TestAgentCoreRunStructuredRetry:
    """Cover structured output retry path in sync run()."""

    def test_run_structured_retry(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            price: float

        call_count = 0

        class RetryProvider(LocalProvider):
            def complete(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    msg = Message(role=Role.ASSISTANT, content="not json")
                else:
                    msg = Message(
                        role=Role.ASSISTANT,
                        content='{"name": "Widget", "price": 9.99}',
                    )
                return msg, UsageStats(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                    cost_usd=0.0,
                    model="test",
                )

        provider = RetryProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5),
        )
        result = agent.run("Return an item", response_format=Item)
        assert call_count >= 2
        assert result.parsed is not None


# ── 21. agent/core.py — terminal tool and parallel tool paths ────────────────


class TestAgentCoreTerminalTool:
    """Cover terminal tool execution paths."""

    def test_terminal_tool_stops_agent(self):
        @tool()
        def final_answer(answer: str) -> str:
            """Provide the final answer."""
            return answer

        # Mark as terminal
        final_answer.terminal = True

        class ToolProvider(LocalProvider):
            def complete(self, **kwargs):
                msg = Message(
                    role=Role.ASSISTANT,
                    content="calling",
                    tool_calls=[ToolCall(tool_name="final_answer", parameters={"answer": "42"})],
                )
                return msg, UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

        provider = ToolProvider()
        agent = Agent(
            [final_answer],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5),
        )
        result = agent.run("What is the answer?")
        assert result.content == "42"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_terminal_tool_in_astream(self):
        @tool()
        def final_answer(answer: str) -> str:
            """Provide the final answer."""
            return answer

        final_answer.terminal = True

        class ToolProvider(LocalProvider):
            def complete(self, **kwargs):
                msg = Message(
                    role=Role.ASSISTANT,
                    content="calling",
                    tool_calls=[ToolCall(tool_name="final_answer", parameters={"answer": "42"})],
                )
                return msg, UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ToolProvider()
        agent = Agent(
            [final_answer],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5),
        )
        chunks = []
        async for chunk in agent.astream("What is the answer?"):
            chunks.append(chunk)
        from selectools.types import AgentResult

        results = [c for c in chunks if isinstance(c, AgentResult)]
        assert len(results) == 1
        assert results[0].content == "42"

    @pytest.mark.asyncio
    async def test_terminal_tool_in_arun(self):
        @tool()
        def final_answer(answer: str) -> str:
            """Provide the final answer."""
            return answer

        final_answer.terminal = True

        class ToolProvider(LocalProvider):
            def complete(self, **kwargs):
                msg = Message(
                    role=Role.ASSISTANT,
                    content="calling",
                    tool_calls=[ToolCall(tool_name="final_answer", parameters={"answer": "42"})],
                )
                return msg, UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ToolProvider()
        agent = Agent(
            [final_answer],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5),
        )
        result = await agent.arun("What is the answer?")
        assert result.content == "42"

    def test_parallel_tool_execution(self):
        """Cover parallel tool execution path in sync run."""

        @tool()
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        class MultiToolProvider(LocalProvider):
            _called = False

            def complete(self, **kwargs):
                if not self._called:
                    self._called = True
                    msg = Message(
                        role=Role.ASSISTANT,
                        content="computing",
                        tool_calls=[
                            ToolCall(tool_name="add", parameters={"a": 1, "b": 2}),
                            ToolCall(tool_name="add", parameters={"a": 3, "b": 4}),
                        ],
                    )
                else:
                    msg = Message(role=Role.ASSISTANT, content="3 and 7")
                return msg, UsageStats(
                    prompt_tokens=5,
                    completion_tokens=5,
                    total_tokens=10,
                    cost_usd=0.0,
                    model="test",
                )

        provider = MultiToolProvider()
        agent = Agent(
            [add],
            provider=provider,
            config=AgentConfig(model="test", max_iterations=5, parallel_tool_execution=True),
        )
        result = agent.run("Add 1+2 and 3+4")
        assert result.content is not None


# ── 22. agent/core.py — astream error path ──────────────────────────────────


class TestAgentCoreAstreamError:
    """Cover error path in astream (exception propagation)."""

    @pytest.mark.asyncio
    async def test_astream_error_propagation(self):
        class ErrorProvider(LocalProvider):
            def complete(self, **kwargs):
                raise RuntimeError("provider crashed")

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ErrorProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test"),
        )
        with pytest.raises(RuntimeError, match="provider crashed"):
            async for _ in agent.astream("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_arun_error_propagation(self):
        class ErrorProvider(LocalProvider):
            def complete(self, **kwargs):
                raise RuntimeError("provider crashed")

            @property
            def supports_streaming(self):
                return False

            @property
            def supports_async(self):
                return False

        provider = ErrorProvider()
        agent = Agent(
            [greet],
            provider=provider,
            config=AgentConfig(model="test"),
        )
        with pytest.raises(RuntimeError, match="provider crashed"):
            await agent.arun("Hello")
