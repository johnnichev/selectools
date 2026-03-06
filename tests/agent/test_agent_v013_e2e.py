"""
End-to-end tests for v0.13.0 features against a real OpenAI API.

Tests cover:
1. Structured Output (response_format with Pydantic model)
2. Execution Traces (result.trace populated with TraceStep entries)
3. Reasoning Visibility (result.reasoning, result.reasoning_history)
4. Provider Fallback (FallbackProvider with circuit breaker)
5. Batch Processing (agent.batch, agent.abatch)
6. Tool Policy Engine (allow/review/deny with human-in-the-loop)
7. Tool-Pair-Aware Memory Trimming

To run: pytest --run-e2e tests/agent/test_agent_v013_e2e.py -v
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

import pytest

from selectools import (
    Agent,
    AgentConfig,
    AgentResult,
    ConversationMemory,
    FallbackProvider,
    Message,
    PolicyDecision,
    Role,
    ToolPolicy,
    tool,
)
from selectools.providers.base import Provider, ProviderError
from selectools.providers.openai_provider import OpenAIProvider
from selectools.trace import AgentTrace
from selectools.types import Message as Msg
from selectools.usage import UsageStats

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool()
def get_population(country: str) -> str:
    """Get the approximate population of a country."""
    populations = {
        "france": "67 million",
        "japan": "125 million",
        "brazil": "215 million",
    }
    return populations.get(country.lower(), "unknown")


@tool()
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)


@tool()
def search_docs(query: str) -> str:
    """Search the documentation."""
    return f"Results for '{query}': document A, document B"


@tool()
def delete_account(user_id: str) -> str:
    """Delete a user account permanently."""
    return f"Account {user_id} deleted"


@tool()
def noop(message: str) -> str:
    """A simple echo tool. Returns the message as-is."""
    return message


class _RetriableFailProvider:
    """Test provider that always raises a retriable (rate-limit) error."""

    name = "failing-provider"
    supports_streaming = False
    supports_async = False

    def complete(self, **kwargs: Any) -> Any:
        raise ProviderError("429 rate limit exceeded")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def provider(api_key: str) -> OpenAIProvider:
    return OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# 1. Structured Output
# ---------------------------------------------------------------------------


class TestStructuredOutputE2E:
    """Structured output: LLM returns validated JSON matching a Pydantic schema."""

    @pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
    def test_pydantic_model(self, provider: OpenAIProvider) -> None:
        class CityInfo(BaseModel):  # type: ignore[misc]
            city: str
            country: str
            population_estimate: str

        agent = Agent(
            tools=[get_population],
            provider=provider,
            config=AgentConfig(max_iterations=5),
        )

        result = agent.ask(
            "What is the population of France? Return as structured data about Paris.",
            response_format=CityInfo,
        )

        assert result.parsed is not None
        info = result.parsed
        assert hasattr(info, "city")
        assert hasattr(info, "country")
        assert isinstance(info.city, str)
        assert len(info.city) > 0

    @pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
    def test_dict_json_schema(self, provider: OpenAIProvider) -> None:
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "result": {"type": "number"},
            },
            "required": ["product", "result"],
        }

        agent = Agent(
            tools=[multiply],
            provider=provider,
            config=AgentConfig(max_iterations=5),
        )

        result = agent.ask("What is 7 times 8?", response_format=schema)

        assert result.parsed is not None
        assert "result" in result.parsed
        assert result.parsed["result"] == 56


# ---------------------------------------------------------------------------
# 2. Execution Traces
# ---------------------------------------------------------------------------


class TestTracesE2E:
    """Execution traces: every step in the agent loop is captured."""

    def test_trace_with_tool_call(self, provider: OpenAIProvider) -> None:
        agent = Agent(
            tools=[get_population],
            provider=provider,
            config=AgentConfig(max_iterations=4),
        )

        result = agent.run([Message(role=Role.USER, content="What is the population of Japan?")])

        assert result.trace is not None
        trace: AgentTrace = result.trace
        assert len(trace) > 0

        llm_steps = trace.filter(type="llm_call")
        assert len(llm_steps) >= 1
        assert llm_steps[0].model is not None

        tool_exec_steps = trace.filter(type="tool_execution")
        assert len(tool_exec_steps) >= 1
        assert tool_exec_steps[0].tool_name == "get_population"
        assert tool_exec_steps[0].duration_ms > 0

    def test_timeline_and_export(self, provider: OpenAIProvider) -> None:
        agent = Agent(
            tools=[multiply],
            provider=provider,
            config=AgentConfig(max_iterations=4),
        )

        result = agent.ask("Calculate 12 times 5")

        assert result.trace is not None
        trace: AgentTrace = result.trace

        timeline_str = trace.timeline()
        assert "llm_call" in timeline_str
        assert "Total:" in timeline_str

        trace_dict = trace.to_dict()
        assert "steps" in trace_dict
        assert trace_dict["step_count"] == len(trace)
        assert trace.total_duration_ms > 0

    def test_trace_without_tool_use(self, provider: OpenAIProvider) -> None:
        agent = Agent(
            tools=[noop],
            provider=provider,
        )

        result = agent.ask("Say hello, do not use any tools.")

        assert result.trace is not None
        assert len(result.trace) >= 1
        llm_steps = result.trace.filter(type="llm_call")
        assert len(llm_steps) >= 1


# ---------------------------------------------------------------------------
# 3. Reasoning Visibility
# ---------------------------------------------------------------------------


class TestReasoningE2E:
    """Reasoning visibility: LLM reasoning text captured from tool selections."""

    def test_reasoning_with_tool_call(self, provider: OpenAIProvider) -> None:
        agent = Agent(
            tools=[get_population],
            provider=provider,
            config=AgentConfig(max_iterations=4),
        )

        result = agent.run([Message(role=Role.USER, content="What is the population of Brazil?")])

        assert isinstance(result.reasoning_history, list)

        assert result.trace is not None
        assert len(result.trace.filter(type="tool_selection")) >= 1


# ---------------------------------------------------------------------------
# 4. Provider Fallback
# ---------------------------------------------------------------------------


class TestFallbackProviderE2E:
    """FallbackProvider: automatic failover between real providers."""

    def test_uses_first_working_provider(self, api_key: str) -> None:
        primary = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")
        backup = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")

        fallback = FallbackProvider(providers=[primary, backup])
        agent = Agent(tools=[noop], provider=fallback)

        result = agent.ask("Say 'working'")

        assert len(result.content) > 0
        assert fallback.provider_used is not None

    def test_skips_bad_provider(self, api_key: str) -> None:
        bad_provider = _RetriableFailProvider()
        good_provider = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")

        fallback_events: List[str] = []

        def on_fallback(failed: str, next_p: str, exc: Exception) -> None:
            fallback_events.append(f"{failed}->{next_p}")

        fallback = FallbackProvider(
            providers=[bad_provider, good_provider],
            on_fallback=on_fallback,
        )

        agent = Agent(tools=[noop], provider=fallback)
        result = agent.ask("Say 'recovered'")

        assert len(result.content) > 0
        assert len(fallback_events) >= 1

    def test_all_fail_returns_error(self) -> None:
        bad1 = _RetriableFailProvider()
        bad2 = _RetriableFailProvider()

        fallback = FallbackProvider(providers=[bad1, bad2])
        agent = Agent(tools=[noop], provider=fallback)

        result = agent.ask("This should fail")
        assert "error" in result.content.lower() or "provider" in result.content.lower()


# ---------------------------------------------------------------------------
# 5. Batch Processing
# ---------------------------------------------------------------------------


class TestBatchE2E:
    """Batch processing: concurrent prompt execution."""

    def test_batch_sync(self, provider: OpenAIProvider) -> None:
        agent = Agent(tools=[noop], provider=provider)

        prompts = [
            "What is 2+2? Reply with just the number.",
            "What is 3+3? Reply with just the number.",
            "What is 5+5? Reply with just the number.",
        ]

        results = agent.batch(prompts, max_concurrency=3)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, AgentResult)
            assert len(r.content) > 0

    def test_batch_async(self, provider: OpenAIProvider) -> None:
        agent = Agent(tools=[noop], provider=provider)

        prompts = ["Say 'alpha'", "Say 'beta'"]

        async def _run() -> List[AgentResult]:
            return list(await agent.abatch(prompts, max_concurrency=2))

        results: List[AgentResult] = asyncio.run(_run())

        assert len(results) == 2
        for r in results:
            assert isinstance(r, AgentResult)
            assert len(r.content) > 0

    @pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
    def test_batch_with_structured_output(self, provider: OpenAIProvider) -> None:
        class MathAnswer(BaseModel):  # type: ignore[misc]
            answer: int

        agent = Agent(
            tools=[multiply],
            provider=provider,
            config=AgentConfig(max_iterations=4),
        )

        results = agent.batch(
            ["What is 3 times 4?", "What is 7 times 2?"],
            max_concurrency=2,
            response_format=MathAnswer,
        )

        assert len(results) == 2
        for r in results:
            assert r.parsed is not None
            assert hasattr(r.parsed, "answer")

    def test_batch_each_has_own_trace(self, provider: OpenAIProvider) -> None:
        agent = Agent(tools=[noop], provider=provider)

        results = agent.batch(["Say 'one'", "Say 'two'"], max_concurrency=2)

        assert len(results) == 2
        for r in results:
            assert r.trace is not None
            assert len(r.trace) >= 1


# ---------------------------------------------------------------------------
# 6. Tool Policy Engine
# ---------------------------------------------------------------------------


class TestToolPolicyE2E:
    """Tool policy engine: allow/review/deny rules enforced at execution."""

    def test_allow_runs_tool(self, provider: OpenAIProvider) -> None:
        policy = ToolPolicy(allow=["search_*", "noop"], deny=["delete_*"])

        agent = Agent(
            tools=[search_docs, delete_account, noop],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool_policy=policy,
            ),
        )

        result = agent.run(
            [Message(role=Role.USER, content="Search docs for 'installation guide'")]
        )

        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "search_docs" in tool_names

    def test_deny_blocks_tool(self, provider: OpenAIProvider) -> None:
        policy = ToolPolicy(allow=["search_*", "noop"], deny=["delete_*"])

        agent = Agent(
            tools=[search_docs, delete_account, noop],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool_policy=policy,
            ),
        )

        result = agent.run([Message(role=Role.USER, content="Delete account user-123")])

        assert result.trace is not None
        policy_errors = [
            s for s in result.trace.steps if s.type == "error" and s.tool_name == "delete_account"
        ]
        has_denial = (
            len(policy_errors) >= 1
            or "denied" in result.content.lower()
            or "cannot" in result.content.lower()
            or "not allowed" in result.content.lower()
        )
        assert has_denial, f"Expected policy denial, got: {result.content}"

    def test_review_with_approve(self, provider: OpenAIProvider) -> None:
        approval_log: List[str] = []

        def auto_approve(tool_name: str, tool_args: Dict[str, Any], reason: str) -> bool:
            approval_log.append(tool_name)
            return True

        policy = ToolPolicy(review=["search_*"])

        agent = Agent(
            tools=[search_docs, noop],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool_policy=policy,
                confirm_action=auto_approve,
            ),
        )

        result = agent.run([Message(role=Role.USER, content="Search docs for 'deployment'")])

        if result.tool_calls:
            assert "search_docs" in approval_log

    def test_review_with_reject(self, provider: OpenAIProvider) -> None:
        def always_reject(tool_name: str, tool_args: Dict[str, Any], reason: str) -> bool:
            return False

        policy = ToolPolicy(review=["search_*"])

        agent = Agent(
            tools=[search_docs, noop],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool_policy=policy,
                confirm_action=always_reject,
            ),
        )

        result = agent.run([Message(role=Role.USER, content="Search docs for 'security'")])

        assert result.trace is not None


# ---------------------------------------------------------------------------
# 7. Tool-Pair-Aware Memory Trimming
# ---------------------------------------------------------------------------


class TestToolPairMemoryE2E:
    """Tool-pair-aware memory trimming: assistant+tool messages stay paired."""

    def test_memory_preserves_tool_pairs_after_trim(self, provider: OpenAIProvider) -> None:
        memory = ConversationMemory(max_messages=6)

        agent = Agent(
            tools=[get_population, multiply],
            provider=provider,
            config=AgentConfig(max_iterations=4),
            memory=memory,
        )

        agent.ask("What is the population of France?")
        agent.ask("What is 6 times 9?")
        agent.ask("What is the population of Japan?")

        history = memory.get_history()

        for i, msg in enumerate(history):
            if msg.role == Role.TOOL:
                assert i > 0, "Tool message cannot be first in history"
                prev = history[i - 1]
                assert prev.role == Role.ASSISTANT, (
                    f"Tool message at index {i} not preceded by assistant " f"(found {prev.role})"
                )


# ---------------------------------------------------------------------------
# 8. Combined: multiple features working together
# ---------------------------------------------------------------------------


class TestCombinedE2E:
    """Multiple v0.13.0 features working together in one run."""

    @pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
    def test_structured_output_with_trace_and_tools(self, provider: OpenAIProvider) -> None:
        class PopulationReport(BaseModel):  # type: ignore[misc]
            country: str
            population: str

        agent = Agent(
            tools=[get_population],
            provider=provider,
            config=AgentConfig(max_iterations=6),
        )

        result = agent.ask(
            "Look up the population of Brazil and return a structured report.",
            response_format=PopulationReport,
        )

        assert result.parsed is not None
        assert hasattr(result.parsed, "country")

        assert result.trace is not None
        assert len(result.trace) >= 2
        assert result.trace.filter(type="llm_call")
        assert result.trace.filter(type="tool_execution")

    def test_fallback_with_trace(self, api_key: str) -> None:
        bad = _RetriableFailProvider()
        good = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")

        fallback = FallbackProvider(providers=[bad, good])
        agent = Agent(tools=[noop], provider=fallback)

        result = agent.ask("Say 'hello'")

        assert len(result.content) > 0
        assert result.trace is not None
        assert len(result.trace.filter(type="llm_call")) >= 1
