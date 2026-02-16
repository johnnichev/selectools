"""
Tests for Phase 8: Parallel Tool Execution.

Verifies that when the LLM requests multiple tool calls in a single response,
they are executed concurrently (via asyncio.gather for async, ThreadPoolExecutor
for sync) rather than sequentially.

Tests cover:
- Parallel async execution (arun) proves wall-clock speedup
- Parallel sync execution (run) proves wall-clock speedup
- Result ordering is preserved regardless of completion order
- Error handling: one tool failure doesn't break the others
- Hooks fire for every tool
- Disabling parallel execution falls back to sequential
- Single tool call does not use parallel path
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Tuple

import pytest

from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.types import ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers: a provider that returns multiple native tool calls in one response
# ---------------------------------------------------------------------------


class MultiToolProvider:
    """A fake provider that returns N native tool calls on the first call,
    then a plain text response on the second call."""

    name = "multi-tool-fake"
    supports_streaming = False

    def __init__(self, tool_calls: List[ToolCall], final_text: str = "Done.") -> None:
        self._tool_calls = tool_calls
        self._final_text = final_text
        self._call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self._call_count += 1
        if self._call_count == 1:
            return (
                Message(role=Role.ASSISTANT, content="", tool_calls=self._tool_calls),
                UsageStats(),
            )
        return (
            Message(role=Role.ASSISTANT, content=self._final_text),
            UsageStats(),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools with artificial delays to prove parallel execution
# ---------------------------------------------------------------------------

SLEEP_SECONDS = 0.15  # per tool; parallel should overlap


@tool(description="Slow tool A")
def slow_tool_a(x: int) -> str:
    time.sleep(SLEEP_SECONDS)
    return f"a={x}"


@tool(description="Slow tool B")
def slow_tool_b(x: int) -> str:
    time.sleep(SLEEP_SECONDS)
    return f"b={x}"


@tool(description="Slow tool C")
def slow_tool_c(x: int) -> str:
    time.sleep(SLEEP_SECONDS)
    return f"c={x}"


@tool(description="Failing tool")
def failing_tool(x: int) -> str:
    raise RuntimeError("boom")


# Async-native versions of the slow tools
@tool(description="Async slow tool A")
async def async_slow_a(x: int) -> str:
    await asyncio.sleep(SLEEP_SECONDS)
    return f"async_a={x}"


@tool(description="Async slow tool B")
async def async_slow_b(x: int) -> str:
    await asyncio.sleep(SLEEP_SECONDS)
    return f"async_b={x}"


@tool(description="Async slow tool C")
async def async_slow_c(x: int) -> str:
    await asyncio.sleep(SLEEP_SECONDS)
    return f"async_c={x}"


@tool(description="Async failing tool")
async def async_failing_tool(x: int) -> str:
    raise RuntimeError("async_boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelToolExecution:
    """Verify tools execute concurrently when parallel_tool_execution=True."""

    def _make_calls(self) -> List[ToolCall]:
        return [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="slow_tool_b", parameters={"x": 2}, id="c2"),
            ToolCall(tool_name="slow_tool_c", parameters={"x": 3}, id="c3"),
        ]

    def _make_async_calls(self) -> List[ToolCall]:
        return [
            ToolCall(tool_name="async_slow_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="async_slow_b", parameters={"x": 2}, id="c2"),
            ToolCall(tool_name="async_slow_c", parameters={"x": 3}, id="c3"),
        ]

    # ---- Sync (run) -------------------------------------------------------

    def test_parallel_sync_is_faster_than_sequential(self) -> None:
        """3 tools each sleeping 0.15s: parallel ~0.15s, sequential ~0.45s."""
        provider = MultiToolProvider(self._make_calls())
        agent = Agent(
            tools=[slow_tool_a, slow_tool_b, slow_tool_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )

        start = time.time()
        result = agent.run([Message(role=Role.USER, content="go")])
        elapsed = time.time() - start

        assert result.content == "Done."
        # Parallel: should be significantly faster than 3 * SLEEP_SECONDS
        # Allow generous margin but ensure it's not fully sequential
        assert (
            elapsed < SLEEP_SECONDS * 2.5
        ), f"Parallel took {elapsed:.2f}s, expected < {SLEEP_SECONDS * 2.5:.2f}s"

    def test_sequential_sync_when_disabled(self) -> None:
        """With parallel_tool_execution=False, tools run sequentially."""
        provider = MultiToolProvider(self._make_calls())
        agent = Agent(
            tools=[slow_tool_a, slow_tool_b, slow_tool_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=False),
        )

        start = time.time()
        result = agent.run([Message(role=Role.USER, content="go")])
        elapsed = time.time() - start

        assert result.content == "Done."
        # Sequential: should take at least ~3 * SLEEP_SECONDS
        assert (
            elapsed >= SLEEP_SECONDS * 2.5
        ), f"Sequential took {elapsed:.2f}s, expected >= {SLEEP_SECONDS * 2.5:.2f}s"

    # ---- Async (arun) -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_parallel_async_is_faster_than_sequential(self) -> None:
        """3 async tools each sleeping 0.15s: parallel ~0.15s."""
        provider = MultiToolProvider(self._make_async_calls())
        agent = Agent(
            tools=[async_slow_a, async_slow_b, async_slow_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )

        start = time.time()
        result = await agent.arun([Message(role=Role.USER, content="go")])
        elapsed = time.time() - start

        assert result.content == "Done."
        assert (
            elapsed < SLEEP_SECONDS * 2.5
        ), f"Parallel async took {elapsed:.2f}s, expected < {SLEEP_SECONDS * 2.5:.2f}s"

    @pytest.mark.asyncio
    async def test_sequential_async_when_disabled(self) -> None:
        """With parallel_tool_execution=False, async tools run sequentially."""
        provider = MultiToolProvider(self._make_async_calls())
        agent = Agent(
            tools=[async_slow_a, async_slow_b, async_slow_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=False),
        )

        start = time.time()
        result = await agent.arun([Message(role=Role.USER, content="go")])
        elapsed = time.time() - start

        assert result.content == "Done."
        assert (
            elapsed >= SLEEP_SECONDS * 2.5
        ), f"Sequential async took {elapsed:.2f}s, expected >= {SLEEP_SECONDS * 2.5:.2f}s"


class TestParallelResultOrdering:
    """Verify results are appended in the original tool-call order."""

    def test_result_order_preserved_sync(self) -> None:
        """Tool results appear in history in the same order as the tool calls."""
        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 10}, id="c1"),
            ToolCall(tool_name="slow_tool_b", parameters={"x": 20}, id="c2"),
            ToolCall(tool_name="slow_tool_c", parameters={"x": 30}, id="c3"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a, slow_tool_b, slow_tool_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        agent.run([Message(role=Role.USER, content="go")])

        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 3
        assert tool_results[0].content == "a=10"
        assert tool_results[1].content == "b=20"
        assert tool_results[2].content == "c=30"
        assert tool_results[0].tool_call_id == "c1"
        assert tool_results[1].tool_call_id == "c2"
        assert tool_results[2].tool_call_id == "c3"

    @pytest.mark.asyncio
    async def test_result_order_preserved_async(self) -> None:
        """Tool results appear in history in the same order as the tool calls."""
        calls = [
            ToolCall(tool_name="async_slow_a", parameters={"x": 10}, id="c1"),
            ToolCall(tool_name="async_slow_b", parameters={"x": 20}, id="c2"),
            ToolCall(tool_name="async_slow_c", parameters={"x": 30}, id="c3"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[async_slow_a, async_slow_b, async_slow_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        await agent.arun([Message(role=Role.USER, content="go")])

        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 3
        assert tool_results[0].content == "async_a=10"
        assert tool_results[1].content == "async_b=20"
        assert tool_results[2].content == "async_c=30"


class TestParallelErrorHandling:
    """One tool failure should not prevent other tools from running."""

    def test_partial_failure_sync(self) -> None:
        """If one of three tools fails, the other two still succeed."""
        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="failing_tool", parameters={"x": 0}, id="c2"),
            ToolCall(tool_name="slow_tool_c", parameters={"x": 3}, id="c3"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a, failing_tool, slow_tool_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = agent.run([Message(role=Role.USER, content="go")])

        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 3
        assert tool_results[0].content == "a=1"
        assert "Error" in tool_results[1].content
        assert "boom" in tool_results[1].content
        assert tool_results[2].content == "c=3"
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_partial_failure_async(self) -> None:
        """If one of three async tools fails, the other two still succeed."""
        calls = [
            ToolCall(tool_name="async_slow_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="async_failing_tool", parameters={"x": 0}, id="c2"),
            ToolCall(tool_name="async_slow_c", parameters={"x": 3}, id="c3"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[async_slow_a, async_failing_tool, async_slow_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = await agent.arun([Message(role=Role.USER, content="go")])

        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 3
        assert tool_results[0].content == "async_a=1"
        assert "Error" in tool_results[1].content
        assert "async_boom" in tool_results[1].content
        assert tool_results[2].content == "async_c=3"


class TestParallelUnknownTool:
    """Unknown tool names should produce error messages without crashing."""

    def test_unknown_tool_in_parallel_batch(self) -> None:
        """Unknown tools produce error messages, other tools still execute."""
        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="nonexistent_tool", parameters={"x": 0}, id="c2"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = agent.run([Message(role=Role.USER, content="go")])

        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 2
        assert tool_results[0].content == "a=1"
        assert "Unknown tool" in tool_results[1].content


class TestParallelHooks:
    """Verify hooks fire for all tools in a parallel batch."""

    def test_hooks_fire_for_all_parallel_tools(self) -> None:
        """on_tool_start and on_tool_end should fire for each parallel tool."""
        started: List[str] = []
        ended: List[str] = []

        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="slow_tool_b", parameters={"x": 2}, id="c2"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a, slow_tool_b],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                parallel_tool_execution=True,
                hooks={
                    "on_tool_start": lambda name, args: started.append(name),
                    "on_tool_end": lambda name, result, dur: ended.append(name),
                },
            ),
        )
        agent.run([Message(role=Role.USER, content="go")])

        assert "slow_tool_a" in started
        assert "slow_tool_b" in started
        assert "slow_tool_a" in ended
        assert "slow_tool_b" in ended

    def test_error_hooks_fire_for_failed_parallel_tools(self) -> None:
        """on_tool_error fires for failed tools in a parallel batch."""
        errors: List[str] = []

        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="failing_tool", parameters={"x": 0}, id="c2"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a, failing_tool],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                parallel_tool_execution=True,
                hooks={
                    "on_tool_error": lambda name, exc, args: errors.append(name),
                },
            ),
        )
        agent.run([Message(role=Role.USER, content="go")])

        assert "failing_tool" in errors
        assert "slow_tool_a" not in errors


class TestSingleToolNoParallel:
    """Single tool call should not use the parallel path (optimization)."""

    def test_single_tool_uses_sequential_path(self) -> None:
        """A single tool call should work fine even with parallel_tool_execution=True."""
        calls = [ToolCall(tool_name="slow_tool_a", parameters={"x": 42}, id="c1")]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = agent.run([Message(role=Role.USER, content="go")])

        assert result.content == "Done."
        tool_results = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_results) == 1
        assert tool_results[0].content == "a=42"


class TestParallelToolCallTracking:
    """Verify all_tool_calls in AgentResult contains all parallel tool calls."""

    def test_all_tool_calls_tracked(self) -> None:
        """AgentResult.tool_calls should contain all tool calls from the batch."""
        calls = [
            ToolCall(tool_name="slow_tool_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="slow_tool_b", parameters={"x": 2}, id="c2"),
            ToolCall(tool_name="slow_tool_c", parameters={"x": 3}, id="c3"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[slow_tool_a, slow_tool_b, slow_tool_c],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = agent.run([Message(role=Role.USER, content="go")])

        assert len(result.tool_calls) == 3
        names = [tc.tool_name for tc in result.tool_calls]
        assert names == ["slow_tool_a", "slow_tool_b", "slow_tool_c"]

    @pytest.mark.asyncio
    async def test_all_tool_calls_tracked_async(self) -> None:
        """AgentResult.tool_calls should contain all async parallel tool calls."""
        calls = [
            ToolCall(tool_name="async_slow_a", parameters={"x": 1}, id="c1"),
            ToolCall(tool_name="async_slow_b", parameters={"x": 2}, id="c2"),
        ]
        provider = MultiToolProvider(calls)
        agent = Agent(
            tools=[async_slow_a, async_slow_b],
            provider=provider,
            config=AgentConfig(max_iterations=3, parallel_tool_execution=True),
        )
        result = await agent.arun([Message(role=Role.USER, content="go")])

        assert len(result.tool_calls) == 2
        names = [tc.tool_name for tc in result.tool_calls]
        assert names == ["async_slow_a", "async_slow_b"]
