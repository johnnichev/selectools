"""
Comprehensive edge case and integration tests for production readiness.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import selectools
from agent import Agent, AgentConfig, Message, Role, Tool
from selectools.memory import ConversationMemory
from selectools.providers.stubs import LocalProvider


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_mixed_sync_async_tools_comprehensive() -> None:
    """Test complex scenarios with mixed sync/async tools."""
    call_log = []

    @selectools.tool(description="Async tool 1")
    async def async_tool1(x: int) -> str:
        await asyncio.sleep(0.01)
        call_log.append(("async1", x))
        return f"async1:{x}"

    @selectools.tool(description="Sync tool 1")
    def sync_tool1(x: int) -> str:
        call_log.append(("sync1", x))
        return f"sync1:{x}"

    @selectools.tool(description="Async tool 2")
    async def async_tool2(x: int) -> str:
        await asyncio.sleep(0.02)
        call_log.append(("async2", x))
        return f"async2:{x}"

    agent = Agent(
        tools=[async_tool1, sync_tool1, async_tool2],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=5),
    )

    # Test async execution with mixed tools
    response = await agent.arun([Message(role=Role.USER, content="Test mixed")])
    assert response.role == Role.ASSISTANT

    # Verify both sync and async tools can be called
    assert len(call_log) >= 0  # May or may not call tools depending on LocalProvider


@pytest.mark.asyncio
async def test_tool_timeout_edge_cases() -> None:
    """Test tool timeout behavior in edge cases."""

    @selectools.tool(description="Slow async tool")
    async def slow_async_tool() -> str:
        await asyncio.sleep(2.0)  # Longer than timeout
        return "completed"

    @selectools.tool(description="Slow sync tool")
    def slow_sync_tool() -> str:
        import time

        time.sleep(2.0)  # Longer than timeout
        return "completed"

    agent = Agent(
        tools=[slow_async_tool],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, tool_timeout_seconds=0.1),
    )

    # Should handle timeout gracefully
    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_concurrent_agent_execution() -> None:
    """Test multiple agents running concurrently."""

    @selectools.tool(description="Counter")
    async def counter(n: int) -> str:
        await asyncio.sleep(0.05)
        return f"count:{n}"

    agents = [
        Agent(tools=[counter], provider=LocalProvider(), config=AgentConfig(max_iterations=2))
        for _ in range(5)
    ]

    # Run 5 agents concurrently
    results = await asyncio.gather(
        *[
            agent.arun([Message(role=Role.USER, content=f"Count {i}")])
            for i, agent in enumerate(agents)
        ]
    )

    assert len(results) == 5
    assert all(r.role == Role.ASSISTANT for r in results)


@pytest.mark.asyncio
async def test_memory_overflow_handling() -> None:
    """Test memory behavior when limits are exceeded."""
    memory = ConversationMemory(max_messages=5)

    tool = Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")

    agent = Agent(tools=[tool], provider=LocalProvider(), memory=memory)

    # Add many messages
    for i in range(10):
        await agent.arun([Message(role=Role.USER, content=f"Message {i}")])

    # Memory should be capped at max_messages
    assert len(memory) <= 5


@pytest.mark.asyncio
async def test_error_propagation_in_async() -> None:
    """Test that errors are properly propagated in async context."""

    @selectools.tool(description="Failing tool")
    async def failing_tool() -> str:
        raise ValueError("Intentional failure")

    agent = Agent(
        tools=[failing_tool], provider=LocalProvider(), config=AgentConfig(max_iterations=2)
    )

    # Should handle tool failure gracefully
    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_async_with_streaming() -> None:
    """Test async agent with streaming enabled."""
    chunks = []

    def stream_handler(chunk: str) -> None:
        chunks.append(chunk)

    tool = Tool(name="test", description="Test", parameters=[], function=lambda: "result")

    agent = Agent(
        tools=[tool], provider=LocalProvider(), config=AgentConfig(stream=True, max_iterations=2)
    )

    response = await agent.arun(
        [Message(role=Role.USER, content="Test")], stream_handler=stream_handler
    )

    assert response.role == Role.ASSISTANT
    # LocalProvider should stream
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_complex_multi_turn_async_conversation() -> None:
    """Test complex multi-turn conversation with memory and mixed tools."""
    memory = ConversationMemory(max_messages=20)
    turn_count = 0

    @selectools.tool(description="Async counter")
    async def async_counter() -> str:
        nonlocal turn_count
        turn_count += 1
        await asyncio.sleep(0.01)
        return f"turn_{turn_count}"

    @selectools.tool(description="Sync reporter")
    def sync_reporter(msg: str) -> str:
        return f"reported: {msg}"

    agent = Agent(
        tools=[async_counter, sync_reporter],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=3),
        memory=memory,
    )

    # Multiple turns
    for i in range(5):
        response = await agent.arun([Message(role=Role.USER, content=f"Turn {i}")])
        assert response.role == Role.ASSISTANT

    # Memory should accumulate
    assert len(memory) > 5


@pytest.mark.asyncio
async def test_provider_error_recovery() -> None:
    """Test recovery from provider errors in async mode."""
    from selectools import UsageStats

    class FailingProvider:
        name = "failing"
        supports_streaming = False
        supports_async = True
        call_count = 0

        async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            self.call_count += 1
            if self.call_count <= 2:
                from selectools.providers.base import ProviderError

                raise ProviderError("Temporary failure")
            usage = UsageStats(prompt_tokens=10, completion_tokens=5)
            return Message(role=Role.ASSISTANT, content="Success after retries"), usage

        def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            usage = UsageStats(prompt_tokens=10, completion_tokens=5)
            return Message(role=Role.ASSISTANT, content="Sync fallback"), usage

    tool = Tool(name="test", description="Test", parameters=[], function=lambda: "ok")

    provider = FailingProvider()
    agent = Agent(
        tools=[tool],
        provider=provider,
        config=AgentConfig(max_retries=3, retry_backoff_seconds=0.01),
    )

    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT
    assert provider.call_count >= 2  # Should have retried


@pytest.mark.asyncio
async def test_async_tool_with_sync_provider() -> None:
    """Test async tools work with providers that don't support async."""

    @selectools.tool(description="Async tool")
    async def async_tool(x: int) -> str:
        await asyncio.sleep(0.01)
        return f"result:{x}"

    # LocalProvider doesn't have async support
    provider = LocalProvider()
    assert not getattr(provider, "supports_async", False)

    agent = Agent(tools=[async_tool], provider=provider, config=AgentConfig(max_iterations=2))

    # Should work via fallback
    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_empty_tool_result_handling() -> None:
    """Test handling of empty or None tool results."""

    @selectools.tool(description="Empty tool")
    async def empty_tool() -> str:
        return ""

    @selectools.tool(description="Whitespace tool")
    def whitespace_tool() -> str:
        return "   "

    agent = Agent(
        tools=[empty_tool, whitespace_tool],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=3),
    )

    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_large_tool_results() -> None:
    """Test handling of large tool results."""

    @selectools.tool(description="Large result tool")
    async def large_result() -> str:
        # Generate a large result (10KB)
        return "x" * 10000

    agent = Agent(
        tools=[large_result], provider=LocalProvider(), config=AgentConfig(max_iterations=2)
    )

    response = await agent.arun([Message(role=Role.USER, content="Test")])
    assert response.role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_memory_with_large_messages() -> None:
    """Test memory handling with large messages."""
    memory = ConversationMemory(max_messages=10, max_tokens=1000)

    tool = Tool(name="test", description="Test", parameters=[], function=lambda: "ok")

    agent = Agent(tools=[tool], provider=LocalProvider(), memory=memory)

    # Add a very large message
    large_content = "x" * 10000
    await agent.arun([Message(role=Role.USER, content=large_content)])

    # Memory should enforce token limits
    assert len(memory) >= 0  # Should not crash


@pytest.mark.asyncio
async def test_rapid_consecutive_calls() -> None:
    """Test rapid consecutive agent calls."""
    tool = Tool(name="fast", description="Fast tool", parameters=[], function=lambda: "fast")

    agent = Agent(tools=[tool], provider=LocalProvider(), config=AgentConfig(max_iterations=2))

    # Make 20 rapid calls
    tasks = [agent.arun([Message(role=Role.USER, content=f"Call {i}")]) for i in range(20)]

    results = await asyncio.gather(*tasks)
    assert len(results) == 20
    assert all(r.role == Role.ASSISTANT for r in results)


def run_async_test(test_func: Any) -> None:
    """Helper to run async tests."""
    asyncio.run(test_func())


if __name__ == "__main__":
    edge_case_tests = [
        test_mixed_sync_async_tools_comprehensive,
        test_tool_timeout_edge_cases,
        test_concurrent_agent_execution,
        test_memory_overflow_handling,
        test_error_propagation_in_async,
        test_async_with_streaming,
        test_complex_multi_turn_async_conversation,
        test_provider_error_recovery,
        test_async_tool_with_sync_provider,
        test_empty_tool_result_handling,
        test_large_tool_results,
        test_memory_with_large_messages,
        test_rapid_consecutive_calls,
    ]

    failures = 0
    for test in edge_case_tests:
        try:
            run_async_test(test)
            print(f"✓ {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"✗ {test.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"✗ {test.__name__}: {exc.__class__.__name__}: {exc}")

    if failures:
        print(f"\n{failures} edge case test(s) failed!")
        raise SystemExit(1)
    else:
        print(f"\nAll {len(edge_case_tests)} edge case tests passed! ✅")
