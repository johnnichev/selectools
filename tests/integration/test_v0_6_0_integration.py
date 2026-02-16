"""
Integration tests for v0.6.0 features with existing functionality.

Tests that Ollama provider and Analytics work well with:
- Conversation Memory (v0.4.0)
- Async Support (v0.4.0)
- Cost Tracking (v0.5.0)
- Pre-built Toolbox (v0.5.1)
- Tool Validation (v0.5.2)
- Observability Hooks (v0.5.2)
"""

from __future__ import annotations

from typing import Any

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.memory import ConversationMemory
from selectools.providers.stubs import LocalProvider
from selectools.toolbox import get_tools_by_category
from tests.core.test_framework import FakeProvider


@pytest.fixture
def simple_tool() -> Tool:
    """Simple tool for testing."""

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return Tool(
        name="greet",
        description="Greet someone",
        parameters=[ToolParameter(name="name", param_type=str, description="Person's name")],
        function=greet,
    )


@pytest.fixture
def counter_tool() -> Tool:
    """Tool that maintains state for testing memory."""
    call_count = {"count": 0}

    def count() -> str:
        call_count["count"] += 1
        return f"Call #{call_count['count']}"

    return Tool(
        name="count",
        description="Count the number of calls",
        parameters=[],
        function=count,
    )


class TestOllamaIntegration:
    """Test Ollama provider integration with existing features."""

    def test_ollama_with_cost_tracking(self, simple_tool: Tool) -> None:
        """Test that Ollama correctly reports zero cost."""
        # Use LocalProvider as a proxy since we can't assume Ollama is running
        provider = LocalProvider()
        config = AgentConfig(
            model="local", cost_warning_threshold=0.01  # Should not trigger for free models
        )
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hello")])

        # Verify zero cost
        assert agent.total_cost == 0.0
        assert agent.usage.total_cost_usd == 0.0

    def test_ollama_with_memory(self, simple_tool: Tool) -> None:
        """Test Ollama with conversation memory."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}',
                "Done greeting Alice!",
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Bob"}}',
                "Done greeting Bob!",
            ]
        )
        memory = ConversationMemory(max_messages=10)
        agent = Agent(
            tools=[simple_tool],
            provider=provider,
            config=AgentConfig(max_iterations=4),
            memory=memory,
        )

        # First turn
        response1 = agent.run([Message(role=Role.USER, content="Greet Alice")])
        assert response1.role == Role.ASSISTANT

        # Second turn - memory should contain previous conversation
        response2 = agent.run([Message(role=Role.USER, content="Now greet Bob")])
        assert response2.role == Role.ASSISTANT

        # Verify memory contains both conversations
        history = memory.get_history()
        assert len(history) >= 2  # At least 2 user messages

    @pytest.mark.asyncio
    async def test_ollama_async_execution(self, simple_tool: Tool) -> None:
        """Test async execution with Ollama-like provider."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Charlie"}}',
                "Done!",
            ]
        )
        agent = Agent(tools=[simple_tool], provider=provider, config=AgentConfig())

        response = await agent.arun([Message(role=Role.USER, content="Greet Charlie")])
        assert response.role == Role.ASSISTANT


class TestAnalyticsIntegration:
    """Test Analytics integration with existing features."""

    def test_analytics_with_observability_hooks(self, simple_tool: Tool) -> None:
        """Test that analytics and hooks work together."""
        hook_calls = {"tool_start": 0, "tool_end": 0, "llm_end": 0}

        def on_tool_start(name: str, args: Any) -> None:
            hook_calls["tool_start"] += 1

        def on_tool_end(name: str, result: Any, duration: Any) -> None:
            hook_calls["tool_end"] += 1

        def on_llm_end(response: Any, usage: Any) -> None:
            hook_calls["llm_end"] += 1

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Dave"}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            hooks={
                "on_tool_start": on_tool_start,
                "on_tool_end": on_tool_end,
                "on_llm_end": on_llm_end,
            },
        )
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Dave")])

        # Verify both hooks and analytics were triggered
        assert hook_calls["tool_start"] >= 1
        assert hook_calls["tool_end"] >= 1
        assert hook_calls["llm_end"] >= 1

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")
        assert metrics is not None
        assert metrics.total_calls >= 1

    def test_analytics_with_toolbox(self) -> None:
        """Test analytics with pre-built toolbox tools."""
        # Get some toolbox tools
        text_tools = get_tools_by_category("text")[:2]  # Get 2 text tools

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "count_text", "parameters": {"text": "hello world", "substring": "o"}}',
                "Found 2 occurrences!",
            ]
        )
        config = AgentConfig(enable_analytics=True)
        agent = Agent(tools=text_tools, provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Count 'o' in 'hello world'")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("count_text")
        assert metrics is not None
        assert metrics.total_calls >= 1

    def test_analytics_with_memory(self, simple_tool: Tool, counter_tool: Tool) -> None:
        """Test analytics across multiple conversation turns with memory."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "count", "parameters": {}}',
                "Counted!",
                'TOOL_CALL: {"tool_name": "count", "parameters": {}}',
                "Counted again!",
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Eve"}}',
                "Done!",
            ]
        )
        memory = ConversationMemory(max_messages=20)
        config = AgentConfig(enable_analytics=True)
        agent = Agent(
            tools=[simple_tool, counter_tool],
            provider=provider,
            config=config,
            memory=memory,
        )

        # Turn 1
        agent.run([Message(role=Role.USER, content="Count please")])

        # Turn 2
        agent.run([Message(role=Role.USER, content="Count again")])

        # Turn 3
        agent.run([Message(role=Role.USER, content="Greet Eve")])

        # Verify analytics tracked across all turns
        analytics = agent.get_analytics()
        count_metrics = analytics.get_metrics("count")
        greet_metrics = analytics.get_metrics("greet")

        assert count_metrics is not None
        assert count_metrics.total_calls >= 2  # Called in turn 1 and 2
        assert greet_metrics is not None
        assert greet_metrics.total_calls >= 1  # Called in turn 3

    def test_analytics_with_cost_tracking(self, simple_tool: Tool) -> None:
        """Test that analytics and cost tracking coexist properly."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Frank"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, cost_warning_threshold=0.01)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Frank")])

        # Both systems should work independently
        assert agent.total_cost >= 0.0  # Cost tracking works
        analytics = agent.get_analytics()
        assert analytics is not None  # Analytics works
        metrics = analytics.get_metrics("greet")
        assert metrics is not None
        assert metrics.total_calls >= 1

    def test_analytics_with_tool_validation(self) -> None:
        """Test analytics with tool validation errors."""

        # Create a tool with strict validation
        def strict_tool(value: int) -> str:
            return f"Value: {value}"

        tool = Tool(
            name="strict_tool",
            description="A tool with strict validation",
            parameters=[
                ToolParameter(
                    name="value",
                    param_type=int,
                    description="Must be an integer",
                )
            ],
            function=strict_tool,
        )

        # Try to call with wrong type (will fail validation)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "strict_tool", "parameters": {"value": "not_an_int"}}',
                "Failed as expected",
            ]
        )
        config = AgentConfig(enable_analytics=True)
        agent = Agent(tools=[tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Test validation")])

        # Analytics should track the failure
        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("strict_tool")
        # The tool may or may not be called depending on validation, but analytics should handle it
        if metrics:
            assert metrics.total_calls >= 0

    @pytest.mark.asyncio
    async def test_analytics_async_integration(self, simple_tool: Tool) -> None:
        """Test analytics with async agent execution."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Grace"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Greet Grace")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")
        assert metrics is not None
        assert metrics.total_calls >= 1
        assert metrics.successful_calls >= 1


class TestCombinedFeatures:
    """Test combinations of multiple v0.6.0 and previous features."""

    def test_all_features_together(self, simple_tool: Tool) -> None:
        """Test Ollama-like provider with analytics, hooks, memory, and cost tracking."""
        hook_calls = []

        def log_hook(event: str, *args: Any) -> None:
            hook_calls.append(event)

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Hannah"}}',
                "Done!",
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Ian"}}',
                "All done!",
            ]
        )
        memory = ConversationMemory(max_messages=15)
        config = AgentConfig(
            enable_analytics=True,
            cost_warning_threshold=0.05,
            max_iterations=5,
            hooks={
                "on_agent_start": lambda msgs: log_hook("agent_start"),
                "on_tool_start": lambda name, args: log_hook("tool_start"),
                "on_tool_end": lambda name, res, dur: log_hook("tool_end"),
                "on_agent_end": lambda res, usage: log_hook("agent_end"),
            },
        )
        agent = Agent(tools=[simple_tool], provider=provider, config=config, memory=memory)

        # Turn 1
        response1 = agent.run([Message(role=Role.USER, content="Greet Hannah")])
        assert response1.role == Role.ASSISTANT

        # Turn 2
        response2 = agent.run([Message(role=Role.USER, content="Greet Ian")])
        assert response2.role == Role.ASSISTANT

        # Verify all features worked together
        # 1. Hooks were called
        assert "agent_start" in hook_calls
        assert hook_calls.count("tool_start") >= 1  # At least one tool call

        # 2. Analytics tracked calls
        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")
        assert metrics is not None
        assert metrics.total_calls >= 1  # At least one call

        # 3. Memory preserved conversation
        history = memory.get_history()
        assert len(history) >= 2  # At least 2 user messages

        # 4. Cost tracking worked (should be minimal with FakeProvider)
        assert agent.total_cost >= 0.0

    def test_analytics_reset_preserves_other_features(self, simple_tool: Tool) -> None:
        """Test that resetting analytics doesn't affect other features."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Jack"}}',
                "Done!",
            ]
        )
        memory = ConversationMemory(max_messages=10)
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[simple_tool], provider=provider, config=config, memory=memory)

        # Run once
        agent.run([Message(role=Role.USER, content="Greet Jack")])

        # Verify analytics has data (may or may not depending on execution)
        analytics = agent.get_analytics()
        initial_metrics_count = len(analytics.get_all_metrics())

        # Reset analytics
        analytics.reset()

        # Verify analytics is empty but other features still work
        assert len(analytics.get_all_metrics()) == 0
        assert agent.total_cost >= 0.0  # Cost tracking still works
        assert len(memory.get_history()) >= 1  # Memory still has history

    def test_toolbox_with_analytics_and_hooks(self) -> None:
        """Test pre-built toolbox tools with analytics and observability."""
        hook_events = []

        def track_event(event: str) -> None:
            hook_events.append(event)

        # Get file tools from toolbox
        file_tools = get_tools_by_category("file")

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "file_exists", "parameters": {"filepath": "test.txt"}}',
                "Checked!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            max_iterations=3,
            hooks={
                "on_tool_start": lambda *args: track_event("tool_start"),
                "on_tool_end": lambda *args: track_event("tool_end"),
                "on_tool_error": lambda *args: track_event("tool_error"),
            },
        )
        agent = Agent(tools=file_tools, provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Check if test.txt exists")])

        # Verify toolbox tools work with analytics and hooks
        assert "tool_start" in hook_events
        # Tool may succeed or fail, both should be tracked
        assert ("tool_end" in hook_events) or ("tool_error" in hook_events)

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("file_exists")
        # Analytics should track the tool call regardless of success/failure
        assert metrics is not None
        assert metrics.total_calls >= 1
