"""
Integration tests for streaming tools with other features.

Tests streaming integration with:
- Observability hooks
- Analytics tracking
- Conversation memory
- Cost tracking
- Tool validation
- Multiple features combined
"""

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.providers.stubs import LocalProvider
from tests.core.test_framework import FakeProvider


@tool(description="Simple streaming tool", streaming=True)
def stream_numbers(count: int) -> Generator[str, None, None]:
    """Yield numbers as chunks."""
    for i in range(count):
        yield f"{i} "


@tool(description="Non-streaming helper tool")
def get_info(name: str) -> str:
    """Return info about a name."""
    return f"Info about {name}"


class TestStreamingWithHooks:
    """Test streaming integration with observability hooks."""

    def test_streaming_with_all_hooks(self):
        """Test streaming with complete set of hooks."""
        hook_events = []

        def track(event_name, *args, **kwargs):
            hook_events.append(event_name)

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 5}}',
                "Finished!",
            ]
        )
        config = AgentConfig(
            hooks={
                "on_agent_start": lambda *a, **k: track("agent_start"),
                "on_tool_start": lambda *a, **k: track("tool_start"),
                "on_tool_chunk": lambda *a, **k: track("tool_chunk"),
                "on_tool_end": lambda *a, **k: track("tool_end"),
                "on_agent_end": lambda *a, **k: track("agent_end"),
            },
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 5 numbers")])

        assert "agent_start" in hook_events
        assert "tool_start" in hook_events
        assert hook_events.count("tool_chunk") == 5  # Should have 5 chunks
        assert "tool_end" in hook_events
        assert "agent_end" in hook_events

    @pytest.mark.asyncio
    async def test_streaming_with_async_hooks(self):
        """Test streaming works with async agent and hooks."""
        hook_events = []

        def track(event_name, *args, **kwargs):
            hook_events.append(event_name)

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            hooks={
                "on_tool_chunk": lambda *a, **k: track("chunk"),
            },
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Stream numbers")])

        assert hook_events.count("chunk") == 3


class TestStreamingWithAnalytics:
    """Test streaming integration with analytics."""

    def test_streaming_analytics_with_observability(self):
        """Test analytics and hooks work together for streaming."""
        chunks_received = []

        def on_chunk(tool_name, chunk):
            chunks_received.append(chunk)

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 4}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            hooks={"on_tool_chunk": on_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 4")])

        # Check analytics
        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("stream_numbers")
        assert metrics.total_chunks == 4
        assert metrics.streaming_calls == 1

        # Check hooks
        assert len(chunks_received) == 4

    def test_streaming_analytics_summary_shows_chunks(self):
        """Test that analytics summary includes streaming info."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 10}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=2)
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 10")])

        analytics = agent.get_analytics()
        summary = analytics.summary()

        assert "Streaming calls:" in summary
        assert "Total chunks:" in summary
        assert "10" in summary  # Should show 10 chunks


class TestStreamingWithMemory:
    """Test streaming integration with conversation memory."""

    def test_streaming_with_memory_context(self):
        """Test streaming tools work with conversation memory."""
        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Streamed numbers.",
            ]
        )
        config = AgentConfig(max_iterations=2)
        agent = Agent(tools=[stream_numbers], provider=provider, config=config, memory=memory)

        response = agent.run([Message(role=Role.USER, content="Stream some numbers")])

        # Memory should contain the interaction
        history = memory.get_history()
        assert len(history) > 0
        assert any(msg.role == Role.USER for msg in history)
        assert any(msg.role == Role.ASSISTANT for msg in history)

    @pytest.mark.asyncio
    async def test_streaming_memory_multi_turn(self):
        """Test streaming with memory across multiple turns."""
        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 2}}',
                "First stream done.",
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Second stream done.",
            ]
        )
        config = AgentConfig(max_iterations=2)
        agent = Agent(tools=[stream_numbers], provider=provider, config=config, memory=memory)

        # First turn
        await agent.arun([Message(role=Role.USER, content="Stream 2")])

        # Second turn
        await agent.arun([Message(role=Role.USER, content="Stream 3")])

        # Memory should have both interactions
        history = memory.get_history()
        assert len(history) >= 4  # At least 2 user + 2 assistant messages


class TestStreamingWithCostTracking:
    """Test streaming integration with cost tracking."""

    def test_streaming_cost_attribution(self):
        """Test that streaming tools work with cost tracking."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 5}}',
                "Done!",
            ]
        )
        config = AgentConfig(max_iterations=2)
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 5")])

        # Should have accumulated some cost (from FakeProvider)
        assert agent.total_cost > 0
        assert agent.usage.total_cost_usd > 0


class TestStreamingCombinedFeatures:
    """Test streaming with multiple features enabled simultaneously."""

    def test_all_features_with_streaming(self):
        """Test streaming with analytics, hooks, memory, and cost tracking."""
        hook_events = []

        def track_event(event_name, *args, **kwargs):
            hook_events.append(event_name)

        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 5}}',
                'TOOL_CALL: {"tool_name": "get_info", "parameters": {"name": "test"}}',
                "All done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            hooks={
                "on_agent_start": lambda *a, **k: track_event("agent_start"),
                "on_tool_start": lambda *a, **k: track_event("tool_start"),
                "on_tool_chunk": lambda *a, **k: track_event("tool_chunk"),
                "on_tool_end": lambda *a, **k: track_event("tool_end"),
                "on_agent_end": lambda *a, **k: track_event("agent_end"),
            },
            max_iterations=5,
        )
        agent = Agent(
            tools=[stream_numbers, get_info], provider=provider, config=config, memory=memory
        )

        response = agent.run([Message(role=Role.USER, content="Run both tools")])

        # Verify hooks
        assert "agent_start" in hook_events
        assert hook_events.count("tool_start") == 2  # Both tools called
        assert hook_events.count("tool_chunk") == 5  # Only streaming tool has chunks
        assert hook_events.count("tool_end") == 2
        assert "agent_end" in hook_events

        # Verify analytics
        analytics = agent.get_analytics()
        stream_metrics = analytics.get_metrics("stream_numbers")
        info_metrics = analytics.get_metrics("get_info")

        assert stream_metrics.total_chunks == 5
        assert stream_metrics.streaming_calls == 1
        assert info_metrics.total_chunks == 0  # Non-streaming
        assert info_metrics.streaming_calls == 0

        # Verify memory
        history = memory.get_history()
        assert len(history) > 0

        # Verify cost tracking
        assert agent.total_cost > 0

    @pytest.mark.asyncio
    async def test_all_features_async_streaming(self):
        """Test all features with async streaming."""

        @tool(description="Async streaming tool", streaming=True)
        async def async_stream(count: int) -> AsyncGenerator[str, None]:
            for i in range(count):
                await asyncio.sleep(0.001)
                yield f"Async{i} "

        chunks = []

        def on_chunk(tool_name, chunk):
            chunks.append(chunk)

        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "async_stream", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            hooks={"on_tool_chunk": on_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[async_stream], provider=provider, config=config, memory=memory)

        await agent.arun([Message(role=Role.USER, content="Async stream")])

        # Verify all features work
        assert len(chunks) == 3
        assert agent.get_analytics().get_metrics("async_stream").total_chunks == 3
        assert len(memory.get_history()) > 0
        assert agent.total_cost > 0

    def test_streaming_reset_analytics_preserves_other_features(self):
        """Test resetting analytics doesn't affect streaming or other features."""
        chunks = []

        def on_chunk(tool_name, chunk):
            chunks.append(chunk)

        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            hooks={"on_tool_chunk": on_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config, memory=memory)

        agent.run([Message(role=Role.USER, content="Stream")])

        # Verify analytics has data
        analytics = agent.get_analytics()
        assert analytics.get_metrics("stream_numbers").total_chunks == 3

        # Reset analytics
        analytics.reset()

        # Verify streaming still works but analytics is empty
        assert len(analytics.get_all_metrics()) == 0
        assert len(chunks) == 3  # Hooks still worked
        assert len(memory.get_history()) > 0  # Memory still has data


class TestStreamingToolboxIntegration:
    """Test streaming toolbox tools with full feature set."""

    def test_toolbox_streaming_with_analytics(self, tmp_path):
        """Test toolbox streaming tools work with analytics."""
        from selectools.toolbox.file_tools import read_file_stream

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")

        provider = FakeProvider(
            responses=[
                f'TOOL_CALL: {{"tool_name": "read_file_stream", "parameters": {{"filepath": "{test_file}"}}}}',
                "File read!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=2)
        agent = Agent(tools=[read_file_stream], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Read file")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("read_file_stream")

        assert metrics is not None
        assert metrics.streaming_calls == 1
        assert metrics.total_chunks > 0  # Should have multiple chunks

    def test_toolbox_csv_streaming_with_hooks(self, tmp_path):
        """Test CSV streaming with hooks."""
        from selectools.toolbox.data_tools import process_csv_stream

        # Create test CSV
        test_csv = tmp_path / "test.csv"
        test_csv.write_text("name,value\nA,1\nB,2\n")

        chunks = []

        def on_chunk(tool_name, chunk):
            chunks.append(chunk)

        provider = FakeProvider(
            responses=[
                f'TOOL_CALL: {{"tool_name": "process_csv_stream", "parameters": {{"filepath": "{test_csv}"}}}}',
                "CSV processed!",
            ]
        )
        config = AgentConfig(hooks={"on_tool_chunk": on_chunk}, max_iterations=2)
        agent = Agent(tools=[process_csv_stream], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Process CSV")])

        # Should have received chunks
        assert len(chunks) > 0
        combined = "".join(chunks)
        assert "name" in combined
        assert "value" in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
