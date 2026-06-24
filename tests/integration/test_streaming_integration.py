"""
Integration tests for streaming tools with other features.

Tests streaming integration with:
- Observers
- Analytics tracking
- Conversation memory
- Cost tracking
- Tool validation
- Multiple features combined
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.observer import AgentObserver
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


class _EventRecorder(AgentObserver):
    """Records lifecycle event tags in order."""

    def __init__(self, sink: list) -> None:
        self.sink = sink

    def on_run_start(self, run_id, messages, system_prompt) -> None:
        self.sink.append("agent_start")

    def on_run_end(self, run_id, result) -> None:
        self.sink.append("agent_end")

    def on_tool_start(self, run_id, call_id, tool_name, tool_args) -> None:
        self.sink.append("tool_start")

    def on_tool_chunk(self, run_id, call_id, tool_name, chunk) -> None:
        self.sink.append("tool_chunk")

    def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms) -> None:
        self.sink.append("tool_end")


class _ChunkRecorder(AgentObserver):
    """Records raw chunks from on_tool_chunk events."""

    def __init__(self, sink: list) -> None:
        self.sink = sink

    def on_tool_chunk(self, run_id, call_id, tool_name, chunk) -> None:
        self.sink.append(chunk)


class TestStreamingWithObservers:
    """Test streaming integration with observers."""

    def test_streaming_with_all_events(self) -> None:
        """Test streaming with the complete set of lifecycle events."""
        hook_events: list = []

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 5}}',
                "Finished!",
            ]
        )
        config = AgentConfig(
            observers=[_EventRecorder(hook_events)],
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
    async def test_streaming_with_async_observers(self) -> None:
        """Test streaming works with async agent and observers."""
        hook_events: list = []

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            observers=[_ChunkRecorder(hook_events)],
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Stream numbers")])

        assert len(hook_events) == 3


class TestStreamingWithAnalytics:
    """Test streaming integration with analytics."""

    def test_streaming_analytics_with_observability(self) -> None:
        """Test analytics and observers work together for streaming."""
        chunks_received: list = []

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 4}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            observers=[_ChunkRecorder(chunks_received)],
            max_iterations=2,
        )
        agent = Agent(tools=[stream_numbers], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 4")])

        # Check analytics
        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("stream_numbers")
        assert metrics.total_chunks == 4
        assert metrics.streaming_calls == 1

        # Check observer events
        assert len(chunks_received) == 4

    def test_streaming_analytics_summary_shows_chunks(self) -> None:
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

    def test_streaming_with_memory_context(self) -> None:
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

        agent.run([Message(role=Role.USER, content="Stream some numbers")])

        # Memory should contain the interaction
        history = memory.get_history()
        assert len(history) > 0
        assert any(msg.role == Role.USER for msg in history)
        assert any(msg.role == Role.ASSISTANT for msg in history)

    @pytest.mark.asyncio
    async def test_streaming_memory_multi_turn(self) -> None:
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

    def test_streaming_cost_attribution(self) -> None:
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

    def test_all_features_with_streaming(self) -> None:
        """Test streaming with analytics, observers, memory, and cost tracking."""
        hook_events: list = []

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
            observers=[_EventRecorder(hook_events)],
            max_iterations=5,
        )
        agent = Agent(
            tools=[stream_numbers, get_info], provider=provider, config=config, memory=memory
        )

        agent.run([Message(role=Role.USER, content="Run both tools")])

        # Verify observer events
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
    async def test_all_features_async_streaming(self) -> None:
        """Test all features with async streaming."""

        @tool(description="Async streaming tool", streaming=True)
        async def async_stream(count: int) -> AsyncGenerator[str, None]:
            for i in range(count):
                await asyncio.sleep(0.001)
                yield f"Async{i} "

        chunks: list = []

        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "async_stream", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            observers=[_ChunkRecorder(chunks)],
            max_iterations=2,
        )
        agent = Agent(tools=[async_stream], provider=provider, config=config, memory=memory)

        await agent.arun([Message(role=Role.USER, content="Async stream")])

        # Verify all features work
        assert len(chunks) == 3
        assert agent.get_analytics().get_metrics("async_stream").total_chunks == 3
        assert len(memory.get_history()) > 0
        assert agent.total_cost > 0

    def test_streaming_reset_analytics_preserves_other_features(self) -> None:
        """Test resetting analytics doesn't affect streaming or other features."""
        chunks: list = []

        memory = ConversationMemory(max_messages=10)
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "stream_numbers", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            enable_analytics=True,
            observers=[_ChunkRecorder(chunks)],
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
        assert len(chunks) == 3  # Observer still worked
        assert len(memory.get_history()) > 0  # Memory still has data


class TestStreamingToolboxIntegration:
    """Test streaming toolbox tools with full feature set."""

    def test_toolbox_streaming_with_analytics(self, tmp_path: Path) -> None:
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

    def test_toolbox_csv_streaming_with_observers(self, tmp_path: Path) -> None:
        """Test CSV streaming with observers."""
        from selectools.toolbox.data_tools import process_csv_stream

        # Create test CSV
        test_csv = tmp_path / "test.csv"
        test_csv.write_text("name,value\nA,1\nB,2\n")

        chunks: list = []

        provider = FakeProvider(
            responses=[
                f'TOOL_CALL: {{"tool_name": "process_csv_stream", "parameters": {{"filepath": "{test_csv}"}}}}',
                "CSV processed!",
            ]
        )
        config = AgentConfig(observers=[_ChunkRecorder(chunks)], max_iterations=2)
        agent = Agent(tools=[process_csv_stream], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Process CSV")])

        # Should have received chunks
        assert len(chunks) > 0
        combined = "".join(chunks)
        assert "name" in combined
        assert "value" in combined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
