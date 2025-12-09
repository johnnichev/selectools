"""
Unit tests for streaming tools functionality.

Tests cover:
- Basic streaming tool execution
- Async streaming tool execution
- Chunk callback invocation
- Analytics tracking for streaming
- Error handling mid-stream
- Integration with hooks
"""

import asyncio
import inspect
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter, tool
from selectools.analytics import AgentAnalytics
from selectools.providers.stubs import LocalProvider
from tests.test_framework import FakeProvider


@tool(description="Simple streaming tool for testing", streaming=True)
def simple_stream(count: int) -> Generator[str, None, None]:
    """Yield count chunks."""
    for i in range(count):
        yield f"Chunk {i}\n"


@tool(description="Async streaming tool for testing", streaming=True)
async def async_stream(count: int) -> AsyncGenerator[str, None]:
    """Async yield count chunks."""
    for i in range(count):
        await asyncio.sleep(0.001)
        yield f"AsyncChunk {i}\n"


@tool(description="Streaming tool that raises an error mid-stream", streaming=True)
def error_stream(error_at: int) -> Generator[str, None, None]:
    """Yield chunks and raise error at specified index."""
    for i in range(10):
        if i == error_at:
            raise ValueError(f"Error at chunk {i}")
        yield f"Chunk {i}\n"


@tool(description="Non-streaming tool for comparison")
def non_streaming(text: str) -> str:
    """Regular non-streaming tool."""
    return f"Result: {text}"


class TestBasicStreamingExecution:
    """Test basic streaming tool execution."""

    def test_sync_streaming_tool_execute(self):
        """Test synchronous streaming tool execution."""
        result = simple_stream.execute({"count": 3})
        assert result == "Chunk 0\nChunk 1\nChunk 2\n"

    def test_streaming_tool_is_streaming_property(self):
        """Test is_streaming property."""
        assert simple_stream.is_streaming is True
        assert simple_stream.streaming is True
        assert non_streaming.is_streaming is False
        assert non_streaming.streaming is False

    def test_streaming_tool_chunk_accumulation(self):
        """Test that chunks are accumulated correctly."""
        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        result = simple_stream.execute({"count": 5}, chunk_callback=callback)

        assert len(chunks_received) == 5
        assert chunks_received == [f"Chunk {i}\n" for i in range(5)]
        assert result == "".join(chunks_received)

    def test_non_streaming_tool_no_chunks(self):
        """Test that non-streaming tools don't trigger chunk callback."""
        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        result = non_streaming.execute({"text": "hello"}, chunk_callback=callback)

        assert len(chunks_received) == 0
        assert result == "Result: hello"

    def test_streaming_tool_empty_stream(self):
        """Test streaming tool with zero chunks."""
        result = simple_stream.execute({"count": 0})
        assert result == ""

    def test_streaming_tool_single_chunk(self):
        """Test streaming tool with single chunk."""
        result = simple_stream.execute({"count": 1})
        assert result == "Chunk 0\n"


class TestAsyncStreamingExecution:
    """Test async streaming tool execution."""

    @pytest.mark.asyncio
    async def test_async_streaming_tool_aexecute(self):
        """Test async streaming tool execution."""
        result = await async_stream.aexecute({"count": 3})
        assert result == "AsyncChunk 0\nAsyncChunk 1\nAsyncChunk 2\n"

    @pytest.mark.asyncio
    async def test_async_streaming_chunk_accumulation(self):
        """Test that async chunks are accumulated correctly."""
        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        result = await async_stream.aexecute({"count": 5}, chunk_callback=callback)

        assert len(chunks_received) == 5
        assert chunks_received == [f"AsyncChunk {i}\n" for i in range(5)]
        assert result == "".join(chunks_received)

    @pytest.mark.asyncio
    async def test_sync_streaming_in_async_context(self):
        """Test sync streaming tool in async context."""
        result = await simple_stream.aexecute({"count": 3})
        assert result == "Chunk 0\nChunk 1\nChunk 2\n"

    @pytest.mark.asyncio
    async def test_sync_streaming_with_callback_in_async(self):
        """Test sync streaming with callback in async context."""
        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        result = await simple_stream.aexecute({"count": 3}, chunk_callback=callback)

        assert len(chunks_received) == 3
        assert result == "Chunk 0\nChunk 1\nChunk 2\n"


class TestStreamingErrorHandling:
    """Test error handling for streaming tools."""

    def test_streaming_error_mid_stream(self):
        """Test that error mid-stream is caught properly."""
        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        with pytest.raises(Exception) as exc_info:
            error_stream.execute({"error_at": 3}, chunk_callback=callback)

        # Should have received chunks before error
        assert len(chunks_received) == 3
        assert "Error at chunk 3" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_streaming_error_handling(self):
        """Test async streaming error handling."""

        @tool(description="Async error streaming", streaming=True)
        async def async_error_stream(error_at: int) -> AsyncGenerator[str, None]:
            for i in range(10):
                if i == error_at:
                    raise ValueError(f"Async error at {i}")
                await asyncio.sleep(0.001)
                yield f"Chunk {i}\n"

        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        with pytest.raises(Exception) as exc_info:
            await async_error_stream.aexecute({"error_at": 2}, chunk_callback=callback)

        assert len(chunks_received) == 2
        assert "Async error at 2" in str(exc_info.value)


class TestStreamingWithAnalytics:
    """Test streaming integration with analytics."""

    def test_analytics_tracks_streaming_chunks(self):
        """Test that analytics tracks chunk count for streaming tools."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "simple_stream", "parameters": {"count": 5}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=2)
        agent = Agent(tools=[simple_stream], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream 5 chunks")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("simple_stream")

        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.streaming_calls == 1
        assert metrics.total_chunks == 5

    def test_analytics_non_streaming_zero_chunks(self):
        """Test that non-streaming tools show zero chunks."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "non_streaming", "parameters": {"text": "hello"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=2)
        agent = Agent(tools=[non_streaming], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Run non-streaming tool")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("non_streaming")

        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.streaming_calls == 0
        assert metrics.total_chunks == 0

    def test_analytics_mixed_streaming_non_streaming(self):
        """Test analytics with both streaming and non-streaming calls."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "simple_stream", "parameters": {"count": 3}}',
                'TOOL_CALL: {"tool_name": "non_streaming", "parameters": {"text": "test"}}',
                'TOOL_CALL: {"tool_name": "simple_stream", "parameters": {"count": 2}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=5)
        agent = Agent(tools=[simple_stream, non_streaming], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Run mixed tools")])

        analytics = agent.get_analytics()
        stream_metrics = analytics.get_metrics("simple_stream")
        non_stream_metrics = analytics.get_metrics("non_streaming")

        assert stream_metrics.total_calls == 2
        assert stream_metrics.streaming_calls == 2
        assert stream_metrics.total_chunks == 5  # 3 + 2

        assert non_stream_metrics.total_calls == 1
        assert non_stream_metrics.streaming_calls == 0
        assert non_stream_metrics.total_chunks == 0


class TestStreamingWithHooks:
    """Test streaming integration with observability hooks."""

    def test_on_tool_chunk_hook_invoked(self):
        """Test that on_tool_chunk hook is called for each chunk."""
        chunks_received = []

        def on_tool_chunk(tool_name: str, chunk: str):
            chunks_received.append((tool_name, chunk))

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "simple_stream", "parameters": {"count": 3}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            hooks={"on_tool_chunk": on_tool_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[simple_stream], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Stream chunks")])

        assert len(chunks_received) == 3
        assert chunks_received == [
            ("simple_stream", "Chunk 0\n"),
            ("simple_stream", "Chunk 1\n"),
            ("simple_stream", "Chunk 2\n"),
        ]

    def test_on_tool_chunk_not_called_for_non_streaming(self):
        """Test that on_tool_chunk is not called for non-streaming tools."""
        chunks_received = []

        def on_tool_chunk(tool_name: str, chunk: str):
            chunks_received.append((tool_name, chunk))

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "non_streaming", "parameters": {"text": "hello"}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            hooks={"on_tool_chunk": on_tool_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[non_streaming], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Run non-streaming")])

        assert len(chunks_received) == 0

    @pytest.mark.asyncio
    async def test_on_tool_chunk_hook_async(self):
        """Test on_tool_chunk hook with async agent."""
        chunks_received = []

        def on_tool_chunk(tool_name: str, chunk: str):
            chunks_received.append((tool_name, chunk))

        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "async_stream", "parameters": {"count": 4}}',
                "Done!",
            ]
        )
        config = AgentConfig(
            hooks={"on_tool_chunk": on_tool_chunk},
            max_iterations=2,
        )
        agent = Agent(tools=[async_stream], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Async stream chunks")])

        assert len(chunks_received) == 4
        assert all(tool_name == "async_stream" for tool_name, _ in chunks_received)


class TestStreamingToolValidation:
    """Test validation for streaming tools."""

    def test_streaming_tool_with_generator_function(self):
        """Test that generator functions are detected."""

        def gen_func() -> Generator[str, None, None]:
            yield "test"

        assert inspect.isgeneratorfunction(gen_func)

    def test_streaming_tool_creation(self):
        """Test creating a streaming tool."""
        tool_obj = Tool(
            name="test_stream",
            description="Test streaming tool",
            parameters=[ToolParameter(name="count", param_type=int, description="Count")],
            function=simple_stream.function,
            streaming=True,
        )

        assert tool_obj.streaming is True
        assert tool_obj.is_streaming is True

    def test_non_streaming_tool_default(self):
        """Test that streaming defaults to False."""
        tool_obj = Tool(
            name="test_regular",
            description="Test regular tool",
            parameters=[],
            function=lambda: "test",
        )

        assert tool_obj.streaming is False
        assert tool_obj.is_streaming is False


class TestStreamingToolboxTools:
    """Test streaming tools from toolbox."""

    def test_read_file_stream_tool_exists(self):
        """Test that read_file_stream tool exists in toolbox."""
        from selectools.toolbox.file_tools import read_file_stream

        assert read_file_stream.is_streaming is True

    def test_process_csv_stream_tool_exists(self):
        """Test that process_csv_stream tool exists in toolbox."""
        from selectools.toolbox.data_tools import process_csv_stream

        assert process_csv_stream.is_streaming is True

    def test_read_file_stream_basic_execution(self, tmp_path):
        """Test basic execution of read_file_stream."""
        from selectools.toolbox.file_tools import read_file_stream

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")

        result = read_file_stream.execute({"filepath": str(test_file)})

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "[Line" in result  # Should have line numbers

    def test_process_csv_stream_basic_execution(self, tmp_path):
        """Test basic execution of process_csv_stream."""
        from selectools.toolbox.data_tools import process_csv_stream

        # Create a test CSV file
        test_csv = tmp_path / "test.csv"
        test_csv.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")

        result = process_csv_stream.execute({"filepath": str(test_csv)})

        assert "Alice" in result
        assert "Bob" in result
        assert "name" in result
        assert "age" in result
        assert "city" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
