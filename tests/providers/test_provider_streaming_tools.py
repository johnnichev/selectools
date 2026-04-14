"""
Tests that streaming methods receive tools and yield ToolCall objects correctly.

These tests exist because of a systemic bug class where:
- Provider stream()/astream() accepted tools in their signature but the agent
  or the provider itself never actually passed them through.
- Provider astream() only yielded text, silently dropping native tool calls.
- Mock providers used **kwargs, hiding the fact that tools were never sent.

Every test here uses *recording* providers that capture call arguments
so assertions can verify the exact contract between agent ↔ provider.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.providers.base import Provider, ProviderError
from selectools.tools import Tool, tool
from selectools.types import AgentResult, Message, Role, StreamChunk, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")


# ---------------------------------------------------------------------------
# Recording providers that capture call kwargs
# ---------------------------------------------------------------------------


class RecordingProvider(Provider):
    """Provider that records every call's kwargs for later assertions."""

    name = "recording"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        complete_responses: Optional[List[Message]] = None,
        stream_chunks: Optional[List[str]] = None,
        astream_chunks: Optional[List[Union[str, ToolCall]]] = None,
    ):
        self.default_model = "recording-model"
        self._complete_responses = complete_responses or [
            Message(role=Role.ASSISTANT, content="done")
        ]
        self._stream_chunks = stream_chunks or ["hello"]
        self._astream_chunks = astream_chunks or ["hello"]
        self._call_idx = 0

        self.complete_calls: List[Dict[str, Any]] = []
        self.stream_calls: List[Dict[str, Any]] = []
        self.astream_calls: List[Dict[str, Any]] = []
        self.acomplete_calls: List[Dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self.complete_calls.append({"model": model, "tools": tools, "messages": messages})
        resp = self._complete_responses[min(self._call_idx, len(self._complete_responses) - 1)]
        self._call_idx += 1
        return resp, _DUMMY_USAGE

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Iterable[str]:
        self.stream_calls.append({"model": model, "tools": tools, "messages": messages})
        for chunk in self._stream_chunks:
            yield chunk

    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> Tuple[Message, UsageStats]:
        self.acomplete_calls.append({"model": model, "tools": tools, "messages": messages})
        resp = self._complete_responses[min(self._call_idx, len(self._complete_responses) - 1)]
        self._call_idx += 1
        return resp, _DUMMY_USAGE

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Optional[float] = None,
    ) -> AsyncIterable[Union[str, ToolCall]]:
        self.astream_calls.append({"model": model, "tools": tools, "messages": messages})
        for chunk in self._astream_chunks:
            yield chunk


@tool()
def dummy_tool(x: int) -> str:
    """A dummy tool for testing."""
    return str(x * 2)


# ============================================================================
# 1. Agent passes tools to provider streaming methods
# ============================================================================


class TestAgentPassesToolsToStream:
    """Verify the agent sends tools to provider.stream() and provider.astream()."""

    def test_run_stream_passes_tools(self) -> None:
        """agent.run(stream=True) must pass tools to provider.stream()."""
        provider = RecordingProvider(stream_chunks=["Hello"])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1, stream=True),
        )
        agent.run([Message(role=Role.USER, content="hi")])

        assert len(provider.stream_calls) == 1
        passed_tools = provider.stream_calls[0]["tools"]
        assert passed_tools is not None, "tools was not passed to stream()"
        assert len(passed_tools) == 1
        assert passed_tools[0].name == "dummy_tool"

    @pytest.mark.asyncio
    async def test_arun_stream_passes_tools(self) -> None:
        """agent.arun(stream=True) must pass tools to provider.astream()."""
        provider = RecordingProvider(astream_chunks=["Hello"])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1, stream=True),
        )
        await agent.arun([Message(role=Role.USER, content="hi")])

        assert len(provider.astream_calls) == 1
        passed_tools = provider.astream_calls[0]["tools"]
        assert passed_tools is not None, "tools was not passed to astream()"
        assert len(passed_tools) == 1
        assert passed_tools[0].name == "dummy_tool"

    @pytest.mark.asyncio
    async def test_agent_astream_passes_tools(self) -> None:
        """agent.astream() must pass tools to provider.astream()."""
        provider = RecordingProvider(astream_chunks=["Hello"])
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1),
        )

        async for _ in agent.astream([Message(role=Role.USER, content="hi")]):
            pass

        assert len(provider.astream_calls) == 1
        passed_tools = provider.astream_calls[0]["tools"]
        assert passed_tools is not None, "tools was not passed to astream()"
        assert len(passed_tools) == 1

    def test_run_non_stream_passes_tools(self) -> None:
        """agent.run(stream=False) must pass tools to provider.complete()."""
        provider = RecordingProvider()
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1, stream=False),
        )
        agent.run([Message(role=Role.USER, content="hi")])

        assert len(provider.complete_calls) == 1
        passed_tools = provider.complete_calls[0]["tools"]
        assert passed_tools is not None
        assert len(passed_tools) == 1

    @pytest.mark.asyncio
    async def test_arun_non_stream_passes_tools(self) -> None:
        """agent.arun(stream=False) must pass tools to provider.acomplete()."""
        provider = RecordingProvider()
        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1, stream=False),
        )
        await agent.arun([Message(role=Role.USER, content="hi")])

        assert len(provider.acomplete_calls) == 1
        passed_tools = provider.acomplete_calls[0]["tools"]
        assert passed_tools is not None
        assert len(passed_tools) == 1


# ============================================================================
# 2. Agent astream correctly handles ToolCall objects (not str()-ified)
# ============================================================================


class TestAstreamToolCallHandling:
    """Verify ToolCall objects from astream are properly handled, not corrupted."""

    @pytest.mark.asyncio
    async def test_astream_yields_toolcall_as_streamchunk(self) -> None:
        """ToolCall from provider.astream() must appear in StreamChunk.tool_calls."""
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": 5}, id="call_abc")
        provider = RecordingProvider(astream_chunks=["Thinking...", tc, "Done"])

        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=2),
        )

        items: List[Union[StreamChunk, AgentResult]] = []
        async for item in agent.astream([Message(role=Role.USER, content="hi")]):
            items.append(item)

        tool_chunks = [i for i in items if isinstance(i, StreamChunk) and i.tool_calls]
        assert len(tool_chunks) >= 1, "ToolCall not yielded as StreamChunk"
        assert tool_chunks[0].tool_calls[0].tool_name == "dummy_tool"
        assert tool_chunks[0].tool_calls[0].parameters == {"x": 5}

    @pytest.mark.asyncio
    async def test_astreaming_call_does_not_stringify_toolcalls(self) -> None:
        """_astreaming_call must not convert ToolCall objects to strings."""
        tc = ToolCall(tool_name="dummy_tool", parameters={"x": 5}, id="call_abc")
        provider = RecordingProvider(astream_chunks=["Hello", tc])

        agent = Agent(
            tools=[dummy_tool],
            provider=provider,
            config=AgentConfig(max_iterations=1, stream=True),
        )
        result = await agent.arun([Message(role=Role.USER, content="hi")])

        assert "ToolCall(" not in result.content, (
            "ToolCall object was str()-ified into the response text"
        )


# ============================================================================
# 3. Ollama _format_messages: TOOL role and ASSISTANT tool_calls
# ============================================================================


class TestOllamaMessageFormatting:
    """Unit tests for OllamaProvider._format_messages tool handling."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        try:
            return OllamaProvider.__new__(OllamaProvider)
        except Exception:
            pytest.skip("Cannot instantiate OllamaProvider without openai")

    def test_tool_role_formatted_correctly(self) -> None:
        """TOOL messages must use role='tool' with tool_call_id (OpenAI compat)."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="Result: 42",
                tool_call_id="call_xyz",
                tool_name="calculator",
            )
        ]
        formatted = provider._format_messages("system prompt", messages)

        tool_msg = formatted[1]  # [0] is system
        assert tool_msg["role"] == "tool", f"TOOL role should be 'tool', got '{tool_msg['role']}'"
        assert tool_msg["tool_call_id"] == "call_xyz"
        assert tool_msg["content"] == "Result: 42"

    def test_tool_role_not_mapped_to_assistant(self) -> None:
        """TOOL messages must NOT be mapped to 'assistant' role."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="ok",
                tool_call_id="call_1",
                tool_name="test",
            )
        ]
        formatted = provider._format_messages("system", messages)
        assert formatted[1]["role"] != "assistant"

    def test_assistant_includes_tool_calls(self) -> None:
        """ASSISTANT messages with tool_calls must include them in the payload."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="calculator",
                        parameters={"expression": "2+2"},
                        id="call_abc",
                    )
                ],
            )
        ]
        formatted = provider._format_messages("system", messages)
        assistant_msg = formatted[1]

        assert "tool_calls" in assistant_msg, "ASSISTANT message must include tool_calls"
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "calculator"
        args = json.loads(tc["function"]["arguments"])
        assert args["expression"] == "2+2"

    def test_assistant_without_tool_calls_has_no_key(self) -> None:
        """ASSISTANT messages without tool_calls should not have the key."""
        provider = self._get_provider()
        messages = [Message(role=Role.ASSISTANT, content="Hello!")]
        formatted = provider._format_messages("system", messages)
        assert "tool_calls" not in formatted[1]


# ============================================================================
# 4. Anthropic astream: yields ToolCall from streaming events
# ============================================================================


class TestAnthropicAstreamToolCalls:
    """Verify Anthropic astream() properly yields ToolCall objects."""

    @pytest.mark.asyncio
    async def test_astream_yields_toolcall_from_tool_use_blocks(self) -> None:
        """Anthropic astream must yield ToolCall when content_block events fire."""
        from unittest.mock import AsyncMock, MagicMock

        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        events = [
            _mock_event(
                "content_block_start",
                content_block=_mock_block("tool_use", id="toolu_123", name="calculator"),
            ),
            _mock_event(
                "content_block_delta",
                delta=_mock_delta("input_json_delta", partial_json='{"expression":'),
            ),
            _mock_event(
                "content_block_delta", delta=_mock_delta("input_json_delta", partial_json=' "2+2"}')
            ),
            _mock_event("content_block_stop"),
        ]

        mock_stream = AsyncIteratorFromList(events)
        provider._async_client = MagicMock()
        provider._async_client.messages.create = AsyncMock(return_value=mock_stream)
        provider.default_model = "claude-test"

        results: List[Union[str, ToolCall]] = []
        async for item in provider.astream(
            model="claude-test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="test")],
            tools=None,
            max_tokens=100,
        ):
            results.append(item)

        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1, f"Expected 1 ToolCall, got {len(tool_calls)}"
        assert tool_calls[0].tool_name == "calculator"
        assert tool_calls[0].parameters == {"expression": "2+2"}
        assert tool_calls[0].id == "toolu_123"

    @pytest.mark.asyncio
    async def test_astream_yields_text_and_toolcalls_together(self) -> None:
        """Mixed text + tool_use blocks both yield correctly."""
        from unittest.mock import AsyncMock, MagicMock

        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)

        events = [
            _mock_event("content_block_start", content_block=_mock_block("text")),
            _mock_event(
                "content_block_delta", delta=_mock_delta("text_delta", text="Let me calculate")
            ),
            _mock_event("content_block_stop"),
            _mock_event(
                "content_block_start",
                content_block=_mock_block("tool_use", id="toolu_456", name="add"),
            ),
            _mock_event(
                "content_block_delta",
                delta=_mock_delta("input_json_delta", partial_json='{"a": 1, "b": 2}'),
            ),
            _mock_event("content_block_stop"),
        ]

        mock_stream = AsyncIteratorFromList(events)
        provider._async_client = MagicMock()
        provider._async_client.messages.create = AsyncMock(return_value=mock_stream)
        provider.default_model = "claude-test"

        results: List[Union[str, ToolCall]] = []
        async for item in provider.astream(
            model="claude-test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="test")],
            max_tokens=100,
        ):
            results.append(item)

        texts = [r for r in results if isinstance(r, str)]
        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(texts) >= 1
        assert "Let me calculate" in texts[0]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "add"


# ============================================================================
# 5. Anthropic stream/astream pass tools to API
# ============================================================================


class TestAnthropicStreamPassesTools:
    """Verify Anthropic streaming methods pass tools to the API request."""

    def test_stream_includes_tools_in_request(self) -> None:
        """Anthropic stream() must include tools in request_args."""
        from unittest.mock import MagicMock

        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._client = MagicMock()
        provider._client.messages.create.return_value = iter([])
        provider.default_model = "claude-test"

        mock_tool = MagicMock()
        mock_tool.schema.return_value = {
            "name": "test",
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        }

        list(
            provider.stream(
                model="claude-test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
                tools=[mock_tool],
                max_tokens=100,
            )
        )

        call_kwargs = provider._client.messages.create.call_args
        assert "tools" in call_kwargs[1] or (call_kwargs[0] and "tools" in call_kwargs[0]), (
            "tools not passed to Anthropic stream() API call"
        )

    @pytest.mark.asyncio
    async def test_astream_includes_tools_in_request(self) -> None:
        """Anthropic astream() must include tools in request_args."""
        from unittest.mock import AsyncMock, MagicMock

        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._async_client = MagicMock()
        provider._async_client.messages.create = AsyncMock(return_value=AsyncIteratorFromList([]))
        provider.default_model = "claude-test"

        mock_tool = MagicMock()
        mock_tool.schema.return_value = {
            "name": "test",
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        }

        async for _ in provider.astream(
            model="claude-test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="hi")],
            tools=[mock_tool],
            max_tokens=100,
        ):
            pass

        call_kwargs = provider._async_client.messages.create.call_args
        assert "tools" in call_kwargs[1], "tools not passed to Anthropic astream() API call"


# ============================================================================
# 6. FallbackProvider.astream error handling and failover
# ============================================================================


class TestFallbackProviderAstream:
    """Verify FallbackProvider.astream has error handling and failover."""

    @pytest.mark.asyncio
    async def test_astream_failover_on_retriable_error(self) -> None:
        """If the first provider's astream() raises a retriable error,
        the fallback should try the next provider."""
        from selectools.providers.fallback import FallbackProvider

        class FailingStreamProvider:
            name = "failing"
            supports_streaming = True
            supports_async = True

            def astream(self, **kwargs: Any) -> Any:
                raise ProviderError("503 service unavailable")

        class WorkingStreamProvider:
            name = "working"
            supports_streaming = True
            supports_async = True
            used = False

            async def astream(self, **kwargs: Any) -> AsyncIterable[str]:
                self.used = True
                yield "hello from fallback"  # type: ignore[misc]

        working = WorkingStreamProvider()
        fb = FallbackProvider(providers=[FailingStreamProvider(), working])
        chunks = []
        async for chunk in fb.astream(
            model="test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            chunks.append(chunk)
        assert fb.provider_used == "working"
        assert chunks == ["hello from fallback"]

    @pytest.mark.asyncio
    async def test_astream_records_failure_for_circuit_breaker(self) -> None:
        """Failed astream() calls must be recorded for the circuit breaker."""
        from selectools.providers.fallback import FallbackProvider

        class FailProvider:
            name = "fail"
            supports_streaming = True
            supports_async = True

            def astream(self, **kwargs: Any) -> Any:
                raise ProviderError("timeout")

        class OkProvider:
            name = "ok"
            supports_streaming = True
            supports_async = True

            async def astream(self, **kwargs: Any) -> AsyncIterable[str]:
                yield "ok"  # type: ignore[misc]

        fb = FallbackProvider(
            providers=[FailProvider(), OkProvider()],
            circuit_breaker_threshold=2,
        )

        async for _ in fb.astream(
            model="test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            pass
        assert fb._failures.get("fail", 0) == 1

    @pytest.mark.asyncio
    async def test_astream_raises_if_all_exhausted(self) -> None:
        """If all providers fail in astream(), ProviderError must be raised."""
        from selectools.providers.fallback import FallbackProvider

        class FailProvider:
            name = "fail"
            supports_streaming = True
            supports_async = True

            def astream(self, **kwargs: Any) -> Any:
                raise ProviderError("rate limit 429")

        fb = FallbackProvider(providers=[FailProvider()])
        with pytest.raises(ProviderError, match="Last error"):
            async for _ in fb.astream(
                model="test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass


# ============================================================================
# 7. FallbackProvider.stream passes tools through
# ============================================================================


class TestFallbackStreamPassesTools:
    """Verify FallbackProvider delegates tools to child providers."""

    def test_stream_passes_tools_to_child(self) -> None:
        """FallbackProvider.stream() must forward tools to child.stream()."""
        from selectools.providers.fallback import FallbackProvider

        received_tools: List[Any] = []

        class CapturingProvider:
            name = "capturing"
            supports_streaming = True

            def stream(
                self,
                *,
                tools: Any = None,
                **kwargs: Any,
            ) -> Generator[str, None, None]:
                received_tools.append(tools)
                yield "ok"

        mock_tool_obj = MagicMock()
        fb = FallbackProvider(providers=[CapturingProvider()])
        list(
            fb.stream(
                model="test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
                tools=[mock_tool_obj],
            )
        )

        assert len(received_tools) == 1
        assert received_tools[0] is not None
        assert len(received_tools[0]) == 1

    @pytest.mark.asyncio
    async def test_astream_passes_tools_to_child(self) -> None:
        """FallbackProvider.astream() must forward tools to child.astream()."""
        from selectools.providers.fallback import FallbackProvider

        received_tools: List[Any] = []

        class CapturingAsyncProvider:
            name = "capturing-async"
            supports_streaming = True
            supports_async = True

            async def astream(
                self,
                *,
                tools: Any = None,
                **kwargs: Any,
            ) -> AsyncIterable[str]:
                received_tools.append(tools)
                yield "ok"  # type: ignore[misc]

        mock_tool_obj = MagicMock()
        fb = FallbackProvider(providers=[CapturingAsyncProvider()])
        async for _ in fb.astream(
            model="test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="hi")],
            tools=[mock_tool_obj],
        ):
            pass

        assert len(received_tools) == 1
        assert received_tools[0] is not None


# ============================================================================
# 8. OpenAI stream passes tools
# ============================================================================


class TestOpenAIStreamPassesTools:
    """Verify OpenAI stream() passes tools to the API call."""

    def test_stream_includes_tools_in_api_call(self) -> None:
        """OpenAI stream() must pass tools to create() kwargs."""
        from unittest.mock import MagicMock

        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = iter([])
        provider.default_model = "gpt-4o"
        provider.api_key = "test"

        mock_tool = MagicMock()
        mock_tool.schema.return_value = {
            "name": "test",
            "description": "desc",
            "parameters": {"type": "object", "properties": {}},
        }

        list(
            provider.stream(
                model="gpt-4o",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
                tools=[mock_tool],
                max_tokens=100,
            )
        )

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs, "tools not passed to OpenAI stream() API call"


# ============================================================================
# Helpers
# ============================================================================


class AsyncIteratorFromList:
    """Wrap a list as an async iterator for mocking streaming responses."""

    def __init__(self, items: list) -> None:
        self._items = items
        self._idx = 0

    def __aiter__(self) -> "AsyncIteratorFromList":
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _mock_event(event_type: str, **attrs: Any) -> Any:
    """Create a mock Anthropic streaming event."""
    from unittest.mock import MagicMock

    event = MagicMock()
    event.type = event_type
    for k, v in attrs.items():
        setattr(event, k, v)
    return event


def _mock_delta(delta_type: str, **attrs: Any) -> Any:
    """Create a mock Anthropic delta object."""
    from unittest.mock import MagicMock

    delta = MagicMock()
    delta.type = delta_type
    delta.text = attrs.get("text")
    delta.partial_json = attrs.get("partial_json")
    return delta


def _mock_block(block_type: str, **attrs: Any) -> Any:
    """Create a mock Anthropic content block."""
    from unittest.mock import MagicMock

    block = MagicMock()
    block.type = block_type
    block.id = attrs.get("id")
    block.name = attrs.get("name")
    return block


# Make MagicMock available at module level for FallbackProvider tests
from unittest.mock import MagicMock

# ============================================================================
# REGRESSION: Gemini stream/astream crash on safety-filtered chunks (ValueError)
# ============================================================================


class TestGeminiStreamSafetyFilter:
    """Verify Gemini stream/astream skip safety-filtered chunks without crashing."""

    def _make_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        return provider

    def test_stream_skips_safety_filtered_chunk(self) -> None:
        """stream() must not crash when chunk.text raises ValueError (safety filter)."""
        provider = self._make_provider()

        safe_chunk = MagicMock()
        safe_chunk.text = "hello"
        safe_chunk.candidates = None

        filtered_chunk = MagicMock()
        # Accessing .text on a safety-filtered chunk raises ValueError in the Gemini SDK
        type(filtered_chunk).text = property(
            lambda self: (_ for _ in ()).throw(ValueError("no candidates"))
        )
        filtered_chunk.candidates = None

        mock_stream = iter([safe_chunk, filtered_chunk])

        provider._client = MagicMock()
        provider._client.models.generate_content_stream.return_value = mock_stream
        provider.default_model = "gemini-test"

        from google.genai import types  # type: ignore[import]

        provider._genai = MagicMock()

        # Build minimal config mock
        config_mock = MagicMock()
        with (
            MagicMock() as _mock_types,
        ):
            pass

        # Use a direct mock of _format_contents and _map_tool_to_gemini to avoid SDK calls
        provider._format_contents = MagicMock(return_value=[])

        # Patch the types import inside stream
        import unittest.mock as um

        with um.patch("selectools.providers.gemini_provider.GeminiProvider.stream") as _patched:
            # We need to actually call the REAL stream method with mocked internals
            pass

        # Call the real stream method with mocked internals
        from selectools.types import Message, Role

        provider._format_contents = MagicMock(return_value=[])

        with um.patch("google.genai.types.GenerateContentConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            provider._client.models.generate_content_stream.return_value = iter(
                [safe_chunk, filtered_chunk]
            )
            results = list(
                provider.stream(
                    model="gemini-test",
                    system_prompt="test",
                    messages=[Message(role=Role.USER, content="hi")],
                    max_tokens=100,
                )
            )

        # The safe chunk's text should appear; the filtered chunk should be silently skipped
        text_results = [r for r in results if isinstance(r, str)]
        assert "hello" in text_results, "safe chunk text should be yielded"
        assert len(results) == 1, f"filtered chunk should be skipped, got {results}"

    @pytest.mark.asyncio
    async def test_astream_skips_safety_filtered_chunk(self) -> None:
        """astream() must not crash when chunk.text raises ValueError (safety filter)."""
        from unittest.mock import AsyncMock

        provider = self._make_provider()

        safe_chunk = MagicMock()
        safe_chunk.text = "world"
        safe_chunk.candidates = None

        filtered_chunk = MagicMock()
        type(filtered_chunk).text = property(
            lambda self: (_ for _ in ()).throw(ValueError("no candidates"))
        )
        filtered_chunk.candidates = None

        provider._format_contents = MagicMock(return_value=[])

        import unittest.mock as um

        # The Gemini astream() does: stream = await client.aio.models.generate_content_stream(...)
        # So generate_content_stream must be an async mock that returns an async iterable.
        class AsyncIterableChunks:
            def __aiter__(self) -> "AsyncIterableChunks":
                self._idx = 0
                return self

            async def __anext__(self) -> Any:
                chunks = [safe_chunk, filtered_chunk]
                if self._idx >= len(chunks):
                    raise StopAsyncIteration
                chunk = chunks[self._idx]
                self._idx += 1
                return chunk

        mock_stream = AsyncIterableChunks()

        with um.patch("google.genai.types.GenerateContentConfig") as MockConfig:
            MockConfig.return_value = MagicMock()
            provider._client = MagicMock()
            provider._client.aio = MagicMock()
            provider._client.aio.models = MagicMock()
            provider._client.aio.models.generate_content_stream = AsyncMock(
                return_value=mock_stream
            )

            from selectools.types import Message, Role

            results: List[Any] = []
            async for item in provider.astream(
                model="gemini-test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
                max_tokens=100,
            ):
                results.append(item)

        text_results = [r for r in results if isinstance(r, str)]
        assert "world" in text_results, "safe chunk text should be yielded"
        assert len(results) == 1, f"filtered chunk should be skipped, got {results}"


# ============================================================================
# REGRESSION: FallbackProvider.astream() skips sync-only streaming providers
# ============================================================================


class TestFallbackAstreamSkipsSyncOnly:
    """FallbackProvider.astream() must skip providers with supports_async=False."""

    @pytest.mark.asyncio
    async def test_astream_skips_sync_only_streaming_provider(self) -> None:
        """A provider with supports_streaming=True but supports_async=False must be skipped."""
        from selectools.providers.fallback import FallbackProvider

        class SyncOnlyStreamProvider:
            name = "sync-only"
            supports_streaming = True
            supports_async = False  # Async not supported

            async def astream(self, **kwargs: Any) -> AsyncIterable[str]:
                raise NotImplementedError("sync-only provider")
                yield  # pragma: no cover

        class AsyncStreamProvider:
            name = "async-capable"
            supports_streaming = True
            supports_async = True

            async def astream(self, **kwargs: Any) -> AsyncIterable[str]:
                yield "from async provider"  # type: ignore[misc]

        async_provider = AsyncStreamProvider()
        fb = FallbackProvider(providers=[SyncOnlyStreamProvider(), async_provider])

        chunks: List[str] = []
        async for chunk in fb.astream(
            model="test",
            system_prompt="test",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            chunks.append(chunk)  # type: ignore[arg-type]

        assert chunks == ["from async provider"], f"Expected async provider, got {chunks}"
        assert fb.provider_used == "async-capable"

    @pytest.mark.asyncio
    async def test_astream_raises_if_only_sync_providers(self) -> None:
        """If all providers are sync-only, astream() must raise ProviderError."""
        from selectools.providers.fallback import FallbackProvider

        class SyncOnlyProvider:
            name = "sync-only"
            supports_streaming = True
            supports_async = False

        fb = FallbackProvider(providers=[SyncOnlyProvider()])
        with pytest.raises(ProviderError):
            async for _ in fb.astream(
                model="test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass


# ============================================================================
# REGRESSION: FallbackProvider.stream() return type includes ToolCall
# ============================================================================


class TestFallbackStreamYieldsToolCalls:
    """FallbackProvider.stream() must yield ToolCall objects, not just strings."""

    def test_stream_passes_toolcall_from_child(self) -> None:
        """FallbackProvider.stream() must forward ToolCall objects from child providers."""
        from selectools.providers.fallback import FallbackProvider

        tc = ToolCall(tool_name="my_tool", parameters={"x": 1}, id="call_1")

        class ToolCallStreamProvider:
            name = "tc-provider"
            supports_streaming = True

            def stream(self, **kwargs: Any) -> Iterable[Union[str, ToolCall]]:
                yield "text chunk"
                yield tc

        fb = FallbackProvider(providers=[ToolCallStreamProvider()])
        results = list(
            fb.stream(
                model="test",
                system_prompt="test",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )

        assert len(results) == 2
        assert results[0] == "text chunk"
        assert isinstance(results[1], ToolCall)
        assert results[1].tool_name == "my_tool"


# ============================================================================
# REGRESSION: _OpenAICompatibleBase stream/astream flush tool_call_deltas
# when stream ends without a finish_reason chunk (e.g. Ollama).
# ============================================================================


class TestOpenAICompatFlushToolCallsAfterStream:
    """Verify stream/astream flush accumulated tool_call_deltas after the loop
    when no finish_reason chunk was received (e.g. some Ollama models)."""

    def _build_delta_chunk(
        self,
        tc_index: int,
        tc_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        finish_reason: str | None = None,
        content: str | None = None,
    ) -> Any:
        from unittest.mock import MagicMock

        chunk = MagicMock()
        delta = MagicMock()
        delta.content = content
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        chunk.choices = [choice]

        if name is not None or arguments is not None or tc_id is not None:
            tc_delta = MagicMock()
            tc_delta.index = tc_index
            tc_delta.id = tc_id
            func = MagicMock()
            func.name = name
            func.arguments = arguments
            tc_delta.function = func
            delta.tool_calls = [tc_delta]
        else:
            delta.tool_calls = None

        return chunk

    def test_stream_flushes_tool_calls_when_no_finish_reason(self) -> None:
        """stream() must yield ToolCall even when finish_reason is never 'tool_calls'."""
        from unittest.mock import MagicMock

        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"

        # Two chunks: first accumulates the tool call, second has no finish_reason.
        chunks = [
            self._build_delta_chunk(0, tc_id="call_1", name="my_tool", arguments='{"x": 42}'),
            self._build_delta_chunk(0, finish_reason=None),  # stream ends, no finish_reason
        ]
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = iter(chunks)

        results = list(
            provider.stream(
                model="gpt-4o",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="go")],
                max_tokens=100,
            )
        )

        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1, f"Expected 1 ToolCall via flush, got {tool_calls}"
        assert tool_calls[0].tool_name == "my_tool"
        assert tool_calls[0].parameters == {"x": 42}

    @pytest.mark.asyncio
    async def test_astream_flushes_tool_calls_when_no_finish_reason(self) -> None:
        """astream() must yield ToolCall even when finish_reason is never set."""
        from unittest.mock import AsyncMock, MagicMock

        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"

        chunks = [
            self._build_delta_chunk(0, tc_id="call_2", name="tool_b", arguments='{"y": 7}'),
            self._build_delta_chunk(0, finish_reason=None),
        ]
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = AsyncMock(
            return_value=AsyncIteratorFromList(chunks)
        )

        results = []
        async for item in provider.astream(
            model="gpt-4o",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="go")],
            max_tokens=100,
        ):
            results.append(item)

        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1, f"Expected 1 ToolCall via async flush, got {tool_calls}"
        assert tool_calls[0].tool_name == "tool_b"
        assert tool_calls[0].parameters == {"y": 7}

    def test_stream_does_not_double_emit_when_finish_reason_present(self) -> None:
        """When finish_reason='tool_calls' IS present, no duplicate emission from flush."""
        from unittest.mock import MagicMock

        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"

        chunks = [
            self._build_delta_chunk(0, tc_id="call_3", name="my_tool", arguments='{"z": 1}'),
            self._build_delta_chunk(0, finish_reason="tool_calls"),
        ]
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = iter(chunks)

        results = list(
            provider.stream(
                model="gpt-4o",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="go")],
                max_tokens=100,
            )
        )

        tool_calls = [r for r in results if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1, f"Expected exactly 1 ToolCall, got {len(tool_calls)}"
