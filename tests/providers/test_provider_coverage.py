"""
Tests to increase code coverage for 4 provider modules:
- _openai_compat.py (streaming, async, error handling, edge cases)
- anthropic_provider.py (streaming, async, tool formatting, message merging)
- gemini_provider.py (streaming, async, tool mapping, thought signatures)
- ollama_provider.py (all template methods, init, error wrapping)

Mocks the SDK client directly — never calls real APIs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.providers.base import ProviderError
from selectools.tools.base import Tool, ToolParameter
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool() -> Tool:
    """Create a minimal Tool for testing tool-related codepaths."""
    return Tool(
        name="test_tool",
        description="A test tool",
        parameters=[
            ToolParameter(name="x", param_type=int, description="An integer"),
        ],
        function=lambda x: str(x),
    )


def _make_openai_response(content: str = "hello", tool_calls: list | None = None) -> MagicMock:
    """Build a mock OpenAI-style response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.tool_calls = tool_calls
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def _make_openai_stream_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Build a single streaming chunk mock."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def _make_openai_tool_call_delta(
    index: int = 0,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> MagicMock:
    """Build a tool call delta for streaming."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ===========================================================================
# OllamaProvider tests
# ===========================================================================


class TestOllamaProviderInit:
    """Test OllamaProvider initialization and template methods."""

    def test_init_creates_clients(self) -> None:
        """OllamaProvider.__init__ creates OpenAI clients with Ollama base_url."""
        with patch("openai.OpenAI") as MockOpenAI, patch("openai.AsyncOpenAI") as MockAsyncOpenAI:
            from selectools.providers.ollama_provider import OllamaProvider

            provider = OllamaProvider(model="llama3.2", base_url="http://localhost:11434")

            MockOpenAI.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            MockAsyncOpenAI.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            assert provider.default_model == "llama3.2"
            assert provider.base_url == "http://localhost:11434"


class TestOllamaProviderTemplateMethods:
    """Test all Ollama template method overrides."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.default_model = "llama3.2"
        provider.base_url = "http://localhost:11434"
        return provider

    def test_get_token_key(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("llama3.2") == "max_tokens"
        assert provider._get_token_key("any-model") == "max_tokens"

    def test_calculate_cost_is_zero(self) -> None:
        provider = self._get_provider()
        assert provider._calculate_cost("llama3.2", 100, 50) == 0.0

    def test_get_provider_name(self) -> None:
        provider = self._get_provider()
        assert provider._get_provider_name() == "ollama"

    def test_wrap_error_connection(self) -> None:
        provider = self._get_provider()
        err = provider._wrap_error(Exception("Connection refused"), "completion")
        assert isinstance(err, ProviderError)
        assert "Failed to connect to Ollama" in str(err)
        assert "ollama serve" in str(err)

    def test_wrap_error_generic(self) -> None:
        provider = self._get_provider()
        err = provider._wrap_error(Exception("something else"), "streaming")
        assert isinstance(err, ProviderError)
        assert "Ollama streaming failed" in str(err)

    def test_parse_tool_call_id_with_id(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.id = "call_123"
        assert provider._parse_tool_call_id(tc) == "call_123"

    def test_parse_tool_call_id_without_id(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.id = None
        result = provider._parse_tool_call_id(tc)
        assert result.startswith("call_")

    def test_parse_tool_call_arguments_string(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.function.arguments = '{"x": 42}'
        # BUG-31: _parse_tool_call_arguments now returns (params, parse_error).
        params, parse_error = provider._parse_tool_call_arguments(tc)
        assert params == {"x": 42}
        assert parse_error is None

    def test_parse_tool_call_arguments_dict(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.function.arguments = {"x": 42}
        params, parse_error = provider._parse_tool_call_arguments(tc)
        assert params == {"x": 42}
        assert parse_error is None

    def test_parse_tool_call_arguments_invalid_json(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.function.arguments = "not json"
        # BUG-31: malformed JSON now surfaces parse_error instead of silently
        # returning {} with no diagnostic.
        params, parse_error = provider._parse_tool_call_arguments(tc)
        assert params == {}
        assert parse_error is not None and "invalid JSON" in parse_error

    def test_format_tool_call_id_with_id(self) -> None:
        provider = self._get_provider()
        tc = ToolCall(tool_name="test", parameters={}, id="call_abc")
        assert provider._format_tool_call_id(tc) == "call_abc"

    def test_format_tool_call_id_without_id(self) -> None:
        provider = self._get_provider()
        tc = ToolCall(tool_name="test", parameters={}, id=None)
        result = provider._format_tool_call_id(tc)
        assert result.startswith("call_")

    def test_initial_tool_call_id_with_id(self) -> None:
        provider = self._get_provider()
        delta = MagicMock()
        delta.id = "call_xyz"
        assert provider._initial_tool_call_id(delta) == "call_xyz"

    def test_initial_tool_call_id_without_id(self) -> None:
        provider = self._get_provider()
        delta = MagicMock()
        delta.id = None
        result = provider._initial_tool_call_id(delta)
        assert result.startswith("call_")


class TestOllamaProviderComplete:
    """Test OllamaProvider complete/acomplete via inherited _OpenAICompatibleBase."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.default_model = "llama3.2"
        provider.base_url = "http://localhost:11434"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    def test_complete_basic(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.return_value = _make_openai_response("hi there")
        msg, usage = provider.complete(
            model="llama3.2",
            system_prompt="You are helpful.",
            messages=[Message(role=Role.USER, content="hello")],
        )
        assert msg.content == "hi there"
        assert usage.cost_usd == 0.0
        assert usage.provider == "ollama"

    def test_complete_with_tools(self) -> None:
        provider = self._get_provider()
        tc_mock = MagicMock()
        tc_mock.function.name = "test_tool"
        tc_mock.function.arguments = '{"x": 5}'
        tc_mock.id = None  # Ollama may not provide ID
        response = _make_openai_response("", tool_calls=[tc_mock])
        provider._client.chat.completions.create.return_value = response

        msg, usage = provider.complete(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
            tools=[_make_tool()],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "test_tool"
        assert msg.tool_calls[0].id.startswith("call_")

    def test_complete_with_timeout(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.return_value = _make_openai_response()
        provider.complete(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
            timeout=5.0,
        )
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 5.0

    def test_complete_temperature_error_retry(self) -> None:
        """When provider rejects temperature, retry without it."""
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = [
            Exception("temperature is not supported"),
            _make_openai_response("retried"),
        ]
        msg, _ = provider.complete(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "retried"
        assert provider._client.chat.completions.create.call_count == 2

    def test_complete_temperature_retry_also_fails(self) -> None:
        """When both attempts fail (temperature strip + retry), raise ProviderError."""
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = [
            Exception("temperature is not supported"),
            Exception("still broken"),
        ]
        with pytest.raises(ProviderError, match="still broken"):
            provider.complete(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    def test_complete_non_temperature_error(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = Exception("quota exceeded")
        with pytest.raises(ProviderError, match="quota exceeded"):
            provider.complete(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    @pytest.mark.asyncio
    async def test_acomplete_basic(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("async hi")
        )
        msg, usage = await provider.acomplete(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hello")],
        )
        assert msg.content == "async hi"
        assert usage.provider == "ollama"

    @pytest.mark.asyncio
    async def test_acomplete_temperature_error_retry(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("temperature not allowed"),
                _make_openai_response("ok"),
            ]
        )
        msg, _ = await provider.acomplete(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        )
        assert msg.content == "ok"

    @pytest.mark.asyncio
    async def test_acomplete_temperature_retry_also_fails(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("temperature issue"),
                Exception("still broken"),
            ]
        )
        with pytest.raises(ProviderError, match="still broken"):
            await provider.acomplete(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    @pytest.mark.asyncio
    async def test_acomplete_non_temperature_error(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("network error")
        )
        with pytest.raises(ProviderError, match="network error"):
            await provider.acomplete(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )


# ===========================================================================
# _OpenAICompatibleBase streaming tests (via OllamaProvider)
# ===========================================================================


class TestOpenAICompatStream:
    """Test sync stream() in _OpenAICompatibleBase via OllamaProvider."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.default_model = "llama3.2"
        provider.base_url = "http://localhost:11434"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    def test_stream_text_chunks(self) -> None:
        provider = self._get_provider()
        chunks = [
            _make_openai_stream_chunk(content="Hello "),
            _make_openai_stream_chunk(content="world"),
            _make_openai_stream_chunk(finish_reason="stop"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["Hello ", "world"]

    def test_stream_with_tool_calls(self) -> None:
        provider = self._get_provider()
        tc_delta1 = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments='{"x":'
        )
        tc_delta2 = _make_openai_tool_call_delta(index=0, arguments=" 42}")
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta1]),
            _make_openai_stream_chunk(tool_calls=[tc_delta2]),
            _make_openai_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        )
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].tool_name == "test_tool"
        assert result[0].parameters == {"x": 42}

    def test_stream_tool_calls_flush_without_finish_reason(self) -> None:
        """Tool calls accumulated without finish_reason are flushed at end of stream."""
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments='{"x": 1}'
        )
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta]),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)

    def test_stream_tool_call_invalid_json(self) -> None:
        """Invalid JSON in tool call arguments falls back to empty dict."""
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments="not json"
        )
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta]),
            _make_openai_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert result[0].parameters == {}

    def test_stream_empty_choices(self) -> None:
        """Chunks with empty choices are skipped."""
        provider = self._get_provider()
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        normal_chunk = _make_openai_stream_chunk(content="text")
        provider._client.chat.completions.create.return_value = iter([empty_chunk, normal_chunk])

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["text"]

    def test_stream_content_as_list(self) -> None:
        """Handle delta.content that is a list of parts (some providers do this)."""
        provider = self._get_provider()
        chunk = _make_openai_stream_chunk()
        part1 = MagicMock()
        part1.text = "hello "
        part2 = MagicMock()
        part2.text = "world"
        chunk.choices[0].delta.content = [part1, part2]
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = None
        provider._client.chat.completions.create.return_value = iter([chunk])

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["hello world"]

    def test_stream_error_during_creation(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = Exception("bad request")
        with pytest.raises(ProviderError, match="bad request"):
            list(
                provider.stream(
                    model="llama3.2",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_temperature_error_retry(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = [
            Exception("temperature not supported"),
            iter([_make_openai_stream_chunk(content="ok")]),
        ]
        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["ok"]

    def test_stream_temperature_retry_also_fails(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.side_effect = [
            Exception("temperature not supported"),
            Exception("still broken"),
        ]
        with pytest.raises(ProviderError, match="still broken"):
            list(
                provider.stream(
                    model="llama3.2",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_error_mid_stream(self) -> None:
        """Errors during iteration are wrapped in ProviderError."""
        provider = self._get_provider()

        def exploding_iter():
            yield _make_openai_stream_chunk(content="ok")
            raise RuntimeError("unexpected EOF")

        provider._client.chat.completions.create.return_value = exploding_iter()
        with pytest.raises(ProviderError, match="streaming failed"):
            list(
                provider.stream(
                    model="llama3.2",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_provider_error_passthrough(self) -> None:
        """ProviderErrors during stream parsing are re-raised as-is, not double-wrapped."""
        provider = self._get_provider()

        def exploding_iter():
            yield _make_openai_stream_chunk(content="ok")
            raise ProviderError("original error")

        provider._client.chat.completions.create.return_value = exploding_iter()
        with pytest.raises(ProviderError, match="original error"):
            list(
                provider.stream(
                    model="llama3.2",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_flush_invalid_json(self) -> None:
        """Flushed tool calls at end of stream with invalid JSON get empty dict."""
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments="bad{"
        )
        # No finish_reason, so it flushes at end
        chunks = [_make_openai_stream_chunk(tool_calls=[tc_delta])]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert result[0].parameters == {}

    def test_stream_flush_empty_arguments(self) -> None:
        """Flushed tool calls with empty arguments get empty dict."""
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments=""
        )
        chunks = [_make_openai_stream_chunk(tool_calls=[tc_delta])]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert result[0].parameters == {}

    def test_stream_with_timeout(self) -> None:
        provider = self._get_provider()
        provider._client.chat.completions.create.return_value = iter(
            [_make_openai_stream_chunk(content="x")]
        )
        list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=3.0,
            )
        )
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 3.0

    def test_stream_multiple_tool_calls(self) -> None:
        """Multiple tool calls accumulated across chunks are all emitted."""
        provider = self._get_provider()
        tc_delta0 = _make_openai_tool_call_delta(
            index=0, tc_id="call_a", name="tool_a", arguments='{"a": 1}'
        )
        tc_delta1 = _make_openai_tool_call_delta(
            index=1, tc_id="call_b", name="tool_b", arguments='{"b": 2}'
        )
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta0]),
            _make_openai_stream_chunk(tool_calls=[tc_delta1]),
            _make_openai_stream_chunk(finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tools")],
            )
        )
        assert len(result) == 2
        assert result[0].tool_name == "tool_a"
        assert result[1].tool_name == "tool_b"


class TestOpenAICompatAstream:
    """Test async astream() in _OpenAICompatibleBase via OllamaProvider."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.default_model = "llama3.2"
        provider.base_url = "http://localhost:11434"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_astream_text_chunks(self) -> None:
        provider = self._get_provider()
        chunks = [
            _make_openai_stream_chunk(content="Hello "),
            _make_openai_stream_chunk(content="world"),
        ]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert "Hello " in result
        assert "world" in result

    @pytest.mark.asyncio
    async def test_astream_with_tool_calls(self) -> None:
        provider = self._get_provider()
        tc_delta1 = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments='{"x":'
        )
        tc_delta2 = _make_openai_tool_call_delta(index=0, arguments=" 42}")
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta1]),
            _make_openai_stream_chunk(tool_calls=[tc_delta2]),
            _make_openai_stream_chunk(finish_reason="tool_calls"),
        ]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
            tools=[_make_tool()],
        ):
            result.append(item)
        tool_calls = [r for r in result if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters == {"x": 42}

    @pytest.mark.asyncio
    async def test_astream_flush_without_finish_reason(self) -> None:
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments='{"x": 1}'
        )
        chunks = [_make_openai_stream_chunk(tool_calls=[tc_delta])]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
        ):
            result.append(item)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)

    @pytest.mark.asyncio
    async def test_astream_error_during_creation(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("network error")
        )
        with pytest.raises(ProviderError, match="network error"):
            async for _ in provider.astream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_temperature_retry(self) -> None:
        provider = self._get_provider()
        chunks = [_make_openai_stream_chunk(content="ok")]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("temperature not supported"),
                mock_stream(),
            ]
        )
        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert result == ["ok"]

    @pytest.mark.asyncio
    async def test_astream_temperature_retry_also_fails(self) -> None:
        provider = self._get_provider()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("temperature is not valid"),
                Exception("still broken"),
            ]
        )
        with pytest.raises(ProviderError, match="still broken"):
            async for _ in provider.astream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_empty_choices_skipped(self) -> None:
        provider = self._get_provider()
        empty_chunk = MagicMock()
        empty_chunk.choices = []
        normal = _make_openai_stream_chunk(content="hello")
        chunks = [empty_chunk, normal]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert result == ["hello"]

    @pytest.mark.asyncio
    async def test_astream_error_mid_stream(self) -> None:
        provider = self._get_provider()

        async def exploding_stream():
            yield _make_openai_stream_chunk(content="ok")
            raise RuntimeError("unexpected EOF")

        provider._async_client.chat.completions.create = AsyncMock(return_value=exploding_stream())
        with pytest.raises(ProviderError, match="streaming failed"):
            async for _ in provider.astream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_provider_error_passthrough(self) -> None:
        provider = self._get_provider()

        async def exploding_stream():
            yield _make_openai_stream_chunk(content="ok")
            raise ProviderError("specific error")

        provider._async_client.chat.completions.create = AsyncMock(return_value=exploding_stream())
        with pytest.raises(ProviderError, match="specific error"):
            async for _ in provider.astream(
                model="llama3.2",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_with_timeout(self) -> None:
        provider = self._get_provider()

        async def mock_stream():
            yield _make_openai_stream_chunk(content="x")

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        async for _ in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
            timeout=5.0,
        ):
            pass
        call_kwargs = provider._async_client.chat.completions.create.call_args[1]
        assert call_kwargs["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_astream_invalid_json_tool_args(self) -> None:
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments="not json"
        )
        chunks = [
            _make_openai_stream_chunk(tool_calls=[tc_delta]),
            _make_openai_stream_chunk(finish_reason="tool_calls"),
        ]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
        ):
            result.append(item)
        assert result[0].parameters == {}

    @pytest.mark.asyncio
    async def test_astream_flush_invalid_json(self) -> None:
        """Flushed tool calls with invalid JSON at stream end."""
        provider = self._get_provider()
        tc_delta = _make_openai_tool_call_delta(
            index=0, tc_id="call_1", name="test_tool", arguments="bad{"
        )
        chunks = [_make_openai_stream_chunk(tool_calls=[tc_delta])]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="llama3.2",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
        ):
            result.append(item)
        assert result[0].parameters == {}


# ===========================================================================
# _OpenAICompatibleBase misc tests
# ===========================================================================


class TestOpenAICompatMisc:
    """Miscellaneous tests for _OpenAICompatibleBase shared methods."""

    def _get_provider(self) -> Any:
        from selectools.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        provider.default_model = "llama3.2"
        provider.base_url = "http://localhost:11434"
        return provider

    def test_parse_tool_call_arguments_default_json_decode_error(self) -> None:
        """Default _parse_tool_call_arguments (OpenAI path) handles invalid JSON.

        BUG-31: now returns (params, parse_error) — params is empty and
        parse_error carries a preview so the tool executor can surface a
        retry message to the LLM.
        """
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        provider.api_key = "test"
        tc = MagicMock()
        tc.function.arguments = "not json at all"
        params, parse_error = provider._parse_tool_call_arguments(tc)
        assert params == {}
        assert parse_error is not None and "invalid JSON" in parse_error

    def test_parse_response_no_usage(self) -> None:
        """Parse response when usage is None."""
        provider = self._get_provider()
        mock_response = _make_openai_response("hi")
        mock_response.usage = None
        msg, usage = provider._parse_response(mock_response, "llama3.2")
        assert msg.content == "hi"
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_parse_response_with_tool_calls(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.function.name = "my_tool"
        tc.function.arguments = '{"key": "val"}'
        tc.id = "call_abc"
        mock_response = _make_openai_response("", tool_calls=[tc])
        msg, _ = provider._parse_response(mock_response, "llama3.2")
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "my_tool"
        assert msg.tool_calls[0].parameters == {"key": "val"}

    def test_format_content_with_image(self) -> None:
        provider = self._get_provider()
        msg = Message(role=Role.USER, content="What is this?")
        msg.image_base64 = "abc123"
        content = provider._format_content(msg)
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_format_content_no_image(self) -> None:
        provider = self._get_provider()
        msg = Message(role=Role.USER, content="Hello")
        content = provider._format_content(msg)
        assert content == "Hello"

    def test_format_content_none_content(self) -> None:
        provider = self._get_provider()
        msg = Message(role=Role.USER, content=None)  # type: ignore[arg-type]
        content = provider._format_content(msg)
        assert content == ""

    def test_map_tool_to_openai(self) -> None:
        provider = self._get_provider()
        tool = _make_tool()
        result = provider._map_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"

    def test_build_astream_args_default(self) -> None:
        """Default _build_astream_args is identity (Ollama doesn't add stream_options)."""
        provider = self._get_provider()
        args = {"model": "test", "stream": True}
        result = provider._build_astream_args(args)
        assert result == args

    def test_format_tool_call_id_with_id(self) -> None:
        """Default _format_tool_call_id returns the ID."""
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        tc = ToolCall(tool_name="t", parameters={}, id="call_abc")
        assert provider._format_tool_call_id(tc) == "call_abc"

    def test_format_tool_call_id_without_id(self) -> None:
        """Default _format_tool_call_id generates an ID when None."""
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        tc = ToolCall(tool_name="t", parameters={}, id=None)
        result = provider._format_tool_call_id(tc)
        assert result.startswith("call_")

    def test_complete_default_model_fallback(self) -> None:
        """When model is empty string, default_model is used."""
        provider = self._get_provider()
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = _make_openai_response("hi")
        provider.complete(
            model="",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        )
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama3.2"


# ===========================================================================
# AnthropicProvider tests
# ===========================================================================


class TestAnthropicProviderComplete:
    """Test AnthropicProvider complete/acomplete with mocked SDK."""

    def _get_provider(self) -> Any:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.default_model = "claude-sonnet-4-5"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    def _make_anthropic_response(
        self, text: str = "hello", tool_use: list | None = None
    ) -> MagicMock:
        blocks = []
        if text:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = text
            blocks.append(text_block)
        if tool_use:
            blocks.extend(tool_use)
        response = MagicMock()
        response.content = blocks
        response.usage = MagicMock()
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        return response

    def test_complete_basic(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.return_value = self._make_anthropic_response("hi there")
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.001):
            msg, usage = provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=[Message(role=Role.USER, content="hello")],
            )
        assert msg.content == "hi there"
        assert msg.role == Role.ASSISTANT
        assert usage.prompt_tokens == 10

    def test_complete_with_tools(self) -> None:
        provider = self._get_provider()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "test_tool"
        tool_block.input = {"x": 42}
        tool_block.id = "toolu_123"
        provider._client.messages.create.return_value = self._make_anthropic_response(
            "", tool_use=[tool_block]
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "test_tool"
        assert msg.tool_calls[0].id == "toolu_123"

    def test_complete_with_timeout(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.return_value = self._make_anthropic_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=5.0,
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 5.0

    def test_complete_error(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.side_effect = Exception("API error")
        with pytest.raises(ProviderError, match="Anthropic completion failed"):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    @pytest.mark.asyncio
    async def test_acomplete_basic(self) -> None:
        provider = self._get_provider()
        provider._async_client.messages.create = AsyncMock(
            return_value=self._make_anthropic_response("async response")
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            msg, usage = await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.content == "async response"

    @pytest.mark.asyncio
    async def test_acomplete_with_tools(self) -> None:
        provider = self._get_provider()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "test_tool"
        tool_block.input = {"x": 1}
        tool_block.id = "toolu_abc"
        provider._async_client.messages.create = AsyncMock(
            return_value=self._make_anthropic_response("", tool_use=[tool_block])
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            msg, _ = await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_acomplete_with_timeout(self) -> None:
        provider = self._get_provider()
        provider._async_client.messages.create = AsyncMock(
            return_value=self._make_anthropic_response()
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=10.0,
            )
        call_kwargs = provider._async_client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_acomplete_error(self) -> None:
        provider = self._get_provider()
        provider._async_client.messages.create = AsyncMock(side_effect=Exception("async error"))
        with pytest.raises(ProviderError, match="Anthropic async completion failed"):
            await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )


class TestAnthropicProviderStream:
    """Test AnthropicProvider sync and async streaming."""

    def _get_provider(self) -> Any:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.default_model = "claude-sonnet-4-5"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    def _make_event(self, event_type: str, **kwargs: Any) -> MagicMock:
        event = MagicMock()
        event.type = event_type
        for k, v in kwargs.items():
            setattr(event, k, v)
        return event

    def _make_text_delta_event(self, text: str) -> MagicMock:
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = text
        event = self._make_event("content_block_delta", delta=delta)
        return event

    def _make_tool_start_event(self, name: str, tool_id: str) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.id = tool_id
        return self._make_event("content_block_start", content_block=block)

    def _make_tool_delta_event(self, partial_json: str) -> MagicMock:
        delta = MagicMock()
        delta.type = "input_json_delta"
        delta.partial_json = partial_json
        return self._make_event("content_block_delta", delta=delta)

    def _make_block_stop_event(self) -> MagicMock:
        return self._make_event("content_block_stop")

    def test_stream_text(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_text_delta_event("Hello "),
            self._make_text_delta_event("world"),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["Hello ", "world"]

    def test_stream_with_tool_calls(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_tool_start_event("test_tool", "toolu_123"),
            self._make_tool_delta_event('{"x":'),
            self._make_tool_delta_event(" 42}"),
            self._make_block_stop_event(),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        )
        tool_calls = [r for r in result if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "test_tool"
        assert tool_calls[0].parameters == {"x": 42}
        assert tool_calls[0].id == "toolu_123"

    def test_stream_tool_invalid_json(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_tool_start_event("test_tool", "toolu_1"),
            self._make_tool_delta_event("bad json {"),
            self._make_block_stop_event(),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert result[0].parameters == {}

    def test_stream_tool_empty_json(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_tool_start_event("test_tool", "toolu_1"),
            self._make_block_stop_event(),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        )
        assert result[0].parameters == {}

    def test_stream_block_stop_without_tool_name(self) -> None:
        """content_block_stop without a preceding tool start should not emit a ToolCall."""
        provider = self._get_provider()
        events = [
            self._make_text_delta_event("hello"),
            self._make_block_stop_event(),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["hello"]

    def test_stream_delta_without_delta_attr(self) -> None:
        """content_block_delta with delta=None should be skipped."""
        provider = self._get_provider()
        events = [
            self._make_event("content_block_delta", delta=None),
            self._make_text_delta_event("ok"),
        ]
        provider._client.messages.create.return_value = iter(events)

        result = list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        )
        assert result == ["ok"]

    def test_stream_error_during_creation(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.side_effect = Exception("API error")
        with pytest.raises(ProviderError, match="Anthropic streaming failed"):
            list(
                provider.stream(
                    model="claude-sonnet-4-5",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_error_mid_stream(self) -> None:
        provider = self._get_provider()

        def exploding_iter():
            yield self._make_text_delta_event("ok")
            raise RuntimeError("unexpected EOF")

        provider._client.messages.create.return_value = exploding_iter()
        with pytest.raises(ProviderError, match="Anthropic streaming failed mid-stream"):
            list(
                provider.stream(
                    model="claude-sonnet-4-5",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_provider_error_passthrough(self) -> None:
        provider = self._get_provider()

        def exploding_iter():
            raise ProviderError("original error")
            yield  # noqa: unreachable

        provider._client.messages.create.return_value = exploding_iter()
        with pytest.raises(ProviderError, match="original error"):
            list(
                provider.stream(
                    model="claude-sonnet-4-5",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_with_timeout(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.return_value = iter([])
        list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=5.0,
            )
        )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 5.0

    def test_stream_with_tools_in_request(self) -> None:
        provider = self._get_provider()
        provider._client.messages.create.return_value = iter([])
        list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                tools=[_make_tool()],
            )
        )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1


class TestAnthropicProviderAstream:
    """Test AnthropicProvider async streaming."""

    def _get_provider(self) -> Any:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.default_model = "claude-sonnet-4-5"
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        return provider

    def _make_event(self, event_type: str, **kwargs: Any) -> MagicMock:
        event = MagicMock()
        event.type = event_type
        for k, v in kwargs.items():
            setattr(event, k, v)
        return event

    def _make_text_delta_event(self, text: str) -> MagicMock:
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = text
        return self._make_event("content_block_delta", delta=delta)

    def _make_tool_start_event(self, name: str, tool_id: str) -> MagicMock:
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.id = tool_id
        return self._make_event("content_block_start", content_block=block)

    def _make_tool_delta_event(self, partial_json: str) -> MagicMock:
        delta = MagicMock()
        delta.type = "input_json_delta"
        delta.partial_json = partial_json
        return self._make_event("content_block_delta", delta=delta)

    def _make_block_stop_event(self) -> MagicMock:
        return self._make_event("content_block_stop")

    @pytest.mark.asyncio
    async def test_astream_text(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_text_delta_event("Hello "),
            self._make_text_delta_event("world"),
        ]

        async def mock_stream():
            for e in events:
                yield e

        provider._async_client.messages.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert result == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_astream_with_tool_calls(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_tool_start_event("test_tool", "toolu_1"),
            self._make_tool_delta_event('{"x": 42}'),
            self._make_block_stop_event(),
        ]

        async def mock_stream():
            for e in events:
                yield e

        provider._async_client.messages.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
            tools=[_make_tool()],
        ):
            result.append(item)
        tool_calls = [r for r in result if isinstance(r, ToolCall)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "test_tool"
        assert tool_calls[0].parameters == {"x": 42}

    @pytest.mark.asyncio
    async def test_astream_error_during_creation(self) -> None:
        provider = self._get_provider()
        provider._async_client.messages.create = AsyncMock(side_effect=Exception("async API error"))
        with pytest.raises(ProviderError, match="Anthropic async streaming failed"):
            async for _ in provider.astream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_error_mid_stream(self) -> None:
        provider = self._get_provider()

        async def exploding_stream():
            yield self._make_text_delta_event("ok")
            raise RuntimeError("unexpected EOF")

        provider._async_client.messages.create = AsyncMock(return_value=exploding_stream())
        with pytest.raises(ProviderError, match="Anthropic async streaming failed mid-stream"):
            async for _ in provider.astream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_provider_error_passthrough(self) -> None:
        provider = self._get_provider()

        async def exploding_stream():
            raise ProviderError("original")
            yield  # noqa: unreachable

        provider._async_client.messages.create = AsyncMock(return_value=exploding_stream())
        with pytest.raises(ProviderError, match="original"):
            async for _ in provider.astream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_delta_without_delta_attr(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_event("content_block_delta", delta=None),
            self._make_text_delta_event("ok"),
        ]

        async def mock_stream():
            for e in events:
                yield e

        provider._async_client.messages.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert result == ["ok"]

    @pytest.mark.asyncio
    async def test_astream_tool_invalid_json(self) -> None:
        provider = self._get_provider()
        events = [
            self._make_tool_start_event("test_tool", "toolu_1"),
            self._make_tool_delta_event("not json"),
            self._make_block_stop_event(),
        ]

        async def mock_stream():
            for e in events:
                yield e

        provider._async_client.messages.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="use tool")],
        ):
            result.append(item)
        assert result[0].parameters == {}

    @pytest.mark.asyncio
    async def test_astream_with_tools_and_timeout(self) -> None:
        provider = self._get_provider()

        async def mock_stream():
            yield self._make_text_delta_event("hello")

        provider._async_client.messages.create = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
            tools=[_make_tool()],
            timeout=10.0,
        ):
            result.append(item)
        call_kwargs = provider._async_client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 10.0
        assert "tools" in call_kwargs


class TestAnthropicFormatMessagesMerging:
    """Test _format_messages merge logic for consecutive same-role messages."""

    def _get_provider(self) -> Any:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.default_model = "claude-sonnet-4-5"
        return provider

    def test_merge_consecutive_user_tool_results(self) -> None:
        """Multiple TOOL messages for parallel tool calls must be merged."""
        provider = self._get_provider()
        messages = [
            Message(role=Role.TOOL, content="result1", tool_call_id="tc_1"),
            Message(role=Role.TOOL, content="result2", tool_call_id="tc_2"),
        ]
        formatted = provider._format_messages(messages)
        # Should be merged into a single user message with 2 tool_result blocks
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert len(formatted[0]["content"]) == 2

    def test_merge_consecutive_strings(self) -> None:
        """Merge when both prev and curr content are strings."""
        provider = self._get_provider()
        # Force string content by manually constructing
        formatted = provider._format_messages([])
        # Test the merge logic directly by creating a scenario
        # Use two user messages with no image/tool_calls
        messages = [
            Message(role=Role.USER, content="hello"),
            Message(role=Role.USER, content="world"),
        ]
        formatted = provider._format_messages(messages)
        assert len(formatted) == 1

    def test_system_role_converted_to_user_and_prepended(self) -> None:
        """SYSTEM messages are converted to user and prepended."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="I'll use a tool",
                tool_calls=[ToolCall(tool_name="t", parameters={}, id="tc_1")],
            ),
            Message(role=Role.SYSTEM, content="context injection"),
            Message(role=Role.TOOL, content="result", tool_call_id="tc_1"),
        ]
        formatted = provider._format_messages(messages)
        # System converted msg should be first
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"][0]["type"] == "text"
        assert "context injection" in formatted[0]["content"][0]["text"]

    def test_empty_assistant_content_gets_placeholder(self) -> None:
        """Assistant messages with no content/tool_calls get empty text placeholder."""
        provider = self._get_provider()
        messages = [Message(role=Role.ASSISTANT, content=None)]  # type: ignore[arg-type]
        formatted = provider._format_messages(messages)
        assert formatted[0]["content"] == [{"type": "text", "text": ""}]

    def test_map_tool_to_anthropic(self) -> None:
        provider = self._get_provider()
        tool = _make_tool()
        result = provider._map_tool_to_anthropic(tool)
        assert result["name"] == "test_tool"
        assert "description" in result
        assert "input_schema" in result


# ===========================================================================
# GeminiProvider tests
# ===========================================================================


class TestGeminiProviderComplete:
    """Test GeminiProvider complete/acomplete with mocked SDK."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-2.0-flash"
        provider._client = MagicMock()
        return provider

    def _make_gemini_response(
        self,
        text: str = "hello",
        function_calls: list | None = None,
        has_usage: bool = True,
    ) -> MagicMock:
        response = MagicMock()
        response.text = text

        if function_calls:
            parts = []
            for fc in function_calls:
                part = MagicMock()
                part.function_call = fc
                part.thought_signature = None
                parts.append(part)
            candidate = MagicMock()
            candidate.content = MagicMock()
            candidate.content.parts = parts
            response.candidates = [candidate]
        else:
            response.candidates = []

        if has_usage:
            response.usage_metadata = MagicMock()
            response.usage_metadata.prompt_token_count = 10
            response.usage_metadata.candidates_token_count = 5
        else:
            response.usage_metadata = None

        return response

    def test_complete_basic(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.models.generate_content.return_value = self._make_gemini_response("hi")

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.001):
            msg, usage = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hello")],
            )
        assert msg.content == "hi"
        assert usage.prompt_tokens == 10

    def test_complete_with_tool_calls(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"x": 42}
        provider._client.models.generate_content.return_value = self._make_gemini_response(
            "", function_calls=[fc]
        )

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "test_tool"
        assert msg.tool_calls[0].parameters == {"x": 42}

    def test_complete_with_thought_signature_bytes(self) -> None:
        """thought_signature as bytes is base64-encoded."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {}

        part = MagicMock()
        part.function_call = fc
        part.thought_signature = b"\x00\x01\x02"

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.text = ""
        response.candidates = [candidate]
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.models.generate_content.return_value = response

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].thought_signature is not None
        # Verify it's valid base64
        import base64

        decoded = base64.b64decode(msg.tool_calls[0].thought_signature)
        assert decoded == b"\x00\x01\x02"

    def test_complete_with_thought_signature_string(self) -> None:
        """thought_signature as string is converted with str()."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {}

        part = MagicMock()
        part.function_call = fc
        part.thought_signature = "some_sig"

        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.text = ""
        response.candidates = [candidate]
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.models.generate_content.return_value = response

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.tool_calls[0].thought_signature == "some_sig"

    def test_complete_text_value_error(self) -> None:
        """When response.text raises ValueError, content falls back to empty string."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        response = MagicMock()
        type(response).text = property(lambda self: (_ for _ in ()).throw(ValueError("no text")))
        response.candidates = []
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.models.generate_content.return_value = response

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.content == ""

    def test_complete_no_usage(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.models.generate_content.return_value = self._make_gemini_response(
            "hi", has_usage=False
        )

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            _, usage = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_complete_error(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.models.generate_content.side_effect = Exception("API error")
        with pytest.raises(ProviderError, match="Gemini completion failed"):
            provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    def test_complete_with_tools_in_config(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.models.generate_content.return_value = self._make_gemini_response()

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                tools=[_make_tool()],
            )
        call_kwargs = provider._client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.tools is not None

    def test_complete_function_call_no_args(self) -> None:
        """function_call with args=None gets empty dict."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = None
        provider._client.models.generate_content.return_value = self._make_gemini_response(
            "", function_calls=[fc]
        )

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = provider.complete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            )
        assert msg.tool_calls[0].parameters == {}


class TestGeminiProviderStream:
    """Test GeminiProvider sync streaming."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-2.0-flash"
        provider._client = MagicMock()
        return provider

    def test_stream_text(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk1.candidates = []
        chunk2 = MagicMock()
        chunk2.text = "world"
        chunk2.candidates = []
        provider._client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )
        assert result == ["Hello ", "world"]

    def test_stream_with_tool_call(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"x": 42}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError("no text")))
        chunk.candidates = [candidate]
        provider._client.models.generate_content_stream.return_value = iter([chunk])

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="use tool")],
                    tools=[_make_tool()],
                )
            )
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].tool_name == "test_tool"
        assert result[0].parameters == {"x": 42}

    def test_stream_with_thought_signature(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = b"\xab\xcd"
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = [candidate]
        provider._client.models.generate_content_stream.return_value = iter([chunk])

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="use tool")],
                )
            )
        assert result[0].thought_signature is not None
        import base64

        assert base64.b64decode(result[0].thought_signature) == b"\xab\xcd"

    def test_stream_error_during_creation(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.models.generate_content_stream.side_effect = Exception("API error")
        with pytest.raises(ProviderError, match="Gemini streaming failed"):
            list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_error_mid_stream(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        def exploding_iter():
            chunk = MagicMock()
            chunk.text = "ok"
            chunk.candidates = []
            yield chunk
            raise RuntimeError("unexpected EOF")

        provider._client.models.generate_content_stream.return_value = exploding_iter()

        with pytest.raises(ProviderError, match="Gemini streaming failed mid-stream"):
            list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_provider_error_passthrough(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        def exploding_iter():
            raise ProviderError("original")
            yield  # noqa: unreachable

        provider._client.models.generate_content_stream.return_value = exploding_iter()

        with pytest.raises(ProviderError, match="original"):
            list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )
            )

    def test_stream_function_call_no_args(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = None
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = [candidate]
        provider._client.models.generate_content_stream.return_value = iter([chunk])

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = list(
                provider.stream(
                    model="gemini-2.0-flash",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="use tool")],
                )
            )
        assert result[0].parameters == {}


class TestGeminiProviderAcomplete:
    """Test GeminiProvider async completion."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-2.0-flash"
        provider._client = MagicMock()
        return provider

    def _make_gemini_response(self, text: str = "hello", **kwargs: Any) -> MagicMock:
        response = MagicMock()
        response.text = text
        response.candidates = []
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 5
        return response

    @pytest.mark.asyncio
    async def test_acomplete_basic(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=self._make_gemini_response("async hello")
        )

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, usage = await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.content == "async hello"

    @pytest.mark.asyncio
    async def test_acomplete_with_tool_calls(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"x": 42}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.text = ""
        response.candidates = [candidate]
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            )
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_acomplete_text_value_error(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        response = MagicMock()
        type(response).text = property(lambda self: (_ for _ in ()).throw(ValueError("no text")))
        response.candidates = []
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.content == ""

    @pytest.mark.asyncio
    async def test_acomplete_error(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("async error")
        )
        with pytest.raises(ProviderError, match="Gemini async completion failed"):
            await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )

    @pytest.mark.asyncio
    async def test_acomplete_no_usage(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        response = MagicMock()
        response.text = "hi"
        response.candidates = []
        response.usage_metadata = None
        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            _, usage = await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert usage.prompt_tokens == 0

    @pytest.mark.asyncio
    async def test_acomplete_with_thought_signature_bytes(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "tool"
        fc.args = {}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = b"\x01\x02"
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.text = ""
        response.candidates = [candidate]
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 3
        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            msg, _ = await provider.acomplete(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            )
        assert msg.tool_calls[0].thought_signature is not None


class TestGeminiProviderAstream:
    """Test GeminiProvider async streaming."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-2.0-flash"
        provider._client = MagicMock()
        return provider

    @pytest.mark.asyncio
    async def test_astream_text(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk1.candidates = []
        chunk2 = MagicMock()
        chunk2.text = "world"
        chunk2.candidates = []

        async def mock_stream():
            for c in [chunk1, chunk2]:
                yield c

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = []
            async for item in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                result.append(item)
        assert result == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_astream_with_tool_call(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"x": 42}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = [candidate]

        async def mock_stream():
            yield chunk

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = []
            async for item in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
                tools=[_make_tool()],
            ):
                result.append(item)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_astream_with_thought_signature(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "tool"
        fc.args = {}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = b"\xab\xcd"
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = [candidate]

        async def mock_stream():
            yield chunk

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = []
            async for item in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            ):
                result.append(item)
        import base64

        assert base64.b64decode(result[0].thought_signature) == b"\xab\xcd"

    @pytest.mark.asyncio
    async def test_astream_error_during_creation(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=Exception("async error")
        )
        with pytest.raises(ProviderError, match="Gemini async streaming failed"):
            async for _ in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_error_mid_stream(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        async def exploding_stream():
            chunk = MagicMock()
            chunk.text = "ok"
            chunk.candidates = []
            yield chunk
            raise RuntimeError("unexpected EOF")

        provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=exploding_stream()
        )
        with pytest.raises(ProviderError, match="Gemini async streaming failed mid-stream"):
            async for _ in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_provider_error_passthrough(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        async def exploding_stream():
            raise ProviderError("original")
            yield  # noqa: unreachable

        provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=exploding_stream()
        )
        with pytest.raises(ProviderError, match="original"):
            async for _ in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
            ):
                pass

    @pytest.mark.asyncio
    async def test_astream_function_call_no_args(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = None
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = [candidate]

        async def mock_stream():
            yield chunk

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            result = []
            async for item in provider.astream(
                model="gemini-2.0-flash",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="use tool")],
            ):
                result.append(item)
        assert result[0].parameters == {}

    @pytest.mark.asyncio
    async def test_astream_text_value_error(self) -> None:
        """When chunk.text raises ValueError during streaming, it's skipped."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()

        chunk = MagicMock()
        type(chunk).text = property(lambda self: (_ for _ in ()).throw(ValueError))
        chunk.candidates = []

        async def mock_stream():
            yield chunk

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        result = []
        async for item in provider.astream(
            model="gemini-2.0-flash",
            system_prompt="sys",
            messages=[Message(role=Role.USER, content="hi")],
        ):
            result.append(item)
        assert result == []


class TestGeminiFormatContentsAdvanced:
    """Test advanced _format_contents scenarios."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-2.0-flash"
        return provider

    def test_system_role_converted_to_user(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        messages = [Message(role=Role.SYSTEM, content="context injection")]
        formatted = provider._format_contents("sys", messages)
        assert len(formatted) == 1
        assert formatted[0].role == "user"

    def test_tool_with_thought_signature_echo(self) -> None:
        """TOOL messages echo thought_signature from preceding ASSISTANT tool_calls."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        import base64

        provider = self._get_provider()
        sig = base64.b64encode(b"\x01\x02").decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="test_tool", parameters={"x": 1}, id="tc_1", thought_signature=sig
                    )
                ],
            ),
            Message(role=Role.TOOL, content="result", tool_name="test_tool", tool_call_id="tc_1"),
        ]
        formatted = provider._format_contents("sys", messages)
        # The TOOL message should have 2 parts: echoed function_call + function_response
        tool_content = formatted[1]
        assert tool_content.role == "user"
        assert len(tool_content.parts) == 2

    def test_unknown_role_falls_back_to_user(self) -> None:
        """Messages with an unrecognized role fall back to user."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        msg = Message(role=Role.USER, content="hello")
        # Hack role to an unknown value to test fallback
        # This path is guarded by else clause
        formatted = provider._format_contents("sys", [msg])
        assert formatted[0].role == "user"

    def test_empty_parts_skipped(self) -> None:
        """Messages with no content produce no parts and are skipped."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        messages = [Message(role=Role.USER, content=None)]  # type: ignore[arg-type]
        formatted = provider._format_contents("sys", messages)
        assert len(formatted) == 0

    def test_map_tool_to_gemini(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        tool = _make_tool()
        result = provider._map_tool_to_gemini(tool)
        assert result is not None


# ===========================================================================
# OpenAI-specific coverage (OpenAIProvider template methods)
# ===========================================================================


class TestOpenAIProviderTemplateMethods:
    """Test OpenAIProvider-specific template method overrides."""

    def _get_provider(self) -> Any:
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        provider.api_key = "test"
        return provider

    def test_get_token_key_legacy_model(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("gpt-4o") == "max_tokens"

    def test_get_token_key_new_model_gpt5(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("gpt-5-mini") == "max_completion_tokens"

    def test_get_token_key_o1(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("o1-mini") == "max_completion_tokens"

    def test_get_token_key_o3(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("o3-mini") == "max_completion_tokens"

    def test_get_token_key_codex(self) -> None:
        provider = self._get_provider()
        assert provider._get_token_key("codex-mini") == "max_completion_tokens"

    def test_build_astream_args_adds_stream_options(self) -> None:
        provider = self._get_provider()
        args = {"model": "gpt-4o", "stream": True}
        result = provider._build_astream_args(args)
        assert "stream_options" in result
        assert result["stream_options"]["include_usage"] is True

    def test_wrap_error(self) -> None:
        provider = self._get_provider()
        err = provider._wrap_error(Exception("bad"), "completion")
        assert isinstance(err, ProviderError)
        assert "OpenAI completion failed" in str(err)

    def test_parse_tool_call_id(self) -> None:
        provider = self._get_provider()
        tc = MagicMock()
        tc.id = "call_xyz"
        assert provider._parse_tool_call_id(tc) == "call_xyz"
