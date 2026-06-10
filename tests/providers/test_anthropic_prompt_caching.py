"""
Tests for AnthropicProvider prompt caching (issue #57).

Covers the opt-in ``cache_system`` / ``cache_tools`` flags across all four
request paths (complete, acomplete, stream, astream) and the new optional
cache token fields on UsageStats.

Mocks the SDK client directly — never calls real APIs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.tools.base import Tool, ToolParameter
from selectools.types import Message, Role
from selectools.usage import UsageStats

EPHEMERAL = {"type": "ephemeral"}


def _make_tool(name: str = "test_tool") -> Tool:
    return Tool(
        name=name,
        description=f"{name} description",
        parameters=[
            ToolParameter(name="x", param_type=int, description="An integer"),
        ],
        function=lambda x: str(x),
    )


def _get_provider(cache_system: bool = False, cache_tools: bool = False) -> Any:
    from selectools.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.default_model = "claude-sonnet-4-5"
    provider._client = MagicMock()
    provider._async_client = AsyncMock()
    provider.cache_system = cache_system
    provider.cache_tools = cache_tools
    return provider


def _make_response(
    text: str = "hello",
    cache_creation_input_tokens: Any = None,
    cache_read_input_tokens: Any = None,
) -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens
    response.usage = usage
    return response


def _messages() -> list:
    return [Message(role=Role.USER, content="hi")]


# ---------------------------------------------------------------------------
# Defaults: flags off -> behavior unchanged
# ---------------------------------------------------------------------------


class TestCachingDisabledByDefault:
    def test_init_defaults_are_false(self) -> None:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")
        assert provider.cache_system is False
        assert provider.cache_tools is False

    def test_complete_system_stays_plain_string(self) -> None:
        provider = _get_provider()
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=_messages(),
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_complete_tools_have_no_cache_control(self) -> None:
        provider = _get_provider()
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
                tools=[_make_tool("a"), _make_tool("b")],
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        for tool in call_kwargs["tools"]:
            assert "cache_control" not in tool


# ---------------------------------------------------------------------------
# cache_system=True -> block form with ephemeral marker
# ---------------------------------------------------------------------------


class TestCacheSystem:
    def _assert_system_block(self, call_kwargs: dict) -> None:
        assert call_kwargs["system"] == [
            {"type": "text", "text": "You are helpful.", "cache_control": EPHEMERAL}
        ]

    def test_complete(self) -> None:
        provider = _get_provider(cache_system=True)
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=_messages(),
            )
        self._assert_system_block(provider._client.messages.create.call_args[1])

    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        provider = _get_provider(cache_system=True)
        provider._async_client.messages.create = AsyncMock(return_value=_make_response())
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=_messages(),
            )
        self._assert_system_block(provider._async_client.messages.create.call_args[1])

    def test_stream(self) -> None:
        provider = _get_provider(cache_system=True)
        provider._client.messages.create.return_value = iter([])
        list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=_messages(),
            )
        )
        self._assert_system_block(provider._client.messages.create.call_args[1])

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        async def _empty_aiter() -> Any:
            return
            yield

        provider = _get_provider(cache_system=True)
        provider._async_client.messages.create = AsyncMock(return_value=_empty_aiter())
        async for _ in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="You are helpful.",
            messages=_messages(),
        ):
            pass
        self._assert_system_block(provider._async_client.messages.create.call_args[1])


# ---------------------------------------------------------------------------
# cache_tools=True -> marker on LAST tool only
# ---------------------------------------------------------------------------


class TestCacheTools:
    def _assert_marker_on_last_tool_only(self, call_kwargs: dict) -> None:
        tools = call_kwargs["tools"]
        assert len(tools) == 3
        for tool in tools[:-1]:
            assert "cache_control" not in tool
        assert tools[-1]["cache_control"] == EPHEMERAL

    def _three_tools(self) -> list:
        return [_make_tool("a"), _make_tool("b"), _make_tool("c")]

    def test_complete(self) -> None:
        provider = _get_provider(cache_tools=True)
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
                tools=self._three_tools(),
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        self._assert_marker_on_last_tool_only(call_kwargs)
        # system stays a plain string when only cache_tools is enabled
        assert call_kwargs["system"] == "sys"

    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        provider = _get_provider(cache_tools=True)
        provider._async_client.messages.create = AsyncMock(return_value=_make_response())
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
                tools=self._three_tools(),
            )
        self._assert_marker_on_last_tool_only(provider._async_client.messages.create.call_args[1])

    def test_stream(self) -> None:
        provider = _get_provider(cache_tools=True)
        provider._client.messages.create.return_value = iter([])
        list(
            provider.stream(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
                tools=self._three_tools(),
            )
        )
        self._assert_marker_on_last_tool_only(provider._client.messages.create.call_args[1])

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        async def _empty_aiter() -> Any:
            return
            yield

        provider = _get_provider(cache_tools=True)
        provider._async_client.messages.create = AsyncMock(return_value=_empty_aiter())
        async for _ in provider.astream(
            model="claude-sonnet-4-5",
            system_prompt="sys",
            messages=_messages(),
            tools=self._three_tools(),
        ):
            pass
        self._assert_marker_on_last_tool_only(provider._async_client.messages.create.call_args[1])

    def test_no_tools_no_crash(self) -> None:
        provider = _get_provider(cache_tools=True)
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# Cache token usage surfaced on UsageStats
# ---------------------------------------------------------------------------


class TestCacheUsageStats:
    def test_usagestats_fields_default_none(self) -> None:
        stats = UsageStats()
        assert stats.cache_creation_input_tokens is None
        assert stats.cache_read_input_tokens is None

    def test_complete_populates_cache_tokens(self) -> None:
        provider = _get_provider(cache_system=True)
        provider._client.messages.create.return_value = _make_response(
            cache_creation_input_tokens=1234,
            cache_read_input_tokens=5678,
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            _, usage = provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
            )
        assert usage.cache_creation_input_tokens == 1234
        assert usage.cache_read_input_tokens == 5678

    @pytest.mark.asyncio
    async def test_acomplete_populates_cache_tokens(self) -> None:
        provider = _get_provider(cache_system=True)
        provider._async_client.messages.create = AsyncMock(
            return_value=_make_response(
                cache_creation_input_tokens=100,
                cache_read_input_tokens=200,
            )
        )
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            _, usage = await provider.acomplete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
            )
        assert usage.cache_creation_input_tokens == 100
        assert usage.cache_read_input_tokens == 200

    def test_complete_cache_tokens_absent_stay_none(self) -> None:
        """SDK responses without cache fields (or non-int mocks) leave None."""
        provider = _get_provider()
        response = _make_response()
        # Simulate an SDK usage object without cache attributes.
        del response.usage.cache_creation_input_tokens
        del response.usage.cache_read_input_tokens
        provider._client.messages.create.return_value = response
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            _, usage = provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="sys",
                messages=_messages(),
            )
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None


# ---------------------------------------------------------------------------
# Combined flags
# ---------------------------------------------------------------------------


class TestCacheSystemAndTools:
    def test_complete_both_flags(self) -> None:
        provider = _get_provider(cache_system=True, cache_tools=True)
        provider._client.messages.create.return_value = _make_response()
        with patch("selectools.providers.anthropic_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="claude-sonnet-4-5",
                system_prompt="You are helpful.",
                messages=_messages(),
                tools=[_make_tool("a"), _make_tool("b")],
            )
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["system"] == [
            {"type": "text", "text": "You are helpful.", "cache_control": EPHEMERAL}
        ]
        tools = call_kwargs["tools"]
        assert "cache_control" not in tools[0]
        assert tools[-1]["cache_control"] == EPHEMERAL
