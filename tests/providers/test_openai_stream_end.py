"""OpenAI-compatible streams end with a StreamEnd carrying the usage chunk.

The Chat Completions API sends usage as a final chunk with an empty ``choices``
list when ``stream_options.include_usage`` is set. Before this, astream()
requested it and then discarded it, and stream() never asked for it.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import MagicMock

import pytest

from selectools.providers.openai_provider import OpenAIProvider
from selectools.types import StreamEnd, ToolCall

_MODEL = "gpt-5-mini"


def _provider() -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    return provider


def _text_chunk(text: str, finish: Any = None) -> MagicMock:
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=delta, finish_reason=finish)]
    return chunk


def _usage_chunk(prompt: int, completion: int, cached: Any = None) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    if cached is None:
        usage.prompt_tokens_details = None
    else:
        usage.prompt_tokens_details = MagicMock(cached_tokens=cached)
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = usage
    return chunk


async def _aiter(items: List[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _stream(provider: OpenAIProvider, chunks: List[Any]) -> List[Any]:
    provider._client.chat.completions.create.return_value = iter(chunks)
    return list(
        provider.stream(model=_MODEL, system_prompt="s", messages=[], tools=None, max_tokens=10)
    )


async def _astream(provider: OpenAIProvider, chunks: List[Any]) -> List[Any]:
    async def create(**kwargs: Any) -> AsyncIterator[Any]:
        return _aiter(chunks)

    provider._async_client.chat.completions.create = create
    return [
        item
        async for item in provider.astream(
            model=_MODEL, system_prompt="s", messages=[], tools=None, max_tokens=10
        )
    ]


def test_sync_stream_yields_text_then_stream_end_with_usage() -> None:
    provider = _provider()

    items = _stream(provider, [_text_chunk("Hel"), _text_chunk("lo", "stop"), _usage_chunk(5, 3)])

    assert items[:2] == ["Hel", "lo"]
    end = items[-1]
    assert isinstance(end, StreamEnd)
    assert end.usage is not None
    assert (
        end.usage.prompt_tokens,
        end.usage.completion_tokens,
        end.usage.total_tokens,
    ) == (
        5,
        3,
        8,
    )
    assert end.usage.model == _MODEL
    assert end.usage.provider == "openai"
    assert end.usage.cost_usd == provider._calculate_cost(_MODEL, 5, 3, cached_input_tokens=0)


def test_sync_stream_requests_usage_in_stream_options() -> None:
    provider = _provider()

    _stream(provider, [_text_chunk("x", "stop"), _usage_chunk(1, 1)])

    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_async_stream_yields_text_then_stream_end_with_usage() -> None:
    provider = _provider()

    items = await _astream(
        provider, [_text_chunk("Hel"), _text_chunk("lo", "stop"), _usage_chunk(5, 3)]
    )

    assert items[:2] == ["Hel", "lo"]
    end = items[-1]
    assert isinstance(end, StreamEnd)
    assert end.usage is not None
    assert end.usage.total_tokens == 8
    assert end.usage.model == _MODEL


def test_stream_end_carries_cached_prompt_tokens() -> None:
    provider = _provider()

    end = _stream(provider, [_text_chunk("x", "stop"), _usage_chunk(10, 2, cached=4)])[-1]

    assert isinstance(end, StreamEnd)
    assert end.usage is not None
    assert end.usage.cache_read_input_tokens == 4


def test_stream_without_usage_chunk_yields_no_stream_end() -> None:
    provider = _provider()

    items = _stream(provider, [_text_chunk("x", "stop")])

    assert items == ["x"]
    assert not any(isinstance(i, StreamEnd) for i in items)


def test_stream_end_comes_after_tool_calls() -> None:
    provider = _provider()
    delta = MagicMock()
    delta.content = None
    fn = MagicMock()
    fn.name = "t"
    fn.arguments = '{"a": 1}'
    delta.tool_calls = [MagicMock(index=0, id="call_1", function=fn)]
    tool_chunk = MagicMock()
    tool_chunk.choices = [MagicMock(delta=delta, finish_reason="tool_calls")]

    items = _stream(provider, [tool_chunk, _usage_chunk(2, 2)])

    assert isinstance(items[0], ToolCall)
    assert isinstance(items[-1], StreamEnd)
