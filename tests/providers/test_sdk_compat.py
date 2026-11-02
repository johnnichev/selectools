"""
Wire-level compatibility tests against the real ``openai`` / ``anthropic`` SDKs.

The other provider tests replace ``_client`` with a MagicMock, so they accept
any kwargs and cannot notice when an SDK major drops a parameter from a
method signature (anthropic 1.0 removed ``temperature`` from
``messages.create``). These tests keep the real SDK clients and swap only the
HTTP transport, so every request goes through the SDK's own signature and
serialization and the captured JSON body is what would hit the wire.

Works on both sides of each SDK's httpx -> httpx2 switch: the transport class
is taken from whichever HTTP library the installed SDK builds on.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List

import pytest

from selectools.tools.base import Tool, ToolParameter
from selectools.types import Message, Role, ToolCall

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")


def _http_module(sdk: ModuleType) -> ModuleType:
    """Return the httpx-compatible module (``httpx`` or ``httpx2``) the SDK uses."""
    base = sdk.DefaultHttpxClient.__mro__[1]
    return sys.modules[base.__module__.split(".")[0]]


def _weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters=[ToolParameter(name="city", param_type=str, description="City name")],
        function=lambda city: f"sunny in {city}",
    )


def _sse(events: List[Dict[str, Any]], *, named: bool, done: bool = False) -> bytes:
    lines = []
    for event in events:
        if named:
            lines.append(f"event: {event['type']}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    if done:
        lines += ["data: [DONE]", ""]
    return ("\n".join(lines) + "\n").encode()


class _Recorder:
    """Records request bodies and serves canned JSON or SSE responses."""

    def __init__(self, http: ModuleType, json_body: Dict[str, Any], sse_body: bytes) -> None:
        self.http = http
        self.json_body = json_body
        self.sse_body = sse_body
        self.requests: List[Dict[str, Any]] = []
        self.headers: List[Dict[str, str]] = []

    def __call__(self, request: Any) -> Any:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        if body.get("stream"):
            return self.http.Response(
                200, headers={"content-type": "text/event-stream"}, content=self.sse_body
            )
        return self.http.Response(200, json=self.json_body)


def _collect(gen: Any) -> List[Any]:
    return list(gen)


def _acollect(agen: Any) -> List[Any]:
    async def run() -> List[Any]:
        return [item async for item in agen]

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

_ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 7,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 4,
    },
}

_ANTHROPIC_EVENTS = [
    {
        "type": "message_start",
        "message": {**_ANTHROPIC_MESSAGE, "content": [], "stop_reason": None},
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]


@pytest.fixture
def anthropic_provider() -> Any:
    from selectools.providers.anthropic_provider import AnthropicProvider

    http = _http_module(anthropic)
    recorder = _Recorder(http, _ANTHROPIC_MESSAGE, _sse(_ANTHROPIC_EVENTS, named=True))
    provider = AnthropicProvider(api_key="test-key", cache_system=True, cache_tools=True)
    provider._client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=http.MockTransport(recorder)),
    )
    provider._async_client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=http.MockTransport(recorder)),
    )
    provider.recorder = recorder
    return provider


def _anthropic_kwargs() -> Dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "system_prompt": "Be brief.",
        "messages": [
            Message(role=Role.USER, content="Weather in Paris?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(tool_name="get_weather", parameters={"city": "Rome"}, id="t0")
                ],
            ),
            Message(role=Role.TOOL, content="rainy", tool_call_id="t0"),
        ],
        "tools": [_weather_tool()],
        "temperature": 0.3,
        "max_tokens": 64,
        "timeout": 5.0,
    }


def _assert_anthropic_request(body: Dict[str, Any], *, stream: bool) -> None:
    assert body["model"] == "claude-sonnet-4-6"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 64
    assert body["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["tools"][0]["name"] == "get_weather"
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][-1]["content"][0]["type"] == "tool_result"
    assert "extra_body" not in body
    assert body.get("stream", False) is stream


def test_anthropic_complete_through_real_sdk(anthropic_provider: Any) -> None:
    msg, usage = anthropic_provider.complete(**_anthropic_kwargs())

    _assert_anthropic_request(anthropic_provider.recorder.requests[-1], stream=False)
    assert msg.content == "Checking."
    assert msg.tool_calls[0].tool_name == "get_weather"
    assert msg.tool_calls[0].parameters == {"city": "Paris"}
    assert (usage.prompt_tokens, usage.completion_tokens) == (12, 7)
    assert usage.cache_creation_input_tokens == 3
    assert usage.cache_read_input_tokens == 4


def test_anthropic_sdk_sends_an_api_version_that_accepts_temperature(
    anthropic_provider: Any,
) -> None:
    """anthropic>=2 sends API version 2026-10-21, which returns 400 for any
    ``temperature`` / ``top_p`` / ``top_k``, even the default. The provider
    sends ``temperature=0.0`` by default, so the supported SDK range must stop
    below 2.0 until the provider handles that version."""
    anthropic_provider.complete(**_anthropic_kwargs())

    assert anthropic_provider.recorder.headers[-1]["anthropic-version"] == "2023-06-01"


def test_anthropic_acomplete_through_real_sdk(anthropic_provider: Any) -> None:
    msg, _ = asyncio.run(anthropic_provider.acomplete(**_anthropic_kwargs()))

    _assert_anthropic_request(anthropic_provider.recorder.requests[-1], stream=False)
    assert msg.tool_calls[0].id == "toolu_1"


@pytest.mark.parametrize("collect", [_collect, _acollect], ids=["stream", "astream"])
def test_anthropic_streaming_through_real_sdk(
    anthropic_provider: Any, collect: Callable[[Any], List[Any]]
) -> None:
    method = anthropic_provider.stream if collect is _collect else anthropic_provider.astream
    chunks = collect(method(**_anthropic_kwargs()))

    _assert_anthropic_request(anthropic_provider.recorder.requests[-1], stream=True)
    assert [c for c in chunks if isinstance(c, str)] == ["Hi"]
    tool_calls = [c for c in chunks if isinstance(c, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].parameters == {"city": "Paris"}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

_OPENAI_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-5-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
    },
}


def _openai_chunk(choices: List[Dict[str, Any]], usage: Any = None) -> Dict[str, Any]:
    chunk: Dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5-mini",
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


_OPENAI_CHUNKS = [
    _openai_chunk([{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}]),
    _openai_chunk(
        [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": '},
                        }
                    ]
                },
            }
        ]
    ),
    _openai_chunk(
        [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
            }
        ]
    ),
    _openai_chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    _openai_chunk([], usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}),
]


@pytest.fixture
def openai_provider() -> Any:
    from selectools.providers.openai_provider import OpenAIProvider

    http = _http_module(openai)
    recorder = _Recorder(http, _OPENAI_COMPLETION, _sse(_OPENAI_CHUNKS, named=False, done=True))
    provider = OpenAIProvider(api_key="test-key")
    provider._client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=http.MockTransport(recorder)),
    )
    provider._async_client = openai.AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=http.MockTransport(recorder)),
    )
    provider.recorder = recorder
    return provider


_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def _openai_kwargs() -> Dict[str, Any]:
    return {
        "model": "gpt-5-mini",
        "system_prompt": "Be brief.",
        "messages": [Message(role=Role.USER, content="Weather in Paris?")],
        "tools": [_weather_tool()],
        "temperature": 0.0,
        "max_tokens": 64,
        "timeout": 5.0,
        "response_format": _SCHEMA,
    }


def _assert_openai_request(body: Dict[str, Any], *, stream: bool) -> None:
    assert body["model"] == "gpt-5-mini"
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    assert body["temperature"] == 0.0
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["response_format"]["json_schema"]["schema"] == _SCHEMA
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body.get("stream", False) is stream


def test_openai_complete_through_real_sdk(openai_provider: Any) -> None:
    msg, usage = openai_provider.complete(**_openai_kwargs())

    _assert_openai_request(openai_provider.recorder.requests[-1], stream=False)
    assert msg.content == ""
    assert msg.tool_calls[0].parameters == {"city": "Paris"}
    assert (usage.prompt_tokens, usage.completion_tokens) == (20, 5)
    # openai<1.51 doesn't type prompt_tokens_details, so cached tokens read as None there.
    from openai.types.completion_usage import CompletionUsage

    if "prompt_tokens_details" in getattr(CompletionUsage, "model_fields", {}):
        assert usage.cache_read_input_tokens == 8


def test_openai_acomplete_through_real_sdk(openai_provider: Any) -> None:
    msg, _ = asyncio.run(openai_provider.acomplete(**_openai_kwargs()))

    _assert_openai_request(openai_provider.recorder.requests[-1], stream=False)
    assert msg.tool_calls[0].id == "call_1"


@pytest.mark.parametrize("collect", [_collect, _acollect], ids=["stream", "astream"])
def test_openai_streaming_through_real_sdk(
    openai_provider: Any, collect: Callable[[Any], List[Any]]
) -> None:
    method = openai_provider.stream if collect is _collect else openai_provider.astream
    chunks = collect(method(**_openai_kwargs()))

    body = openai_provider.recorder.requests[-1]
    _assert_openai_request(body, stream=True)
    if collect is _acollect:
        assert body["stream_options"] == {"include_usage": True}
    assert [c for c in chunks if isinstance(c, str)] == ["Hi"]
    tool_calls = [c for c in chunks if isinstance(c, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].parameters == {"city": "Paris"}


def test_openai_embeddings_through_real_sdk() -> None:
    from selectools.embeddings.openai import OpenAIEmbeddingProvider

    http = _http_module(openai)
    recorder = _Recorder(
        http,
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
        b"",
    )
    embedder = OpenAIEmbeddingProvider(api_key="test-key", dimensions=2)
    embedder.client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=http.MockTransport(recorder)),
    )

    assert embedder.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert recorder.requests[-1]["dimensions"] == 2
