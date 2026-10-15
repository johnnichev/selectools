"""temperature=None means "send no sampling temperature".

Reasoning models reject ``temperature`` outright. Before this, every provider
sent the agent's 0.0 default and relied on a 400 round trip (or nothing) to
drop it. ``None`` now omits the field; ``0.0`` is still sent as a real value.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.agent.core import Agent, AgentConfig
from selectools.cache import CacheKeyBuilder
from selectools.providers.base import Provider
from selectools.providers.openai_provider import OpenAIProvider
from selectools.tools import tool
from selectools.types import Message, Role
from selectools.usage import UsageStats

_USAGE = UsageStats(0, 0, 0, 0.0, "mock", "mock")


@tool()
def _dummy() -> str:
    """No-op."""
    return "ok"


class _KwargsProvider(Provider):
    name = "kwargs"
    supports_streaming = True
    supports_async = True

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.calls.append(kwargs)
        return Message(role=Role.ASSISTANT, content="done"), _USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)

    def stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield "done"

    async def astream(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield "done"


def _openai() -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
    response.usage = None
    provider._client.chat.completions.create.return_value = response
    return provider


def _call_kwargs(provider: OpenAIProvider) -> Dict[str, Any]:
    return provider._client.chat.completions.create.call_args.kwargs


# --- agent boundary ---------------------------------------------------------


def test_agent_config_accepts_none_temperature() -> None:
    assert AgentConfig(temperature=None).temperature is None


def test_agent_forwards_none_temperature_on_complete_and_stream() -> None:
    provider = _KwargsProvider()
    Agent([_dummy], provider=provider, config=AgentConfig(model="m", temperature=None)).run("hi")
    Agent(
        [_dummy],
        provider=provider,
        config=AgentConfig(model="m", temperature=None, stream=True),
    ).run("hi")

    assert [c["temperature"] for c in provider.calls] == [None, None]


def test_cache_key_distinguishes_none_from_zero() -> None:
    msgs = [Message(role=Role.USER, content="q")]
    assert CacheKeyBuilder.build(
        model="m", system_prompt="s", messages=msgs, temperature=None
    ) != CacheKeyBuilder.build(model="m", system_prompt="s", messages=msgs, temperature=0.0)


# --- OpenAI-compatible providers ----------------------------------------------


def test_openai_complete_omits_temperature_when_none() -> None:
    provider = _openai()

    provider.complete(model="gpt-5-mini", system_prompt="s", messages=[], temperature=None)

    assert "temperature" not in _call_kwargs(provider)


def test_openai_complete_still_sends_zero_temperature() -> None:
    provider = _openai()

    provider.complete(model="gpt-5-mini", system_prompt="s", messages=[], temperature=0.0)

    assert _call_kwargs(provider)["temperature"] == 0.0


def test_openai_stream_omits_temperature_when_none() -> None:
    provider = _openai()
    provider._client.chat.completions.create.return_value = iter([])

    list(provider.stream(model="gpt-5-mini", system_prompt="s", messages=[], temperature=None))

    assert "temperature" not in _call_kwargs(provider)


@pytest.mark.asyncio
async def test_openai_async_paths_omit_temperature_when_none() -> None:
    provider = _openai()
    seen: List[Dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        if kwargs.get("stream"):

            async def empty() -> AsyncIterator[Any]:
                if False:  # pragma: no cover
                    yield None

            return empty()
        return provider._client.chat.completions.create.return_value

    provider._async_client.chat.completions.create = create

    await provider.acomplete(model="gpt-5-mini", system_prompt="s", messages=[], temperature=None)
    async for _ in provider.astream(
        model="gpt-5-mini", system_prompt="s", messages=[], temperature=None
    ):
        pass

    assert len(seen) == 2
    assert all("temperature" not in kwargs for kwargs in seen)


# --- Anthropic / Gemini -------------------------------------------------------


def test_anthropic_request_omits_temperature_when_none() -> None:
    from selectools.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.cache_system = False
    provider.cache_tools = False

    def args(temperature: Any) -> Dict[str, Any]:
        return provider._build_request_args(
            model_name="claude-sonnet-4-5",
            system_prompt="s",
            payload=[],
            temperature=temperature,
            max_tokens=5,
            tools=None,
            timeout=None,
        )

    assert "temperature" not in args(None)
    assert args(0.0)["temperature"] == 0.0


def test_gemini_config_leaves_temperature_unset_when_none() -> None:
    pytest.importorskip("google.genai")
    from selectools.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)

    config = provider._build_config(
        system_prompt="s", tools=None, temperature=None, max_tokens=5, timeout=None
    )

    assert config.temperature is None
