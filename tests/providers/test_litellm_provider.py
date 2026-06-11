"""
Tests for LiteLLMProvider.

litellm is an optional dependency and is NOT installed in the test
environment. A fake ``litellm`` module is injected via ``sys.modules``
(same pattern as the voyageai/google fakes in test_embedding_providers.py).
The fake returns OpenAI-shaped response objects, which is exactly what
litellm produces (its ModelResponse mirrors the OpenAI wire format).
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import pytest

from selectools.providers.base import ProviderError
from selectools.types import Message, Role, ToolCall

# ======================================================================
# Fake litellm module + OpenAI-shaped response builders
# ======================================================================


def _usage(prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _tool_call(
    name: str = "get_weather",
    arguments: Union[str, Dict[str, Any]] = '{"location": "Brusque"}',
    call_id: str = "call_123",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    content: Optional[str] = "Hello from litellm",
    tool_calls: Optional[List[SimpleNamespace]] = None,
    usage: Optional[SimpleNamespace] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=usage if usage is not None else _usage(),
    )


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=None,
            )
        ]
    )


def _tool_chunk(
    index: int = 0,
    call_id: Optional[str] = "call_abc",
    name: Optional[str] = "get_weather",
    arguments: Optional[str] = None,
) -> SimpleNamespace:
    tc_delta = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc_delta]),
                finish_reason=None,
            )
        ]
    )


def _finish_chunk(reason: str = "tool_calls") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason=reason,
            )
        ]
    )


def _make_fake_litellm(
    response: Optional[SimpleNamespace] = None,
    stream_chunks: Optional[List[SimpleNamespace]] = None,
    cost: Tuple[float, float] = (0.001, 0.002),
    cost_raises: bool = False,
    completion_raises: Optional[Exception] = None,
) -> types.ModuleType:
    """Build a fake ``litellm`` module exposing the API surface the provider uses."""
    mod = types.ModuleType("litellm")
    calls: List[Dict[str, Any]] = []
    mod.calls = calls  # type: ignore[attr-defined]

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if completion_raises is not None:
            raise completion_raises
        if kwargs.get("stream"):
            return iter(stream_chunks or [])
        return response

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if completion_raises is not None:
            raise completion_raises
        if kwargs.get("stream"):

            async def _gen() -> Any:
                for chunk in stream_chunks or []:
                    yield chunk

            return _gen()
        return response

    def cost_per_token(
        model: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> Tuple[float, float]:
        if cost_raises:
            raise ValueError(f"model not in cost map: {model}")
        return cost

    mod.completion = completion  # type: ignore[attr-defined]
    mod.acompletion = acompletion  # type: ignore[attr-defined]
    mod.cost_per_token = cost_per_token  # type: ignore[attr-defined]
    return mod


def _make_provider(fake: types.ModuleType, **kwargs: Any) -> Any:
    from selectools.providers.litellm_provider import LiteLLMProvider

    with patch.dict("sys.modules", {"litellm": fake}):
        return LiteLLMProvider(**kwargs)


_USER_MSG = [Message(role=Role.USER, content="hi")]


# ======================================================================
# Missing dependency
# ======================================================================


class TestMissingDependency:
    def test_import_error_mentions_extras_group(self) -> None:
        from selectools.providers.litellm_provider import LiteLLMProvider

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match=r"selectools\[litellm\]"):
                LiteLLMProvider(model="groq/llama-3.1-70b")

    def test_module_importable_without_litellm(self) -> None:
        # Importing the module (and the providers package) must not require litellm.
        import selectools.providers
        import selectools.providers.litellm_provider  # noqa: F401

        assert hasattr(selectools.providers, "LiteLLMProvider")


# ======================================================================
# complete()
# ======================================================================


class TestComplete:
    def test_complete_round_trip(self) -> None:
        fake = _make_fake_litellm(response=_response(content="42"))
        provider = _make_provider(fake, model="deepseek/deepseek-chat")

        msg, usage = provider.complete(
            model="deepseek/deepseek-chat",
            system_prompt="You are helpful.",
            messages=_USER_MSG,
        )

        assert msg.role == Role.ASSISTANT
        assert msg.content == "42"
        assert msg.tool_calls is None
        call = fake.calls[0]
        assert call["model"] == "deepseek/deepseek-chat"
        assert call["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert call["messages"][1] == {"role": "user", "content": "hi"}
        assert call["max_tokens"] == 1000
        assert usage.prompt_tokens == 10

    def test_empty_model_falls_back_to_default(self) -> None:
        fake = _make_fake_litellm(response=_response())
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        provider.complete(model="", system_prompt="s", messages=_USER_MSG)

        assert fake.calls[0]["model"] == "groq/llama-3.1-70b"

    def test_usage_mapping(self) -> None:
        fake = _make_fake_litellm(
            response=_response(usage=_usage(prompt=100, completion=25)),
            cost=(0.0003, 0.0007),
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        _, usage = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 125
        assert usage.cost_usd == pytest.approx(0.001)
        assert usage.model == "groq/llama-3.1-70b"
        assert usage.provider == "litellm"
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_cost_falls_back_to_zero_for_unknown_models(self) -> None:
        fake = _make_fake_litellm(response=_response(), cost_raises=True)
        provider = _make_provider(fake, model="custom/unknown-model")

        _, usage = provider.complete(
            model="custom/unknown-model", system_prompt="s", messages=_USER_MSG
        )

        assert usage.cost_usd == 0.0

    def test_none_content_becomes_empty_string(self) -> None:
        fake = _make_fake_litellm(response=_response(content=None, tool_calls=[_tool_call()]))
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        msg, _ = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert msg.content == ""

    def test_provider_error_wrapping(self) -> None:
        fake = _make_fake_litellm(completion_raises=RuntimeError("boom"))
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        with pytest.raises(ProviderError, match="LiteLLM completion failed"):
            provider.complete(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)

    def test_api_key_base_and_extra_kwargs_forwarded(self) -> None:
        fake = _make_fake_litellm(response=_response())
        provider = _make_provider(
            fake,
            model="groq/llama-3.1-70b",
            api_key="sk-test",
            api_base="https://example.test/v1",
            drop_params=True,
        )

        provider.complete(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)

        call = fake.calls[0]
        assert call["api_key"] == "sk-test"
        assert call["api_base"] == "https://example.test/v1"
        assert call["drop_params"] is True

    def test_tools_mapped_to_openai_schema(self) -> None:
        from selectools.tools import tool

        @tool()
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"sunny in {location}"

        fake = _make_fake_litellm(response=_response())
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        provider.complete(
            model="groq/llama-3.1-70b",
            system_prompt="s",
            messages=_USER_MSG,
            tools=[get_weather],
        )

        sent = fake.calls[0]["tools"]
        assert sent[0]["type"] == "function"
        assert sent[0]["function"]["name"] == "get_weather"


# ======================================================================
# Reserved litellm_kwargs keys
# ======================================================================


class TestReservedKwargs:
    @pytest.mark.parametrize(
        "key,value",
        [
            ("messages", [{"role": "user", "content": "hi"}]),
            ("stream", True),
            ("tools", []),
            ("temperature", 0.7),
            ("max_tokens", 256),
        ],
    )
    def test_reserved_key_raises_at_construction(self, key: str, value: Any) -> None:
        fake = _make_fake_litellm(response=_response())

        with pytest.raises(ValueError, match=key):
            _make_provider(fake, model="groq/llama-3.1-70b", **{key: value})

    def test_benign_kwarg_still_flows_through(self) -> None:
        fake = _make_fake_litellm(response=_response())
        provider = _make_provider(fake, model="groq/llama-3.1-70b", api_version="x")

        provider.complete(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)

        assert fake.calls[0]["api_version"] == "x"


# ======================================================================
# Tool-call parsing
# ======================================================================


class TestToolCallParsing:
    def test_tool_call_with_json_string_arguments(self) -> None:
        fake = _make_fake_litellm(response=_response(content=None, tool_calls=[_tool_call()]))
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        msg, _ = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert msg.tool_calls is not None
        tc = msg.tool_calls[0]
        assert tc.tool_name == "get_weather"
        assert tc.parameters == {"location": "Brusque"}
        assert tc.id == "call_123"
        assert tc.parse_error is None

    def test_tool_call_with_dict_arguments(self) -> None:
        # Some litellm-routed providers hand back already-parsed dicts.
        fake = _make_fake_litellm(
            response=_response(
                content=None, tool_calls=[_tool_call(arguments={"location": "Brusque"})]
            )
        )
        provider = _make_provider(fake, model="ollama/llama3")

        msg, _ = provider.complete(model="ollama/llama3", system_prompt="s", messages=_USER_MSG)

        assert msg.tool_calls is not None
        assert msg.tool_calls[0].parameters == {"location": "Brusque"}
        assert msg.tool_calls[0].parse_error is None

    def test_malformed_json_arguments_set_parse_error(self) -> None:
        fake = _make_fake_litellm(
            response=_response(content=None, tool_calls=[_tool_call(arguments='{"location": ')])
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        msg, _ = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert msg.tool_calls is not None
        tc = msg.tool_calls[0]
        assert tc.parameters == {}
        assert tc.parse_error is not None
        assert "invalid JSON" in tc.parse_error

    def test_non_str_non_dict_arguments_set_parse_error(self) -> None:
        # Mirrors Ollama's third branch: a list (or any other type) must not
        # raise an uncaught TypeError from json.loads.
        fake = _make_fake_litellm(
            response=_response(
                content=None,
                tool_calls=[_tool_call(arguments=["not", "valid"])],  # type: ignore[arg-type]
            )
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        msg, _ = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert msg.tool_calls is not None
        tc = msg.tool_calls[0]
        assert tc.parameters == {}
        assert tc.parse_error is not None
        assert "unsupported tool arguments type" in tc.parse_error
        assert "list" in tc.parse_error

    def test_missing_tool_call_id_generates_one(self) -> None:
        fake = _make_fake_litellm(
            response=_response(content=None, tool_calls=[_tool_call(call_id=None)])
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        msg, _ = provider.complete(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        )

        assert msg.tool_calls is not None
        assert msg.tool_calls[0].id  # generated, non-empty


# ======================================================================
# stream()
# ======================================================================


class TestStream:
    def test_stream_yields_text_chunks(self) -> None:
        fake = _make_fake_litellm(stream_chunks=[_text_chunk("Hel"), _text_chunk("lo")])
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        chunks = list(
            provider.stream(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)
        )

        assert chunks == ["Hel", "lo"]
        assert fake.calls[0]["stream"] is True

    def test_stream_assembles_tool_calls(self) -> None:
        fake = _make_fake_litellm(
            stream_chunks=[
                _tool_chunk(call_id="call_s1", name="get_weather", arguments='{"loc'),
                _tool_chunk(call_id=None, name=None, arguments='ation": "Brusque"}'),
                _finish_chunk("tool_calls"),
            ]
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        chunks = list(
            provider.stream(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)
        )

        assert len(chunks) == 1
        tc = chunks[0]
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "get_weather"
        assert tc.parameters == {"location": "Brusque"}
        assert tc.id == "call_s1"
        assert tc.parse_error is None

    def test_stream_malformed_tool_arguments(self) -> None:
        fake = _make_fake_litellm(
            stream_chunks=[
                _tool_chunk(call_id="call_s2", name="get_weather", arguments='{"broken'),
                _finish_chunk("tool_calls"),
            ]
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        chunks = list(
            provider.stream(model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG)
        )

        tc = chunks[0]
        assert isinstance(tc, ToolCall)
        assert tc.parameters == {}
        assert tc.parse_error is not None

    def test_stream_forwards_tools(self) -> None:
        from selectools.tools import tool

        @tool()
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"sunny in {location}"

        fake = _make_fake_litellm(stream_chunks=[_text_chunk("ok")])
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        list(
            provider.stream(
                model="groq/llama-3.1-70b",
                system_prompt="s",
                messages=_USER_MSG,
                tools=[get_weather],
            )
        )

        assert fake.calls[0]["tools"][0]["function"]["name"] == "get_weather"


# ======================================================================
# Async paths
# ======================================================================


class TestAsync:
    @pytest.mark.asyncio
    async def test_acomplete_round_trip(self) -> None:
        fake = _make_fake_litellm(response=_response(content="async hi"))
        provider = _make_provider(fake, model="deepseek/deepseek-chat")

        msg, usage = await provider.acomplete(
            model="deepseek/deepseek-chat", system_prompt="s", messages=_USER_MSG
        )

        assert msg.content == "async hi"
        assert usage.provider == "litellm"

    @pytest.mark.asyncio
    async def test_astream_yields_text_and_tool_calls(self) -> None:
        fake = _make_fake_litellm(
            stream_chunks=[
                _text_chunk("thinking "),
                _tool_chunk(call_id="call_a1", name="get_weather", arguments='{"location": "X"}'),
                _finish_chunk("tool_calls"),
            ]
        )
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        chunks = []
        async for chunk in provider.astream(
            model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
        ):
            chunks.append(chunk)

        assert chunks[0] == "thinking "
        tc = chunks[1]
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "get_weather"
        assert tc.parameters == {"location": "X"}

    @pytest.mark.asyncio
    async def test_acomplete_error_wrapping(self) -> None:
        fake = _make_fake_litellm(completion_raises=RuntimeError("boom"))
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        with pytest.raises(ProviderError, match="LiteLLM async completion failed"):
            await provider.acomplete(
                model="groq/llama-3.1-70b", system_prompt="s", messages=_USER_MSG
            )


# ======================================================================
# Provider protocol conformance
# ======================================================================


class TestProtocol:
    def test_provider_attributes(self) -> None:
        fake = _make_fake_litellm(response=_response())
        provider = _make_provider(fake, model="groq/llama-3.1-70b")

        assert provider.name == "litellm"
        assert provider.supports_streaming is True
        assert provider.supports_async is True
        assert provider.default_model == "groq/llama-3.1-70b"

    def test_stability_marker_is_beta(self) -> None:
        from selectools.providers.litellm_provider import LiteLLMProvider

        assert getattr(LiteLLMProvider, "__stability__", None) == "beta"
