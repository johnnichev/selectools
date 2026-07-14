"""Provider-layer tests for native structured output (issue #159).

``response_format`` used to be pure prompt injection — never sent to the
provider. Providers that support a native JSON-schema mode now accept a
``response_format`` kwarg (the raw JSON Schema dict) and translate it into
their native request shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from selectools.types import Message, Role

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

MESSAGES = [Message(role=Role.USER, content="hi")]


def _openai_provider() -> Any:
    from selectools.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.default_model = "gpt-5-mini"
    provider.api_key = "test"
    provider._client = MagicMock()
    provider._async_client = MagicMock()
    return provider


def _openai_response(content: str = '{"answer": "ok"}') -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.prompt_tokens_details = None
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestCapabilityFlags:
    def test_openai_supports_native_structured_output(self) -> None:
        from selectools.providers.openai_provider import OpenAIProvider

        assert getattr(OpenAIProvider, "supports_native_structured_output", False) is True
        assert (
            getattr(OpenAIProvider, "supports_native_structured_output_with_tools", False) is True
        )

    def test_azure_inherits_native_support(self) -> None:
        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        assert getattr(AzureOpenAIProvider, "supports_native_structured_output", False) is True

    def test_ollama_does_not_advertise_native_support(self) -> None:
        from selectools.providers.ollama_provider import OllamaProvider

        assert getattr(OllamaProvider, "supports_native_structured_output", False) is False

    def test_anthropic_does_not_advertise_native_support(self) -> None:
        from selectools.providers.anthropic_provider import AnthropicProvider

        assert getattr(AnthropicProvider, "supports_native_structured_output", False) is False

    def test_gemini_supports_native_but_not_with_tools(self) -> None:
        from selectools.providers.gemini_provider import GeminiProvider

        assert getattr(GeminiProvider, "supports_native_structured_output", False) is True
        assert (
            getattr(GeminiProvider, "supports_native_structured_output_with_tools", True) is False
        )


class TestOpenAICompatNativeFormat:
    def test_complete_passes_json_schema_envelope(self) -> None:
        provider = _openai_provider()
        provider._client.chat.completions.create.return_value = _openai_response()

        provider.complete(
            model="gpt-5-mini",
            system_prompt="sys",
            messages=MESSAGES,
            response_format=SCHEMA,
        )

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "structured_response", "schema": SCHEMA},
        }

    def test_complete_without_response_format_sends_none(self) -> None:
        provider = _openai_provider()
        provider._client.chat.completions.create.return_value = _openai_response()

        provider.complete(model="gpt-5-mini", system_prompt="sys", messages=MESSAGES)

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs

    def test_complete_retries_without_response_format_on_rejection(self) -> None:
        """Older models reject response_format; the request retries without it."""
        provider = _openai_provider()
        provider._client.chat.completions.create.side_effect = [
            Exception("Unsupported parameter: 'response_format'"),
            _openai_response(),
        ]

        msg, _usage = provider.complete(
            model="gpt-3.5-turbo",
            system_prompt="sys",
            messages=MESSAGES,
            response_format=SCHEMA,
        )

        assert msg.content == '{"answer": "ok"}'
        second_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "response_format" not in second_kwargs

    @pytest.mark.asyncio
    async def test_acomplete_passes_json_schema_envelope(self) -> None:
        provider = _openai_provider()
        provider._async_client.chat.completions.create = AsyncMock(return_value=_openai_response())

        await provider.acomplete(
            model="gpt-5-mini",
            system_prompt="sys",
            messages=MESSAGES,
            response_format=SCHEMA,
        )

        kwargs = provider._async_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"


class TestGeminiNativeFormat:
    def test_build_config_sets_json_schema(self) -> None:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        config = provider._build_config(
            system_prompt="sys",
            tools=None,
            temperature=0.0,
            max_tokens=100,
            timeout=None,
            response_format=SCHEMA,
        )
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == SCHEMA

    def test_build_config_without_format_unchanged(self) -> None:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        config = provider._build_config(
            system_prompt="sys",
            tools=None,
            temperature=0.0,
            max_tokens=100,
            timeout=None,
        )
        assert config.response_mime_type is None
        assert config.response_json_schema is None
