"""
Unit tests for AzureOpenAIProvider.

Mocks the openai.AzureOpenAI / AsyncAzureOpenAI clients directly -- never
calls real Azure APIs.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.exceptions import ProviderConfigurationError
from selectools.providers.base import ProviderError
from selectools.tools.base import Tool, ToolParameter
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "https://my-resource.openai.azure.com"
_KEY = "test-azure-key"
_DEPLOYMENT = "gpt-4o"


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


def _make_tool() -> Tool:
    return Tool(
        name="test_tool",
        description="A test tool",
        parameters=[
            ToolParameter(name="x", param_type=int, description="An integer"),
        ],
        function=lambda x: str(x),
    )


def _make_stream_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderInit:
    """Test AzureOpenAIProvider initialisation."""

    def test_init_with_explicit_params(self) -> None:
        with (
            patch("openai.AzureOpenAI") as MockAzure,
            patch("openai.AsyncAzureOpenAI") as MockAsyncAzure,
        ):
            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            provider = AzureOpenAIProvider(
                azure_endpoint=_ENDPOINT,
                api_key=_KEY,
                azure_deployment=_DEPLOYMENT,
            )

            MockAzure.assert_called_once_with(
                azure_endpoint=_ENDPOINT,
                api_version="2024-10-21",
                api_key=_KEY,
            )
            MockAsyncAzure.assert_called_once_with(
                azure_endpoint=_ENDPOINT,
                api_version="2024-10-21",
                api_key=_KEY,
            )
            assert provider.default_model == _DEPLOYMENT
            assert provider.name == "azure-openai"

    def test_init_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", _ENDPOINT)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", _KEY)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deploy")

        with (
            patch("openai.AzureOpenAI"),
            patch("openai.AsyncAzureOpenAI"),
        ):
            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            provider = AzureOpenAIProvider()

            assert provider.default_model == "my-deploy"
            assert provider.api_key == _KEY

    def test_init_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", _KEY)

        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        with pytest.raises(ProviderConfigurationError, match="Azure endpoint"):
            AzureOpenAIProvider()

    def test_init_missing_key_and_no_aad_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", _ENDPOINT)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        with pytest.raises(ProviderConfigurationError, match="API key or Azure AD token"):
            AzureOpenAIProvider()

    def test_init_with_aad_token(self) -> None:
        with (
            patch("openai.AzureOpenAI") as MockAzure,
            patch("openai.AsyncAzureOpenAI") as MockAsyncAzure,
        ):
            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            provider = AzureOpenAIProvider(
                azure_endpoint=_ENDPOINT,
                azure_ad_token="my-aad-token",
                azure_deployment="gpt-4o",
            )

            call_kwargs = MockAzure.call_args.kwargs
            assert call_kwargs["azure_ad_token"] == "my-aad-token"
            assert call_kwargs["api_key"] == "aad"  # placeholder

            async_kwargs = MockAsyncAzure.call_args.kwargs
            assert async_kwargs["azure_ad_token"] == "my-aad-token"

    def test_init_with_aad_token_and_key(self) -> None:
        """When both AAD token and key are given, key is passed through."""
        with (
            patch("openai.AzureOpenAI") as MockAzure,
            patch("openai.AsyncAzureOpenAI"),
        ):
            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            AzureOpenAIProvider(
                azure_endpoint=_ENDPOINT,
                api_key=_KEY,
                azure_ad_token="my-aad-token",
                azure_deployment="gpt-4o",
            )

            call_kwargs = MockAzure.call_args.kwargs
            assert call_kwargs["azure_ad_token"] == "my-aad-token"
            assert call_kwargs["api_key"] == _KEY

    def test_init_custom_api_version(self) -> None:
        with (
            patch("openai.AzureOpenAI") as MockAzure,
            patch("openai.AsyncAzureOpenAI"),
        ):
            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            AzureOpenAIProvider(
                azure_endpoint=_ENDPOINT,
                api_key=_KEY,
                api_version="2025-01-01",
            )

            assert MockAzure.call_args.kwargs["api_version"] == "2025-01-01"

    def test_default_deployment_fallback(self) -> None:
        """When no deployment is specified and no env var, default is gpt-4o."""
        with (
            patch("openai.AzureOpenAI"),
            patch("openai.AsyncAzureOpenAI"),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Ensure the env var is not set
            os.environ.pop("AZURE_OPENAI_DEPLOYMENT", None)

            from selectools.providers.azure_openai_provider import AzureOpenAIProvider

            provider = AzureOpenAIProvider(
                azure_endpoint=_ENDPOINT,
                api_key=_KEY,
            )

            assert provider.default_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Template method override tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderTemplateMethods:
    """Test template method overrides."""

    def _get_provider(self) -> Any:
        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
        provider._client = MagicMock()
        provider._async_client = MagicMock()
        provider.default_model = _DEPLOYMENT
        provider._azure_endpoint = _ENDPOINT
        provider.api_key = _KEY
        return provider

    def test_name(self) -> None:
        provider = self._get_provider()
        assert provider.name == "azure-openai"

    def test_get_provider_name(self) -> None:
        provider = self._get_provider()
        assert provider._get_provider_name() == "azure-openai"

    def test_wrap_error(self) -> None:
        provider = self._get_provider()
        err = provider._wrap_error(Exception("timeout"), "completion")
        assert isinstance(err, ProviderError)
        assert "Azure OpenAI completion failed" in str(err)

    def test_get_token_key_inherited(self) -> None:
        """Token key logic is inherited from OpenAIProvider."""
        provider = self._get_provider()
        assert provider._get_token_key("gpt-4o") == "max_tokens"
        assert provider._get_token_key("gpt-5-turbo") == "max_completion_tokens"
        assert provider._get_token_key("o3-mini") == "max_completion_tokens"

    def test_supports_streaming(self) -> None:
        provider = self._get_provider()
        assert provider.supports_streaming is True

    def test_supports_async(self) -> None:
        provider = self._get_provider()
        assert provider.supports_async is True

    def test_stability_marker(self) -> None:
        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        assert getattr(AzureOpenAIProvider, "__stability__", None) == "beta"


# ---------------------------------------------------------------------------
# Inherited complete/stream tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderComplete:
    """Test that inherited complete() works through the Azure client."""

    def _get_provider(self) -> Any:
        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        provider.default_model = _DEPLOYMENT
        provider._azure_endpoint = _ENDPOINT
        provider.api_key = _KEY
        return provider

    def test_complete_returns_message_and_usage(self) -> None:
        provider = self._get_provider()
        mock_response = _make_openai_response("Azure says hello")
        provider._client.chat.completions.create.return_value = mock_response

        msg, usage = provider.complete(
            model=_DEPLOYMENT,
            system_prompt="You are helpful.",
            messages=[Message(role=Role.USER, content="Hi")],
        )

        assert msg.content == "Azure says hello"
        assert msg.role == Role.ASSISTANT
        assert usage.provider == "azure-openai"
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        provider._client.chat.completions.create.assert_called_once()

    def test_complete_with_tools(self) -> None:
        provider = self._get_provider()
        tc_mock = MagicMock()
        tc_mock.id = "call_123"
        tc_mock.function.name = "test_tool"
        tc_mock.function.arguments = '{"x": 42}'
        mock_response = _make_openai_response(content="", tool_calls=[tc_mock])
        provider._client.chat.completions.create.return_value = mock_response

        tool = _make_tool()
        msg, usage = provider.complete(
            model=_DEPLOYMENT,
            system_prompt="You are helpful.",
            messages=[Message(role=Role.USER, content="Use the tool")],
            tools=[tool],
        )

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "test_tool"
        assert msg.tool_calls[0].parameters == {"x": 42}

    @pytest.mark.asyncio
    async def test_acomplete_returns_message_and_usage(self) -> None:
        provider = self._get_provider()
        mock_response = _make_openai_response("Async Azure response")
        provider._async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        msg, usage = await provider.acomplete(
            model=_DEPLOYMENT,
            system_prompt="You are helpful.",
            messages=[Message(role=Role.USER, content="Hi")],
        )

        assert msg.content == "Async Azure response"
        assert usage.provider == "azure-openai"


class TestAzureOpenAIProviderStream:
    """Test that inherited stream() works through the Azure client."""

    def _get_provider(self) -> Any:
        from selectools.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
        provider._client = MagicMock()
        provider._async_client = AsyncMock()
        provider.default_model = _DEPLOYMENT
        provider._azure_endpoint = _ENDPOINT
        provider.api_key = _KEY
        return provider

    def test_stream_yields_text(self) -> None:
        provider = self._get_provider()

        chunks = [
            _make_stream_chunk(content="Hello "),
            _make_stream_chunk(content="world"),
            _make_stream_chunk(content=None, finish_reason="stop"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model=_DEPLOYMENT,
                system_prompt="Be helpful.",
                messages=[Message(role=Role.USER, content="Hi")],
            )
        )

        assert result == ["Hello ", "world"]

    def test_stream_yields_tool_calls(self) -> None:
        provider = self._get_provider()

        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_abc"
        tc_delta.function = MagicMock()
        tc_delta.function.name = "test_tool"
        tc_delta.function.arguments = '{"x": 1}'

        chunks = [
            _make_stream_chunk(content=None, tool_calls=[tc_delta]),
            _make_stream_chunk(content=None, tool_calls=None, finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create.return_value = iter(chunks)

        result = list(
            provider.stream(
                model=_DEPLOYMENT,
                system_prompt="Be helpful.",
                messages=[Message(role=Role.USER, content="Use tool")],
                tools=[_make_tool()],
            )
        )

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].tool_name == "test_tool"
        assert result[0].parameters == {"x": 1}


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderExports:
    """Test that the provider is properly exported."""

    def test_importable_from_providers_package(self) -> None:
        from selectools.providers import AzureOpenAIProvider

        assert AzureOpenAIProvider.name == "azure-openai"

    def test_in_providers_all(self) -> None:
        import selectools.providers as providers_mod

        assert "AzureOpenAIProvider" in providers_mod.__all__
