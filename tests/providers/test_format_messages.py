"""
Unit tests for _format_messages() in OpenAI, Anthropic, and Gemini providers.

Verifies correct handling of TOOL role (with tool_call_id), ASSISTANT role
(with tool_calls array), and USER role (with images).

These bugs were missed because no test ever inspected the actual payload
sent to the provider API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from selectools.types import Message, Role, ToolCall


class TestOpenAIFormatMessages:
    def _get_provider(self) -> Any:
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        provider.api_key = "test"
        return provider

    def test_tool_role_formatted(self) -> None:
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="Result: 42",
                tool_call_id="call_abc",
            )
        ]
        formatted = provider._format_messages("system prompt", messages)

        assert formatted[0]["role"] == "system"
        tool_msg = formatted[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_abc"
        assert tool_msg["content"] == "Result: 42"

    def test_assistant_with_tool_calls(self) -> None:
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="calculator",
                        parameters={"expr": "2+2"},
                        id="call_xyz",
                    )
                ],
            )
        ]
        formatted = provider._format_messages("system", messages)
        assistant_msg = formatted[1]

        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "call_xyz"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "calculator"
        args = json.loads(tc["function"]["arguments"])
        assert args["expr"] == "2+2"

    def test_assistant_without_tool_calls(self) -> None:
        provider = self._get_provider()
        messages = [Message(role=Role.ASSISTANT, content="Hello!")]
        formatted = provider._format_messages("system", messages)
        assert "tool_calls" not in formatted[1]

    def test_user_with_image(self) -> None:
        provider = self._get_provider()
        msg = Message(role=Role.USER, content="What is this?")
        msg.image_base64 = "abc123"
        formatted = provider._format_messages("system", [msg])
        content = formatted[1]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_system_prompt_first(self) -> None:
        provider = self._get_provider()
        messages = [Message(role=Role.USER, content="hi")]
        formatted = provider._format_messages("You are helpful.", messages)
        assert formatted[0] == {"role": "system", "content": "You are helpful."}


class TestAnthropicFormatMessages:
    def _get_provider(self) -> Any:
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.default_model = "claude-test"
        return provider

    def test_tool_role_as_user_tool_result(self) -> None:
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="Result: 42",
                tool_call_id="toolu_abc",
            )
        ]
        formatted = provider._format_messages(messages)

        msg = formatted[0]
        assert msg["role"] == "user"
        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_abc"
        assert block["content"] == "Result: 42"

    def test_assistant_with_tool_use(self) -> None:
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Let me calculate.",
                tool_calls=[
                    ToolCall(
                        tool_name="calculator",
                        parameters={"expr": "2+2"},
                        id="toolu_xyz",
                    )
                ],
            )
        ]
        formatted = provider._format_messages(messages)
        msg = formatted[0]

        assert msg["role"] == "assistant"
        text_block = msg["content"][0]
        assert text_block["type"] == "text"
        tool_block = msg["content"][1]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "calculator"
        assert tool_block["id"] == "toolu_xyz"
        assert tool_block["input"] == {"expr": "2+2"}

    def test_user_with_image(self) -> None:
        provider = self._get_provider()
        msg = Message(role=Role.USER, content="What is this?")
        msg.image_base64 = "abc123"
        formatted = provider._format_messages([msg])
        content = formatted[0]["content"]
        types = [b["type"] for b in content]
        assert "image" in types
        assert "text" in types

    def test_tool_missing_id_uses_unknown(self) -> None:
        provider = self._get_provider()
        messages = [Message(role=Role.TOOL, content="Result", tool_call_id=None)]
        formatted = provider._format_messages(messages)
        assert formatted[0]["content"][0]["tool_use_id"] == "unknown"


class TestGeminiFormatContents:
    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-test"
        return provider

    def test_tool_role_as_function_response(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="42",
                tool_name="calculator",
                tool_call_id="call_1",
            )
        ]
        formatted = provider._format_contents("system", messages)

        assert len(formatted) == 1
        content = formatted[0]
        assert content.role == "user"
        part = content.parts[0]
        assert part.function_response is not None
        assert part.function_response.name == "calculator"

    def test_assistant_with_function_call(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search",
                        parameters={"q": "test"},
                        id="call_1",
                    )
                ],
            )
        ]
        formatted = provider._format_contents("system", messages)

        content = formatted[0]
        assert content.role == "model"
        fc_parts = [p for p in content.parts if p.function_call is not None]
        assert len(fc_parts) == 1
        assert fc_parts[0].function_call.name == "search"

    def test_user_with_image(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        msg = Message(role=Role.USER, content="What?")
        msg.image_base64 = "dGVzdA=="  # valid base64 for "test"
        formatted = provider._format_contents("system", [msg])
        parts = formatted[0].parts
        assert len(parts) == 2

    def test_tool_without_name_fallback(self) -> None:
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        provider = self._get_provider()
        messages = [Message(role=Role.TOOL, content="result", tool_name=None)]
        formatted = provider._format_contents("system", messages)
        part = formatted[0].parts[0]
        assert part.text is not None
        assert "Tool output" in part.text


class TestOpenAIFormatMessagesNoneContent:
    """Regression: TOOL messages with None content must be formatted as empty string.

    OpenAI's API rejects None as message content; we must coerce to "".
    """

    def _get_provider(self) -> Any:
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        provider.api_key = "test"
        return provider

    def test_tool_message_none_content_coerced_to_empty_string(self) -> None:
        """TOOL messages with content=None must have content='' in the payload."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content=None,  # type: ignore[arg-type]
                tool_call_id="call_1",
            )
        ]
        formatted = provider._format_messages("sys", messages)
        tool_msg = formatted[1]
        assert tool_msg["content"] == "", f"Expected '', got {tool_msg['content']!r}"
        assert tool_msg["content"] is not None, "content must not be None"

    def test_tool_message_with_content_preserved(self) -> None:
        """TOOL messages with real content must not be affected."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="the result",
                tool_call_id="call_2",
            )
        ]
        formatted = provider._format_messages("sys", messages)
        assert formatted[1]["content"] == "the result"


class TestOpenAIParseResponseEmptyChoices:
    """Regression: _parse_response() must raise ProviderError when choices is empty.

    OpenAI returns an empty choices list when a request is blocked by content
    filtering.  Previously this raised an IndexError (unhandled), now it must
    raise ProviderError with a descriptive message.
    """

    def _get_provider(self) -> Any:
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.default_model = "gpt-4o"
        provider.api_key = "test"
        return provider

    def test_empty_choices_raises_provider_error(self) -> None:

        from selectools.providers.base import ProviderError

        provider = self._get_provider()

        # Simulate an OpenAI response with no choices (content-filtered)
        mock_response = MagicMock()
        mock_response.choices = []

        with pytest.raises(ProviderError, match="empty choices"):
            provider._parse_response(mock_response, "gpt-4o")

    def test_single_choice_still_works(self) -> None:
        """Regression guard: normal single-choice response must still parse correctly."""

        provider = self._get_provider()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_choice.message.tool_calls = None

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        msg, stats = provider._parse_response(mock_response, "gpt-4o")
        assert msg.content == "Hello"
        assert stats.prompt_tokens == 10


# ---------------------------------------------------------------------------
# Regression: GeminiProvider silently ignored the timeout parameter
#
# All four methods (complete, stream, acomplete, astream) accepted a
# `timeout` argument but never applied it to the GenerateContentConfig.
# Users who set a timeout got no protection — the Gemini SDK simply used
# its default (no timeout).  The fix passes
# `http_options=types.HttpOptions(timeout=int(timeout * 1000))` into the
# config when timeout is not None.
# ---------------------------------------------------------------------------


class TestGeminiTimeout:
    """Regression: timeout parameter must be forwarded to GenerateContentConfig."""

    def _get_provider(self) -> Any:
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-test"
        return provider

    def _make_mock_response(self) -> Any:

        mock_response = MagicMock()
        mock_response.text = "hello"
        mock_response.candidates = []
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 3
        mock_response.usage_metadata = mock_usage
        return mock_response

    def test_complete_passes_timeout_to_config(self) -> None:
        """complete() must set http_options with timeout_ms when timeout is given."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        from unittest.mock import patch

        provider = self._get_provider()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._make_mock_response()
        provider._client = mock_client

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="gemini-test",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=5.0,
            )

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.http_options is not None, "http_options must be set when timeout is given"
        assert (
            config.http_options.timeout == 5000
        ), f"Expected timeout_ms=5000, got {config.http_options.timeout}"

    def test_complete_no_timeout_does_not_set_http_options(self) -> None:
        """complete() must not set http_options when timeout=None."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        from unittest.mock import patch

        provider = self._get_provider()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._make_mock_response()
        provider._client = mock_client

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="gemini-test",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=None,
            )

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert (
            config.http_options is None
        ), "http_options must be None when timeout is not specified"

    def test_stream_passes_timeout_to_config(self) -> None:
        """stream() must set http_options with timeout_ms when timeout is given."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        from unittest.mock import patch

        provider = self._get_provider()
        mock_client = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.text = "hello"
        mock_chunk.candidates = []
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])
        provider._client = mock_client

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            list(
                provider.stream(
                    model="gemini-test",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                    timeout=2.5,
                )
            )

        call_kwargs = mock_client.models.generate_content_stream.call_args[1]
        config = call_kwargs["config"]
        assert config.http_options is not None
        assert (
            config.http_options.timeout == 2500
        ), f"Expected timeout_ms=2500, got {config.http_options.timeout}"

    def test_timeout_seconds_converted_to_milliseconds(self) -> None:
        """Timeout in seconds must be converted to milliseconds (SDK uses ms)."""
        try:
            from google.genai import types
        except ImportError:
            pytest.skip("google-genai not installed")

        from unittest.mock import patch

        provider = self._get_provider()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._make_mock_response()
        provider._client = mock_client

        with patch("selectools.providers.gemini_provider.calculate_cost", return_value=0.0):
            provider.complete(
                model="gemini-test",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="hi")],
                timeout=30.0,
            )

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert (
            config.http_options.timeout == 30000
        ), f"Expected 30000ms (30s), got {config.http_options.timeout}"
