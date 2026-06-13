"""
Regression tests for issue #66 — Agent + tool-use returns empty on
gemini-2.5-flash-lite.

Two failure families covered:

1. Model-side: gemini-2.5-flash-lite can return an empty candidate (often
   finish_reason=MALFORMED_FUNCTION_CALL or UNEXPECTED_TOOL_CALL) instead of a
   function call. GeminiProvider previously swallowed this silently, so the
   agent looped to max_iterations with no signal. Now a loud warning is logged
   whenever a tool-equipped response contains neither text nor tool calls.

2. Framework-side (BUG-40): selectools emitted two schema shapes that the
   Gemini API rejects with hard 400s (verified live):
   - bare ``list`` params -> ``{"type": "array"}`` without ``items``
   - ``Dict[K, V]`` params -> ``additionalProperties`` (unsupported by Gemini)
   ``_sanitize_schema_for_gemini`` now injects permissive ``items`` and strips
   ``additionalProperties``.

All tests use mocked genai responses — no API key required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selectools.tools.base import Tool, ToolParameter
from selectools.types import Message, Role

pytest.importorskip("google.genai")

from selectools.providers.gemini_provider import (  # noqa: E402
    GeminiProvider,
    _sanitize_schema_for_gemini,
)


def _make_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters=[ToolParameter(name="city", param_type=str, description="City name")],
        function=lambda city: f"sunny in {city}",
    )


def _make_provider() -> GeminiProvider:
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.default_model = "gemini-2.5-flash-lite"
    provider._client = MagicMock()
    return provider


def _empty_response(finish_reason_name: str = "MALFORMED_FUNCTION_CALL") -> MagicMock:
    """Mock a genai response with an empty candidate (no text, no parts)."""
    response = MagicMock()
    response.text = None
    candidate = MagicMock()
    candidate.content = None
    candidate.finish_reason = MagicMock()
    candidate.finish_reason.name = finish_reason_name
    response.candidates = [candidate]
    response.usage_metadata = None
    return response


def _text_response(text: str = "hello") -> MagicMock:
    response = MagicMock()
    response.text = text
    response.candidates = []
    response.usage_metadata = None
    return response


class TestCompleteEmptyToolResponseWarning:
    """complete()/acomplete() must warn when tools yield neither text nor calls."""

    def test_complete_empty_with_tools_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        provider._client.models.generate_content.return_value = _empty_response()

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            with patch(
                "selectools.providers.gemini_provider.calculate_cost_with_cached_input",
                return_value=0.0,
            ):
                msg, _ = provider.complete(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="weather in Paris?")],
                    tools=[_make_tool()],
                )

        assert (msg.content or "") == ""
        assert msg.tool_calls is None
        warning_text = " ".join(r.message for r in caplog.records)
        assert "MALFORMED_FUNCTION_CALL" in warning_text
        assert "gemini-2.5-flash-lite" in warning_text

    def test_complete_empty_without_tools_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _make_provider()
        provider._client.models.generate_content.return_value = _empty_response()

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            with patch(
                "selectools.providers.gemini_provider.calculate_cost_with_cached_input",
                return_value=0.0,
            ):
                provider.complete(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                )

        assert not caplog.records

    def test_complete_text_response_with_tools_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _make_provider()
        provider._client.models.generate_content.return_value = _text_response("the answer")

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            with patch(
                "selectools.providers.gemini_provider.calculate_cost_with_cached_input",
                return_value=0.0,
            ):
                msg, _ = provider.complete(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                    tools=[_make_tool()],
                )

        assert (msg.content or "") == "the answer"
        assert not caplog.records

    def test_complete_tool_call_response_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        response = MagicMock()
        response.text = None
        fc = MagicMock()
        fc.name = "get_weather"
        fc.args = {"city": "Paris"}
        part = MagicMock()
        part.function_call = fc
        part.thought_signature = None
        candidate = MagicMock()
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = MagicMock()
        candidate.finish_reason.name = "STOP"
        response.candidates = [candidate]
        response.usage_metadata = None
        provider._client.models.generate_content.return_value = response

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            with patch(
                "selectools.providers.gemini_provider.calculate_cost_with_cached_input",
                return_value=0.0,
            ):
                msg, _ = provider.complete(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="weather in Paris?")],
                    tools=[_make_tool()],
                )

        assert msg.tool_calls is not None
        assert msg.tool_calls[0].tool_name == "get_weather"
        assert not caplog.records

    def test_acomplete_empty_with_tools_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _make_provider()
        provider._client.aio.models.generate_content = AsyncMock(return_value=_empty_response())

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            with patch(
                "selectools.providers.gemini_provider.calculate_cost_with_cached_input",
                return_value=0.0,
            ):
                msg, _ = asyncio.run(
                    provider.acomplete(
                        model="gemini-2.5-flash-lite",
                        system_prompt="sys",
                        messages=[Message(role=Role.USER, content="weather in Paris?")],
                        tools=[_make_tool()],
                    )
                )

        assert (msg.content or "") == ""
        assert msg.tool_calls is None
        warning_text = " ".join(r.message for r in caplog.records)
        assert "MALFORMED_FUNCTION_CALL" in warning_text


class TestStreamEmptyToolResponseWarning:
    """stream()/astream() must warn when a tool-equipped stream yields nothing."""

    def _empty_chunk(self, finish_reason_name: str = "UNEXPECTED_TOOL_CALL") -> MagicMock:
        chunk = MagicMock()
        chunk.text = None
        candidate = MagicMock()
        candidate.content = None
        candidate.finish_reason = MagicMock()
        candidate.finish_reason.name = finish_reason_name
        chunk.candidates = [candidate]
        return chunk

    def test_stream_empty_with_tools_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        provider._client.models.generate_content_stream.return_value = iter([self._empty_chunk()])

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            chunks = list(
                provider.stream(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="weather in Paris?")],
                    tools=[_make_tool()],
                )
            )

        assert chunks == []
        warning_text = " ".join(r.message for r in caplog.records)
        assert "UNEXPECTED_TOOL_CALL" in warning_text

    def test_stream_with_text_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        chunk = MagicMock()
        chunk.text = "hi"
        chunk.candidates = []
        provider._client.models.generate_content_stream.return_value = iter([chunk])

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            chunks = list(
                provider.stream(
                    model="gemini-2.5-flash-lite",
                    system_prompt="sys",
                    messages=[Message(role=Role.USER, content="hi")],
                    tools=[_make_tool()],
                )
            )

        assert chunks == ["hi"]
        assert not caplog.records

    def test_astream_empty_with_tools_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _make_provider()
        empty_chunk = self._empty_chunk("MALFORMED_FUNCTION_CALL")

        async def _aiter() -> Any:
            yield empty_chunk

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter())

        async def _consume() -> list:
            collected = []
            async for item in provider.astream(
                model="gemini-2.5-flash-lite",
                system_prompt="sys",
                messages=[Message(role=Role.USER, content="weather in Paris?")],
                tools=[_make_tool()],
            ):
                collected.append(item)
            return collected

        with caplog.at_level(logging.WARNING, logger="selectools.providers.gemini_provider"):
            chunks = asyncio.run(_consume())

        assert chunks == []
        warning_text = " ".join(r.message for r in caplog.records)
        assert "MALFORMED_FUNCTION_CALL" in warning_text


class TestSchemaSanitization:
    """BUG-40: schema shapes the Gemini API hard-rejects must be sanitized."""

    def test_bare_array_gets_items(self) -> None:
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "description": "things"}},
            "required": ["items"],
        }
        cleaned = _sanitize_schema_for_gemini(schema)
        assert cleaned["properties"]["items"]["items"] == {"type": "string"}

    def test_typed_array_items_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "nums": {"type": "array", "description": "n", "items": {"type": "integer"}}
            },
            "required": ["nums"],
        }
        cleaned = _sanitize_schema_for_gemini(schema)
        assert cleaned["properties"]["nums"]["items"] == {"type": "integer"}

    def test_additional_properties_stripped(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "kv",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["payload"],
        }
        cleaned = _sanitize_schema_for_gemini(schema)
        assert "additionalProperties" not in cleaned["properties"]["payload"]
        assert cleaned["properties"]["payload"]["type"] == "object"

    def test_nested_array_inside_items_sanitized(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "matrix": {"type": "array", "items": {"type": "array"}},
            },
            "required": ["matrix"],
        }
        cleaned = _sanitize_schema_for_gemini(schema)
        assert cleaned["properties"]["matrix"]["items"]["items"] == {"type": "string"}

    def test_input_schema_not_mutated(self) -> None:
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "description": "things"}},
            "required": ["items"],
        }
        _sanitize_schema_for_gemini(schema)
        assert "items" not in schema["properties"]["items"]

    def test_map_tool_to_gemini_sanitizes_bare_list(self) -> None:
        bare_list_tool = Tool(
            name="add_items",
            description="Add items to a list",
            parameters=[ToolParameter(name="items", param_type=list, description="things")],
            function=lambda items: "ok",
        )
        provider = _make_provider()
        gemini_tool = provider._map_tool_to_gemini(bare_list_tool)
        decl = gemini_tool.function_declarations[0]
        params = decl.parameters
        # types.Schema object — items must be present for the array property
        items_prop = params.properties["items"]
        assert items_prop.items is not None

    def test_map_tool_to_gemini_strips_additional_properties(self) -> None:
        dict_tool = Tool(
            name="send_payload",
            description="Send a payload",
            parameters=[
                ToolParameter(
                    name="payload",
                    param_type=dict,
                    description="kv metadata",
                    element_type=str,
                )
            ],
            function=lambda payload: "ok",
        )
        # The raw selectools schema emits additionalProperties for Dict[K, V]
        assert "additionalProperties" in dict_tool.schema()["parameters"]["properties"]["payload"]
        provider = _make_provider()
        gemini_tool = provider._map_tool_to_gemini(dict_tool)
        decl = gemini_tool.function_declarations[0]
        payload_prop = decl.parameters.properties["payload"]
        assert getattr(payload_prop, "additional_properties", None) is None
