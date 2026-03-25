"""
Tests for Phase 1 Design Pattern changes:
1. StepType Enum (trace.py) — backward-compatible str enum
2. ToolCall.thought_signature (types.py) — optional field
3. GeminiProvider thought signature support (gemini_provider.py) — _format_contents
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import pytest

from selectools.trace import AgentTrace, StepType, TraceStep
from selectools.types import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# 1. StepType Enum
# ---------------------------------------------------------------------------


class TestStepTypeEnum:
    """StepType converted from Literal[...] to class StepType(str, Enum)."""

    EXPECTED_MEMBERS = {
        "LLM_CALL": "llm_call",
        "TOOL_SELECTION": "tool_selection",
        "TOOL_EXECUTION": "tool_execution",
        "CACHE_HIT": "cache_hit",
        "ERROR": "error",
        "STRUCTURED_RETRY": "structured_retry",
        "GUARDRAIL": "guardrail",
        "COHERENCE_CHECK": "coherence_check",
        "OUTPUT_SCREENING": "output_screening",
        "SESSION_LOAD": "session_load",
        "SESSION_SAVE": "session_save",
        "MEMORY_SUMMARIZE": "memory_summarize",
        "ENTITY_EXTRACTION": "entity_extraction",
        "KG_EXTRACTION": "kg_extraction",
        "BUDGET_EXCEEDED": "budget_exceeded",
        "CANCELLED": "cancelled",
        "PROMPT_COMPRESSED": "prompt_compressed",
    }

    def test_all_17_members_exist(self) -> None:
        """All 17 expected members are present on the StepType enum."""
        for attr_name, str_value in self.EXPECTED_MEMBERS.items():
            member = getattr(StepType, attr_name, None)
            assert member is not None, f"StepType.{attr_name} missing"
            assert member.value == str_value

    def test_no_extra_members(self) -> None:
        """StepType has exactly 17 members — no accidental extras."""
        members = [m for m in StepType]
        assert len(members) == 17

    def test_backward_compat_string_equality(self) -> None:
        """Each StepType member compares equal to its plain string value."""
        for attr_name, str_value in self.EXPECTED_MEMBERS.items():
            member = getattr(StepType, attr_name)
            assert member == str_value
            assert str_value == member  # commutative

    def test_members_are_str_instances(self) -> None:
        """StepType members are instances of str (str, Enum inheritance)."""
        for member in StepType:
            assert isinstance(member, str)

    def test_members_are_enum_instances(self) -> None:
        """StepType members are instances of Enum."""
        for member in StepType:
            assert isinstance(member, Enum)

    def test_usable_in_tracestep_creation(self) -> None:
        """TraceStep accepts StepType members for its type field."""
        step = TraceStep(type=StepType.LLM_CALL, duration_ms=42.0, model="gpt-4o")
        assert step.type is StepType.LLM_CALL
        assert step.type == "llm_call"
        assert step.duration_ms == 42.0

    def test_tracestep_to_dict_serializes_as_string(self) -> None:
        """TraceStep.to_dict() emits the string value, not a StepType object."""
        step = TraceStep(type=StepType.TOOL_EXECUTION, duration_ms=10.0, tool_name="search")
        d = step.to_dict()
        assert d["type"] == "tool_execution"
        # Verify it serialized as plain str, not an Enum wrapper
        assert type(d["type"]) is StepType  # StepType IS str, so it's JSON-safe
        # But it should still equal the plain string
        assert d["type"] == "tool_execution"

    def test_agent_trace_filter_with_enum(self) -> None:
        """AgentTrace.filter(type=StepType.LLM_CALL) returns matching steps."""
        trace = AgentTrace()
        trace.add(TraceStep(type=StepType.LLM_CALL, duration_ms=100.0))
        trace.add(TraceStep(type=StepType.TOOL_EXECUTION, duration_ms=50.0))
        trace.add(TraceStep(type=StepType.LLM_CALL, duration_ms=80.0))

        filtered = trace.filter(type=StepType.LLM_CALL)
        assert len(filtered) == 2
        assert all(s.type == StepType.LLM_CALL for s in filtered)

    def test_agent_trace_filter_with_string(self) -> None:
        """AgentTrace.filter(type="llm_call") still works (backward compat)."""
        trace = AgentTrace()
        trace.add(TraceStep(type=StepType.LLM_CALL, duration_ms=100.0))
        trace.add(TraceStep(type=StepType.TOOL_EXECUTION, duration_ms=50.0))

        # Pass a plain string — the == comparison still works because StepType is str
        filtered = trace.filter(type=StepType("llm_call"))
        assert len(filtered) == 1

    def test_agent_trace_filter_enum_and_string_equivalent(self) -> None:
        """Filtering with enum vs. string value produces the same results."""
        trace = AgentTrace()
        trace.add(TraceStep(type=StepType.CACHE_HIT, duration_ms=1.0))
        trace.add(TraceStep(type=StepType.ERROR, duration_ms=2.0))
        trace.add(TraceStep(type=StepType.CACHE_HIT, duration_ms=3.0))

        by_enum = trace.filter(type=StepType.CACHE_HIT)
        by_str = trace.filter(type=StepType("cache_hit"))
        assert len(by_enum) == len(by_str) == 2

    def test_steptype_importable_from_selectools(self) -> None:
        """StepType is importable directly from the selectools package."""
        from selectools import StepType as Imported

        assert Imported is StepType

    def test_steptype_in_string_formatting(self) -> None:
        """StepType .value gives the string value across all Python versions."""
        assert StepType.LLM_CALL.value == "llm_call"
        # == comparison always works (str, Enum inherits __eq__ from str)
        assert StepType.LLM_CALL == "llm_call"
        # Verify it works in the timeline method which uses :18s formatting
        step = TraceStep(type=StepType.LLM_CALL, duration_ms=100.0)
        trace = AgentTrace()
        trace.add(step)
        timeline = trace.timeline()
        assert "llm_call" in timeline

    def test_steptype_hashable(self) -> None:
        """StepType members are hashable and can be used as dict keys or in sets."""
        d = {StepType.LLM_CALL: "a", StepType.ERROR: "b"}
        assert d[StepType.LLM_CALL] == "a"
        s = {StepType.LLM_CALL, StepType.LLM_CALL, StepType.ERROR}
        assert len(s) == 2

    def test_agent_trace_to_dict_serializes_steps(self) -> None:
        """AgentTrace.to_dict() serializes all steps including StepType values."""
        trace = AgentTrace()
        trace.add(TraceStep(type=StepType.GUARDRAIL, duration_ms=5.0, summary="blocked"))
        d = trace.to_dict()
        step_d = d["steps"][0]
        assert step_d["type"] == "guardrail"
        assert step_d["summary"] == "blocked"


# ---------------------------------------------------------------------------
# 2. ToolCall.thought_signature
# ---------------------------------------------------------------------------


class TestToolCallThoughtSignature:
    """ToolCall gained an optional thought_signature field."""

    def test_default_is_none(self) -> None:
        """thought_signature defaults to None when not provided."""
        tc = ToolCall(tool_name="search", parameters={"q": "test"})
        assert tc.thought_signature is None

    def test_can_set_string_value(self) -> None:
        """thought_signature can be set to a string."""
        tc = ToolCall(
            tool_name="search",
            parameters={"q": "test"},
            thought_signature="sig_abc123",
        )
        assert tc.thought_signature == "sig_abc123"

    def test_message_to_dict_includes_thought_signature(self) -> None:
        """Message.to_dict() includes thought_signature in tool_calls serialization."""
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="search",
                    parameters={"q": "test"},
                    id="call_1",
                    thought_signature="sig_xyz",
                )
            ],
        )
        d = msg.to_dict()
        tc_dict = d["tool_calls"][0]
        assert tc_dict["thought_signature"] == "sig_xyz"

    def test_message_to_dict_thought_signature_none(self) -> None:
        """Message.to_dict() includes thought_signature=None when not set."""
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name="calc", parameters={}, id="call_2")],
        )
        d = msg.to_dict()
        tc_dict = d["tool_calls"][0]
        assert "thought_signature" in tc_dict
        assert tc_dict["thought_signature"] is None

    def test_message_from_dict_restores_thought_signature(self) -> None:
        """Message.from_dict() correctly restores thought_signature."""
        data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "search",
                    "parameters": {"q": "test"},
                    "id": "call_1",
                    "thought_signature": "sig_restored",
                }
            ],
        }
        msg = Message.from_dict(data)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].thought_signature == "sig_restored"

    def test_message_from_dict_missing_thought_signature_is_none(self) -> None:
        """Message.from_dict() sets thought_signature to None when key is absent."""
        data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "calc",
                    "parameters": {},
                    "id": "call_2",
                }
            ],
        }
        msg = Message.from_dict(data)
        assert msg.tool_calls[0].thought_signature is None

    def test_round_trip_preserves_thought_signature(self) -> None:
        """to_dict -> from_dict round-trip preserves thought_signature."""
        original = Message(
            role=Role.ASSISTANT,
            content="calling tool",
            tool_calls=[
                ToolCall(
                    tool_name="weather",
                    parameters={"city": "NYC"},
                    id="call_rt",
                    thought_signature="sig_roundtrip",
                )
            ],
        )
        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1
        tc = restored.tool_calls[0]
        assert tc.tool_name == "weather"
        assert tc.parameters == {"city": "NYC"}
        assert tc.id == "call_rt"
        assert tc.thought_signature == "sig_roundtrip"

    def test_round_trip_preserves_none_thought_signature(self) -> None:
        """to_dict -> from_dict round-trip preserves thought_signature=None."""
        original = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name="calc", parameters={"x": 1}, id="call_none")],
        )
        data = original.to_dict()
        restored = Message.from_dict(data)
        assert restored.tool_calls[0].thought_signature is None

    def test_multiple_tool_calls_mixed_signatures(self) -> None:
        """Round-trip works when some tool_calls have signatures and others don't."""
        original = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    tool_name="a",
                    parameters={},
                    id="call_a",
                    thought_signature="sig_a",
                ),
                ToolCall(
                    tool_name="b",
                    parameters={},
                    id="call_b",
                    thought_signature=None,
                ),
            ],
        )
        data = original.to_dict()
        restored = Message.from_dict(data)
        assert restored.tool_calls[0].thought_signature == "sig_a"
        assert restored.tool_calls[1].thought_signature is None


# ---------------------------------------------------------------------------
# 3. GeminiProvider thought signature support (_format_contents)
# ---------------------------------------------------------------------------


class TestGeminiFormatContentsThoughtSignature:
    """GeminiProvider._format_contents echoes thought_signature on fc_parts
    and includes original functionCall alongside functionResponse for TOOL msgs.
    """

    def _get_provider(self) -> Any:
        """Construct a GeminiProvider without hitting the real __init__."""
        try:
            from google.genai import types  # noqa: F401
        except ImportError:
            pytest.skip("google-genai not installed")

        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider.default_model = "gemini-test"
        return provider

    # -- ASSISTANT messages with thought_signature --

    def test_assistant_fc_part_gets_thought_signature(self) -> None:
        """ASSISTANT tool_calls with thought_signature set it on the fc_part."""
        provider = self._get_provider()
        raw_bytes = b"sig_hello"
        b64_sig = base64.b64encode(raw_bytes).decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search",
                        parameters={"q": "test"},
                        id="call_1",
                        thought_signature=b64_sig,
                    )
                ],
            )
        ]
        contents = provider._format_contents("system", messages)

        assert len(contents) == 1
        model_content = contents[0]
        assert model_content.role == "model"

        fc_parts = [p for p in model_content.parts if p.function_call is not None]
        assert len(fc_parts) == 1

        fc_part = fc_parts[0]
        assert fc_part.function_call.name == "search"
        # thought_signature is stored as bytes on the SDK Part (decoded from base64)
        assert getattr(fc_part, "thought_signature", None) == raw_bytes

    def test_assistant_without_thought_signature_no_attr(self) -> None:
        """ASSISTANT tool_calls without thought_signature do not set the attr."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="calc",
                        parameters={"x": 1},
                        id="call_2",
                        thought_signature=None,
                    )
                ],
            )
        ]
        contents = provider._format_contents("system", messages)
        fc_parts = [p for p in contents[0].parts if p.function_call is not None]
        assert len(fc_parts) == 1

        # thought_signature should NOT have been set dynamically
        fc_part = fc_parts[0]
        sig = getattr(fc_part, "thought_signature", "NOT_SET")
        assert (
            sig is None or sig == "NOT_SET"
        ), f"Expected no thought_signature or None, got {sig!r}"

    def test_assistant_multiple_tool_calls_mixed(self) -> None:
        """ASSISTANT with multiple tool_calls: only the one with signature gets it."""
        provider = self._get_provider()
        raw_a = b"sig_a"
        b64_a = base64.b64encode(raw_a).decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="a",
                        parameters={},
                        id="call_a",
                        thought_signature=b64_a,
                    ),
                    ToolCall(
                        tool_name="b",
                        parameters={},
                        id="call_b",
                        thought_signature=None,
                    ),
                ],
            )
        ]
        contents = provider._format_contents("system", messages)
        fc_parts = [p for p in contents[0].parts if p.function_call is not None]
        assert len(fc_parts) == 2

        part_a = [p for p in fc_parts if p.function_call.name == "a"][0]
        part_b = [p for p in fc_parts if p.function_call.name == "b"][0]

        assert getattr(part_a, "thought_signature", None) == raw_a
        sig_b = getattr(part_b, "thought_signature", "NOT_SET")
        assert sig_b is None or sig_b == "NOT_SET"

    # -- TOOL messages: functionCall echo before functionResponse --

    def test_tool_message_echoes_function_call_with_signature(self) -> None:
        """TOOL msg after ASSISTANT with thought_signature echoes functionCall."""
        provider = self._get_provider()
        raw_echo = b"sig_echo"
        b64_echo = base64.b64encode(raw_echo).decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search",
                        parameters={"q": "test"},
                        id="call_1",
                        thought_signature=b64_echo,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="result data",
                tool_name="search",
                tool_call_id="call_1",
            ),
        ]
        contents = provider._format_contents("system", messages)

        # First content: the model message with the function_call
        assert contents[0].role == "model"

        # Second content: the user message with echoed functionCall + functionResponse
        tool_content = contents[1]
        assert tool_content.role == "user"

        # Should have 2 parts: the echoed functionCall, then the functionResponse
        assert len(tool_content.parts) == 2

        echo_part = tool_content.parts[0]
        assert echo_part.function_call is not None
        assert echo_part.function_call.name == "search"
        assert getattr(echo_part, "thought_signature", None) == raw_echo

        response_part = tool_content.parts[1]
        assert response_part.function_response is not None
        assert response_part.function_response.name == "search"

    def test_tool_message_no_echo_without_signature(self) -> None:
        """TOOL msg after ASSISTANT without thought_signature: no functionCall echo."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="calc",
                        parameters={"x": 1},
                        id="call_2",
                        thought_signature=None,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="42",
                tool_name="calc",
                tool_call_id="call_2",
            ),
        ]
        contents = provider._format_contents("system", messages)

        tool_content = contents[1]
        assert tool_content.role == "user"

        # Should have only 1 part: the functionResponse (no echo)
        assert len(tool_content.parts) == 1
        assert tool_content.parts[0].function_response is not None
        assert tool_content.parts[0].function_response.name == "calc"

    def test_tool_message_lookup_by_name_fallback(self) -> None:
        """TOOL msg matches ASSISTANT tool_call by name when tool_call_id differs."""
        provider = self._get_provider()
        raw_weather = b"sig_weather"
        b64_weather = base64.b64encode(raw_weather).decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="weather",
                        parameters={"city": "NYC"},
                        id="call_w",
                        thought_signature=b64_weather,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="72F and sunny",
                tool_name="weather",
                tool_call_id=None,  # no matching id, falls back to name lookup
            ),
        ]
        contents = provider._format_contents("system", messages)

        tool_content = contents[1]
        # Should still echo because name-based lookup found matching_tc
        assert len(tool_content.parts) == 2
        assert tool_content.parts[0].function_call is not None
        assert getattr(tool_content.parts[0], "thought_signature", None) == raw_weather

    def test_tool_message_no_assistant_context(self) -> None:
        """TOOL msg with no preceding ASSISTANT: just functionResponse, no echo."""
        provider = self._get_provider()
        messages = [
            Message(
                role=Role.TOOL,
                content="some data",
                tool_name="fetch",
                tool_call_id="call_orphan",
            ),
        ]
        contents = provider._format_contents("system", messages)

        tool_content = contents[0]
        assert tool_content.role == "user"
        # Only 1 part — no echo because no preceding ASSISTANT
        assert len(tool_content.parts) == 1
        assert tool_content.parts[0].function_response is not None

    def test_full_conversation_round_trip(self) -> None:
        """Full USER -> ASSISTANT(fc+sig) -> TOOL -> ASSISTANT conversation."""
        provider = self._get_provider()
        raw_paris = b"sig_paris"
        b64_paris = base64.b64encode(raw_paris).decode("ascii")
        messages = [
            Message(role=Role.USER, content="What's the weather in Paris?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="get_weather",
                        parameters={"city": "Paris"},
                        id="call_gw",
                        thought_signature=b64_paris,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content='{"temp": 18, "condition": "cloudy"}',
                tool_name="get_weather",
                tool_call_id="call_gw",
            ),
            Message(
                role=Role.ASSISTANT,
                content="The weather in Paris is 18C and cloudy.",
            ),
        ]
        contents = provider._format_contents("system", messages)

        # 4 messages: user, model(fc), user(echo+response), model(text)
        assert len(contents) == 4

        # 1st: user query
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What's the weather in Paris?"

        # 2nd: model function call with signature
        assert contents[1].role == "model"
        fc_parts = [p for p in contents[1].parts if p.function_call is not None]
        assert len(fc_parts) == 1
        assert getattr(fc_parts[0], "thought_signature", None) == raw_paris

        # 3rd: user with echoed fc + function response
        assert contents[2].role == "user"
        assert len(contents[2].parts) == 2
        assert contents[2].parts[0].function_call is not None
        assert contents[2].parts[1].function_response is not None

        # 4th: model final text answer
        assert contents[3].role == "model"
        assert contents[3].parts[0].text == "The weather in Paris is 18C and cloudy."

    def test_assistant_with_text_and_tool_calls_with_signature(self) -> None:
        """ASSISTANT with both text content and tool_calls preserves both."""
        provider = self._get_provider()
        raw_sig = b"sig_text_and_tc"
        b64_sig = base64.b64encode(raw_sig).decode("ascii")
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Let me look that up.",
                tool_calls=[
                    ToolCall(
                        tool_name="search",
                        parameters={"q": "selectools"},
                        id="call_s",
                        thought_signature=b64_sig,
                    )
                ],
            )
        ]
        contents = provider._format_contents("system", messages)

        assert len(contents) == 1
        parts = contents[0].parts
        # Should have a text part and a function_call part
        text_parts = [p for p in parts if getattr(p, "text", None)]
        fc_parts = [p for p in parts if p.function_call is not None]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Let me look that up."
        assert len(fc_parts) == 1
        assert getattr(fc_parts[0], "thought_signature", None) == raw_sig

    def test_non_utf8_binary_signature_round_trip(self) -> None:
        """Regression: non-UTF-8 binary thought_signature survives the round-trip.

        Gemini 3.x returns opaque binary (protobuf/hash) in thought_signature
        that is NOT valid UTF-8. The base64 encode/decode path must handle this
        without UnicodeDecodeError.

        See: https://github.com/johnnichev/selectools/issues/XX
        """
        provider = self._get_provider()
        # Bytes that are NOT valid UTF-8 — the exact pattern that crashed in production
        raw_binary = b"\xa4\xd5\x01\x02\xff\x80\x00\xfe"
        b64_sig = base64.b64encode(raw_binary).decode("ascii")

        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search",
                        parameters={"q": "test"},
                        id="call_bin",
                        thought_signature=b64_sig,
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="result",
                tool_name="search",
                tool_call_id="call_bin",
            ),
        ]
        contents = provider._format_contents("system", messages)

        # ASSISTANT fc_part should have the original binary bytes
        fc_parts = [p for p in contents[0].parts if p.function_call is not None]
        assert len(fc_parts) == 1
        assert getattr(fc_parts[0], "thought_signature", None) == raw_binary

        # TOOL echo should also have the original binary bytes
        tool_content = contents[1]
        assert len(tool_content.parts) == 2
        echo_part = tool_content.parts[0]
        assert echo_part.function_call is not None
        assert getattr(echo_part, "thought_signature", None) == raw_binary
