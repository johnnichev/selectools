"""
Tests for types.py — AgentResult and Message serialization unit tests.
"""

from __future__ import annotations

from selectools.types import AgentResult, Message, Role, ToolCall


class TestMessageToDict:
    """Tests for Message.to_dict() serialization."""

    def test_simple_user_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["tool_name"] is None
        assert d["tool_result"] is None
        assert d["tool_calls"] is None
        assert d["tool_call_id"] is None
        assert d["image_base64"] is None

    def test_tool_message_includes_tool_name(self) -> None:
        msg = Message(role=Role.TOOL, content="72F", tool_name="get_weather")
        d = msg.to_dict()
        assert d["tool_name"] == "get_weather"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"q": "test"}, id="tc1")
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        d = msg.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "search"
        assert d["tool_calls"][0]["parameters"] == {"q": "test"}
        assert d["tool_calls"][0]["id"] == "tc1"

    def test_tool_result_with_call_id(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content="result",
            tool_name="calc",
            tool_result="42",
            tool_call_id="tc1",
        )
        d = msg.to_dict()
        assert d["tool_name"] == "calc"
        assert d["tool_result"] == "42"
        assert d["tool_call_id"] == "tc1"


class TestMessageFromDict:
    """Tests for Message.from_dict() deserialization."""

    def test_simple_user_message_round_trip(self) -> None:
        original = Message(role=Role.USER, content="Hello")
        restored = Message.from_dict(original.to_dict())
        assert restored.role == Role.USER
        assert restored.content == "Hello"
        assert restored.image_path is None
        assert restored.image_base64 is None

    def test_all_roles_round_trip(self) -> None:
        for role in Role:
            original = Message(role=role, content=f"test-{role.value}")
            restored = Message.from_dict(original.to_dict())
            assert restored.role == role
            assert restored.content == f"test-{role.value}"

    def test_tool_message_preserves_tool_name(self) -> None:
        original = Message(
            role=Role.TOOL, content="result", tool_name="weather", tool_call_id="tc1"
        )
        restored = Message.from_dict(original.to_dict())
        assert restored.tool_name == "weather"
        assert restored.tool_call_id == "tc1"

    def test_tool_calls_round_trip(self) -> None:
        tc1 = ToolCall(tool_name="search", parameters={"q": "ai"}, id="tc1")
        tc2 = ToolCall(tool_name="calc", parameters={"expr": "1+1"}, id="tc2")
        original = Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2])

        restored = Message.from_dict(original.to_dict())
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].tool_name == "search"
        assert restored.tool_calls[0].parameters == {"q": "ai"}
        assert restored.tool_calls[0].id == "tc1"
        assert restored.tool_calls[1].tool_name == "calc"
        assert restored.tool_calls[1].id == "tc2"

    def test_image_base64_preserved_without_path(self) -> None:
        d = {
            "role": "user",
            "content": "What is this?",
            "image_base64": "abc123base64data",
        }
        restored = Message.from_dict(d)
        assert restored.image_base64 == "abc123base64data"
        assert restored.image_path is None

    def test_skips_post_init_image_encoding(self) -> None:
        """from_dict must not try to re-encode from image_path."""
        d = {
            "role": "user",
            "content": "image msg",
            "image_base64": "persisted_data",
        }
        restored = Message.from_dict(d)
        assert restored.image_base64 == "persisted_data"

    def test_none_tool_calls_round_trip(self) -> None:
        original = Message(role=Role.ASSISTANT, content="plain text")
        restored = Message.from_dict(original.to_dict())
        assert restored.tool_calls is None

    def test_empty_content(self) -> None:
        original = Message(role=Role.ASSISTANT, content="")
        restored = Message.from_dict(original.to_dict())
        assert restored.content == ""

    def test_missing_optional_fields_default_to_none(self) -> None:
        d = {"role": "user", "content": "minimal"}
        restored = Message.from_dict(d)
        assert restored.tool_name is None
        assert restored.tool_result is None
        assert restored.tool_calls is None
        assert restored.tool_call_id is None
        assert restored.image_base64 is None

    def test_tool_call_missing_id(self) -> None:
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "fn", "parameters": {"a": 1}}],
        }
        restored = Message.from_dict(d)
        assert restored.tool_calls is not None
        assert restored.tool_calls[0].id is None

    def test_tool_result_message_full_round_trip(self) -> None:
        original = Message(
            role=Role.TOOL,
            content="72F and sunny",
            tool_name="get_weather",
            tool_result="72F and sunny",
            tool_call_id="tc_abc",
        )
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == Role.TOOL
        assert restored.content == "72F and sunny"
        assert restored.tool_name == "get_weather"
        assert restored.tool_result == "72F and sunny"
        assert restored.tool_call_id == "tc_abc"


class TestAgentResult:
    def test_content_property_delegates_to_message(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="Hello world")
        result = AgentResult(message=msg)
        assert result.content == "Hello world"

    def test_role_property_delegates_to_message(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="test")
        result = AgentResult(message=msg)
        assert result.role == Role.ASSISTANT

    def test_defaults(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="x")
        result = AgentResult(message=msg)
        assert result.tool_name is None
        assert result.tool_args == {}
        assert result.iterations == 0
        assert result.tool_calls == []

    def test_tool_calls_list_populated(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"q": "test"})
        msg = Message(role=Role.ASSISTANT, content="done")
        result = AgentResult(
            message=msg,
            tool_name="search",
            tool_args={"q": "test"},
            iterations=2,
            tool_calls=[tc],
        )
        assert result.tool_name == "search"
        assert result.tool_args == {"q": "test"}
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"
