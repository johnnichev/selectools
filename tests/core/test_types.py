"""
Tests for types.py — AgentResult dataclass unit tests.
"""

from __future__ import annotations

from selectools.types import AgentResult, Message, Role, ToolCall


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
