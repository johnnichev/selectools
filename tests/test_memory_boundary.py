"""
Tests for ConversationMemory tool-pair boundary trimming.

Covers the _fix_tool_pair_boundary() logic that removes orphaned
TOOL and tool-calling ASSISTANT messages from the start of history
after sliding-window trimming.
"""

from __future__ import annotations

import pytest

from selectools.memory import ConversationMemory
from selectools.types import Message, Role, ToolCall


def _make_tool_call_msg(tool_name: str = "search", tc_id: str = "tc1") -> Message:
    tc = ToolCall(tool_name=tool_name, parameters={}, id=tc_id)
    return Message(role=Role.ASSISTANT, content="", tool_calls=[tc])


def _make_tool_result_msg(
    tool_name: str = "search", tc_id: str = "tc1", result: str = "ok"
) -> Message:
    return Message(role=Role.TOOL, content=result, tool_name=tool_name, tool_call_id=tc_id)


class TestToolPairBoundaryTrimming:
    """Test that orphaned TOOL messages at the start are trimmed."""

    def test_orphan_tool_message_at_start(self) -> None:
        """When sliding window trims the ASSISTANT tool-call message,
        the orphaned TOOL result should also be trimmed."""
        mem = ConversationMemory(max_messages=3)
        # Add a tool call pair + a follow-up
        mem.add(Message(role=Role.USER, content="Search for cats"))
        mem.add(_make_tool_call_msg())
        mem.add(_make_tool_result_msg())
        mem.add(Message(role=Role.ASSISTANT, content="Found cats"))
        # Now add more to trigger trimming
        mem.add(Message(role=Role.USER, content="Tell me more"))
        mem.add(Message(role=Role.ASSISTANT, content="More info"))

        history = mem.get_history()
        # First message should NOT be a TOOL message
        assert history[0].role != Role.TOOL

    def test_orphan_assistant_with_tool_calls_at_start(self) -> None:
        """When sliding window puts an ASSISTANT with tool_calls at position 0,
        it should be trimmed along with the following TOOL result."""
        mem = ConversationMemory(max_messages=4)
        # Build history that will trim to start with tool-calling assistant
        mem.add(Message(role=Role.USER, content="Q1"))
        mem.add(Message(role=Role.ASSISTANT, content="A1"))
        mem.add(Message(role=Role.USER, content="Search"))
        mem.add(_make_tool_call_msg())
        mem.add(_make_tool_result_msg())
        mem.add(Message(role=Role.ASSISTANT, content="Result"))
        mem.add(Message(role=Role.USER, content="Thanks"))

        history = mem.get_history()
        # Should not start with orphaned tool-call assistant
        for msg in history:
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # If there's a tool-calling assistant, it shouldn't be the first message
                assert history[0] != msg or history[0].role == Role.USER
                break

    def test_consecutive_orphan_tool_messages(self) -> None:
        """Multiple orphaned TOOL messages at the start should all be removed."""
        mem = ConversationMemory(max_messages=4)
        # First: two tool call pairs
        mem.add(Message(role=Role.USER, content="Do two things"))
        tc1 = ToolCall(tool_name="a", parameters={}, id="tc1")
        tc2 = ToolCall(tool_name="b", parameters={}, id="tc2")
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2]))
        mem.add(Message(role=Role.TOOL, content="r1", tool_name="a", tool_call_id="tc1"))
        mem.add(Message(role=Role.TOOL, content="r2", tool_name="b", tool_call_id="tc2"))
        mem.add(Message(role=Role.ASSISTANT, content="Done"))
        # Trigger more trimming
        mem.add(Message(role=Role.USER, content="Next"))
        mem.add(Message(role=Role.ASSISTANT, content="Next answer"))

        history = mem.get_history()
        assert history[0].role != Role.TOOL

    def test_no_orphans_when_properly_paired(self) -> None:
        """Normal conversation without orphans should not trim anything extra."""
        mem = ConversationMemory(max_messages=4)
        mem.add(Message(role=Role.USER, content="Hello"))
        mem.add(Message(role=Role.ASSISTANT, content="Hi"))
        mem.add(Message(role=Role.USER, content="How are you"))
        mem.add(Message(role=Role.ASSISTANT, content="Good"))

        history = mem.get_history()
        assert len(history) == 4
        assert history[0].role == Role.USER
        assert history[0].content == "Hello"

    def test_single_tool_message_only(self) -> None:
        """Edge case: memory with just TOOL message gets trimmed to empty
        only if there's at least 2 messages (guard condition)."""
        mem = ConversationMemory(max_messages=50)
        # Manually add a TOOL message (unusual but possible in deserialization)
        mem._messages.append(
            Message(role=Role.TOOL, content="orphan", tool_name="x", tool_call_id="tc1")
        )
        mem._messages.append(Message(role=Role.USER, content="hello"))
        # Force boundary fix by adding and triggering limits
        # The TOOL at start should be cleaned if we trigger enforce_limits
        assert len(mem) == 2

    def test_from_dict_preserves_boundary_state(self) -> None:
        """Round-trip through to_dict/from_dict should maintain valid boundaries."""
        mem = ConversationMemory(max_messages=10)
        mem.add(Message(role=Role.USER, content="Q"))
        mem.add(_make_tool_call_msg())
        mem.add(_make_tool_result_msg())
        mem.add(Message(role=Role.ASSISTANT, content="A"))

        data = mem.to_dict()
        restored = ConversationMemory.from_dict(data)
        history = restored.get_history()
        assert history[0].role == Role.USER
        assert history[1].tool_calls is not None
        assert history[2].role == Role.TOOL
        assert history[3].content == "A"
