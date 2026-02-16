"""
Comprehensive tests for ConversationMemory (memory.py).

Tests cover:
- Initialization and validation
- add() and add_many() methods
- get_history() and get_recent()
- clear()
- Message count limit enforcement (sliding window)
- Token-based limit enforcement
- to_dict() serialization
- __len__, __bool__, __repr__
"""

from __future__ import annotations

import pytest

from selectools.memory import ConversationMemory
from selectools.types import Message, Role


def _msg(content: str, role: Role = Role.USER) -> Message:
    """Helper to create messages quickly."""
    return Message(role=role, content=content)


class TestConversationMemoryInit:
    """Tests for ConversationMemory initialization."""

    def test_default_max_messages(self) -> None:
        mem = ConversationMemory()
        assert mem.max_messages == 20

    def test_default_max_tokens_none(self) -> None:
        mem = ConversationMemory()
        assert mem.max_tokens is None

    def test_custom_max_messages(self) -> None:
        mem = ConversationMemory(max_messages=5)
        assert mem.max_messages == 5

    def test_custom_max_tokens(self) -> None:
        mem = ConversationMemory(max_tokens=1000)
        assert mem.max_tokens == 1000

    def test_max_messages_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_messages must be at least 1"):
            ConversationMemory(max_messages=0)

    def test_max_messages_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_messages must be at least 1"):
            ConversationMemory(max_messages=-1)

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            ConversationMemory(max_tokens=0)

    def test_max_tokens_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            ConversationMemory(max_tokens=-1)

    def test_starts_empty(self) -> None:
        mem = ConversationMemory()
        assert len(mem) == 0
        assert mem.get_history() == []


class TestAdd:
    """Tests for add() method."""

    def test_add_single_message(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("Hello"))

        assert len(mem) == 1
        assert mem.get_history()[0].content == "Hello"

    def test_add_multiple_messages(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("One"))
        mem.add(_msg("Two"))
        mem.add(_msg("Three"))

        assert len(mem) == 3

    def test_add_preserves_order(self) -> None:
        mem = ConversationMemory()
        for i in range(5):
            mem.add(_msg(f"msg-{i}"))

        history = mem.get_history()
        for i in range(5):
            assert history[i].content == f"msg-{i}"

    def test_add_enforces_limit(self) -> None:
        mem = ConversationMemory(max_messages=3)
        for i in range(5):
            mem.add(_msg(f"msg-{i}"))

        assert len(mem) == 3
        history = mem.get_history()
        assert history[0].content == "msg-2"
        assert history[1].content == "msg-3"
        assert history[2].content == "msg-4"


class TestAddMany:
    """Tests for add_many() method."""

    def test_add_many_basic(self) -> None:
        mem = ConversationMemory()
        msgs = [_msg("A"), _msg("B"), _msg("C")]
        mem.add_many(msgs)

        assert len(mem) == 3

    def test_add_many_preserves_order(self) -> None:
        mem = ConversationMemory()
        msgs = [_msg(f"msg-{i}") for i in range(5)]
        mem.add_many(msgs)

        history = mem.get_history()
        for i in range(5):
            assert history[i].content == f"msg-{i}"

    def test_add_many_enforces_limit(self) -> None:
        mem = ConversationMemory(max_messages=3)
        msgs = [_msg(f"msg-{i}") for i in range(6)]
        mem.add_many(msgs)

        assert len(mem) == 3
        history = mem.get_history()
        assert history[0].content == "msg-3"

    def test_add_many_empty_list(self) -> None:
        mem = ConversationMemory()
        mem.add_many([])
        assert len(mem) == 0

    def test_add_many_after_add(self) -> None:
        mem = ConversationMemory(max_messages=4)
        mem.add(_msg("first"))
        mem.add_many([_msg("second"), _msg("third"), _msg("fourth")])

        assert len(mem) == 4
        history = mem.get_history()
        assert history[0].content == "first"
        assert history[3].content == "fourth"


class TestGetHistory:
    """Tests for get_history() method."""

    def test_returns_copy(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("Hello"))

        history = mem.get_history()
        history.clear()

        assert len(mem) == 1

    def test_empty_history(self) -> None:
        mem = ConversationMemory()
        assert mem.get_history() == []


class TestGetRecent:
    """Tests for get_recent() method."""

    def test_get_recent_subset(self) -> None:
        mem = ConversationMemory()
        for i in range(5):
            mem.add(_msg(f"msg-{i}"))

        recent = mem.get_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "msg-3"
        assert recent[1].content == "msg-4"

    def test_get_recent_more_than_available(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("only"))

        recent = mem.get_recent(5)
        assert len(recent) == 1
        assert recent[0].content == "only"

    def test_get_recent_exact_count(self) -> None:
        mem = ConversationMemory()
        for i in range(3):
            mem.add(_msg(f"msg-{i}"))

        recent = mem.get_recent(3)
        assert len(recent) == 3

    def test_get_recent_zero_raises(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("hello"))

        with pytest.raises(ValueError, match="n must be at least 1"):
            mem.get_recent(0)

    def test_get_recent_negative_raises(self) -> None:
        mem = ConversationMemory()
        with pytest.raises(ValueError, match="n must be at least 1"):
            mem.get_recent(-1)


class TestClear:
    """Tests for clear() method."""

    def test_clear_empties_history(self) -> None:
        mem = ConversationMemory()
        for i in range(5):
            mem.add(_msg(f"msg-{i}"))

        mem.clear()
        assert len(mem) == 0
        assert mem.get_history() == []

    def test_clear_already_empty(self) -> None:
        mem = ConversationMemory()
        mem.clear()
        assert len(mem) == 0

    def test_add_after_clear(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("before"))
        mem.clear()
        mem.add(_msg("after"))

        assert len(mem) == 1
        assert mem.get_history()[0].content == "after"


class TestMessageCountLimit:
    """Tests for message count-based sliding window."""

    def test_sliding_window_removes_oldest(self) -> None:
        mem = ConversationMemory(max_messages=3)
        mem.add(_msg("A"))
        mem.add(_msg("B"))
        mem.add(_msg("C"))
        mem.add(_msg("D"))

        assert len(mem) == 3
        contents = [m.content for m in mem.get_history()]
        assert contents == ["B", "C", "D"]

    def test_max_messages_one(self) -> None:
        mem = ConversationMemory(max_messages=1)
        mem.add(_msg("first"))
        mem.add(_msg("second"))

        assert len(mem) == 1
        assert mem.get_history()[0].content == "second"


class TestTokenLimit:
    """Tests for token-based limit enforcement."""

    def test_token_limit_removes_oldest(self) -> None:
        mem = ConversationMemory(max_messages=100, max_tokens=20)

        mem.add(_msg("A" * 40))
        mem.add(_msg("B" * 40))

        assert len(mem) >= 1
        last = mem.get_history()[-1]
        assert "B" in last.content

    def test_token_limit_keeps_at_least_one(self) -> None:
        mem = ConversationMemory(max_messages=100, max_tokens=1)
        mem.add(_msg("This is a long message that exceeds one token"))

        assert len(mem) == 1


class TestToDict:
    """Tests for to_dict() serialization."""

    def test_empty_memory_to_dict(self) -> None:
        mem = ConversationMemory()
        d = mem.to_dict()

        assert d["max_messages"] == 20
        assert d["max_tokens"] is None
        assert d["message_count"] == 0
        assert d["messages"] == []

    def test_to_dict_with_messages(self) -> None:
        mem = ConversationMemory(max_messages=5, max_tokens=500)
        mem.add(_msg("Hello", Role.USER))
        mem.add(_msg("Hi there!", Role.ASSISTANT))

        d = mem.to_dict()

        assert d["max_messages"] == 5
        assert d["max_tokens"] == 500
        assert d["message_count"] == 2
        assert len(d["messages"]) == 2

    def test_to_dict_messages_are_dicts(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("Test"))

        d = mem.to_dict()
        assert isinstance(d["messages"][0], dict)


class TestDunderMethods:
    """Tests for __len__, __bool__, __repr__."""

    def test_len_empty(self) -> None:
        mem = ConversationMemory()
        assert len(mem) == 0

    def test_len_with_messages(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("A"))
        mem.add(_msg("B"))
        assert len(mem) == 2

    def test_bool_always_true(self) -> None:
        mem = ConversationMemory()
        assert bool(mem) is True

    def test_bool_true_when_empty(self) -> None:
        """Memory object should be truthy even when empty."""
        mem = ConversationMemory()
        assert mem  # truthy

    def test_repr(self) -> None:
        mem = ConversationMemory(max_messages=10, max_tokens=500)
        mem.add(_msg("Hello"))

        r = repr(mem)
        assert "ConversationMemory" in r
        assert "max_messages=10" in r
        assert "max_tokens=500" in r
        assert "current_messages=1" in r

    def test_repr_defaults(self) -> None:
        mem = ConversationMemory()
        r = repr(mem)
        assert "max_messages=20" in r
        assert "max_tokens=None" in r
        assert "current_messages=0" in r


class TestMixedRoles:
    """Tests with messages of different roles."""

    def test_preserves_roles(self) -> None:
        mem = ConversationMemory()
        mem.add(_msg("Question", Role.USER))
        mem.add(_msg("Answer", Role.ASSISTANT))
        mem.add(_msg("Result", Role.TOOL))

        history = mem.get_history()
        assert history[0].role == Role.USER
        assert history[1].role == Role.ASSISTANT
        assert history[2].role == Role.TOOL

    def test_sliding_window_with_roles(self) -> None:
        mem = ConversationMemory(max_messages=2)
        mem.add(_msg("Q1", Role.USER))
        mem.add(_msg("A1", Role.ASSISTANT))
        mem.add(_msg("Q2", Role.USER))

        history = mem.get_history()
        assert len(history) == 2
        assert history[0].role == Role.ASSISTANT
        assert history[1].role == Role.USER
