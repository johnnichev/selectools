"""Tests for conversation branching — ConversationMemory.branch() and SessionStore.branch()."""

from __future__ import annotations

from typing import List

import pytest

from selectools import Message, Role
from selectools.memory import ConversationMemory
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages(n: int) -> List[Message]:
    return [Message(role=Role.USER, content=f"msg {i}") for i in range(n)]


@pytest.fixture
def json_store(tmp_path):
    return JsonFileSessionStore(directory=str(tmp_path))


@pytest.fixture
def sqlite_store(tmp_path):
    return SQLiteSessionStore(db_path=str(tmp_path / "sessions.db"))


# ---------------------------------------------------------------------------
# ConversationMemory.branch()
# ---------------------------------------------------------------------------


def test_branch_returns_independent_copy():
    mem = ConversationMemory(max_messages=20)
    for m in _messages(3):
        mem.add(m)

    branch = mem.branch()

    # Modifying branch does not affect original
    branch.add(Message(role=Role.USER, content="branch only"))
    assert len(mem) == 3
    assert len(branch) == 4


def test_branch_copies_messages():
    mem = ConversationMemory(max_messages=20)
    for m in _messages(5):
        mem.add(m)

    branch = mem.branch()
    assert len(branch) == 5
    assert branch.get_history() == mem.get_history()


def test_branch_copies_summary():
    mem = ConversationMemory(max_messages=20)
    mem.summary = "previous summary"

    branch = mem.branch()
    assert branch.summary == "previous summary"

    # Changing branch summary does not affect original
    branch.summary = "branch summary"
    assert mem.summary == "previous summary"


def test_branch_copies_config():
    mem = ConversationMemory(max_messages=10, max_tokens=500)
    branch = mem.branch()
    assert branch.max_messages == 10
    assert branch.max_tokens == 500


def test_branch_of_empty_memory():
    mem = ConversationMemory(max_messages=5)
    branch = mem.branch()
    assert len(branch) == 0
    branch.add(Message(role=Role.USER, content="hello"))
    assert len(mem) == 0


def test_original_not_affected_when_branch_cleared():
    mem = ConversationMemory(max_messages=20)
    for m in _messages(4):
        mem.add(m)

    branch = mem.branch()
    branch.clear()

    assert len(mem) == 4
    assert len(branch) == 0


def test_branch_messages_are_independent_list():
    """branch()._messages is a new list, not the same object."""
    mem = ConversationMemory(max_messages=20)
    mem.add(Message(role=Role.USER, content="original"))
    branch = mem.branch()
    assert branch._messages is not mem._messages


# ---------------------------------------------------------------------------
# JsonFileSessionStore.branch()
# ---------------------------------------------------------------------------


def test_json_store_branch_creates_new_session(json_store):
    mem = ConversationMemory(max_messages=20)
    for m in _messages(3):
        mem.add(m)
    json_store.save("src", mem)

    json_store.branch("src", "dst")

    assert json_store.exists("dst")
    loaded = json_store.load("dst")
    assert loaded is not None
    assert len(loaded) == 3


def test_json_store_branch_independent(json_store):
    """Modifying dst session does not affect src."""
    mem = ConversationMemory(max_messages=20)
    mem.add(Message(role=Role.USER, content="original"))
    json_store.save("src", mem)

    json_store.branch("src", "dst")

    dst_mem = json_store.load("dst")
    assert dst_mem is not None
    dst_mem.add(Message(role=Role.USER, content="extra"))
    json_store.save("dst", dst_mem)

    src_reloaded = json_store.load("src")
    assert src_reloaded is not None
    assert len(src_reloaded) == 1  # unchanged


def test_json_store_branch_raises_when_source_missing(json_store):
    with pytest.raises(ValueError, match="not found"):
        json_store.branch("nonexistent", "dst")


# ---------------------------------------------------------------------------
# SQLiteSessionStore.branch()
# ---------------------------------------------------------------------------


def test_sqlite_store_branch_creates_new_session(sqlite_store):
    mem = ConversationMemory(max_messages=20)
    for m in _messages(4):
        mem.add(m)
    sqlite_store.save("src", mem)

    sqlite_store.branch("src", "dst")

    assert sqlite_store.exists("dst")
    loaded = sqlite_store.load("dst")
    assert loaded is not None
    assert len(loaded) == 4


def test_sqlite_store_branch_independent(sqlite_store):
    mem = ConversationMemory(max_messages=20)
    mem.add(Message(role=Role.USER, content="original"))
    sqlite_store.save("src", mem)

    sqlite_store.branch("src", "dst")

    dst_mem = sqlite_store.load("dst")
    assert dst_mem is not None
    dst_mem.add(Message(role=Role.USER, content="extra"))
    sqlite_store.save("dst", dst_mem)

    src_reloaded = sqlite_store.load("src")
    assert src_reloaded is not None
    assert len(src_reloaded) == 1


def test_sqlite_store_branch_raises_when_source_missing(sqlite_store):
    with pytest.raises(ValueError, match="not found"):
        sqlite_store.branch("ghost", "dst")
