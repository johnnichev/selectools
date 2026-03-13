"""
Regression tests for v0.16.0 Memory & Persistence bug fixes.

Each test targets a specific bug found during the v0.16.0 audit.
"""

from __future__ import annotations

import os
import time

import pytest

from selectools.entity_memory import Entity, EntityMemory
from selectools.knowledge import KnowledgeMemory
from selectools.memory import ConversationMemory
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore
from selectools.tools import tool
from selectools.types import Message, Role


@tool()
def _dummy_tool(x: str) -> str:
    """A dummy tool for testing."""
    return f"result:{x}"


# ======================================================================
# Bug 1: memory.clear() didn't reset _last_trimmed
# ======================================================================


class TestBug1ClearResetsLastTrimmed:
    def test_clear_resets_last_trimmed(self) -> None:
        mem = ConversationMemory(max_messages=3)
        for i in range(5):
            mem.add(Message(role=Role.USER, content=f"msg-{i}"))
        assert len(mem._last_trimmed) > 0
        mem.clear()
        assert mem._last_trimmed == []

    def test_clear_then_add_no_stale_trimmed(self) -> None:
        mem = ConversationMemory(max_messages=2)
        mem.add(Message(role=Role.USER, content="a"))
        mem.add(Message(role=Role.USER, content="b"))
        mem.add(Message(role=Role.USER, content="c"))
        mem.clear()
        mem.add(Message(role=Role.USER, content="fresh"))
        assert len(mem) == 1
        assert mem._last_trimmed == []


# ======================================================================
# Bug 2: session store overwrites user-provided memory
# ======================================================================


class TestBug2SessionStoreRespectsUserMemory:
    def test_user_memory_not_overwritten_by_session(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        old_mem = ConversationMemory(max_messages=50)
        old_mem.add(Message(role=Role.USER, content="old session data"))
        store.save("sess1", old_mem)

        user_mem = ConversationMemory(max_messages=10)
        user_mem.add(Message(role=Role.USER, content="user provided"))

        from selectools.agent.core import Agent, AgentConfig

        agent = Agent(
            tools=[_dummy_tool],
            provider=_make_mock_provider(),
            config=AgentConfig(session_store=store, session_id="sess1"),
            memory=user_mem,
        )
        assert len(agent.memory) == 1
        assert agent.memory.get_history()[0].content == "user provided"


# ======================================================================
# Bug 3: entity_memory malformed attributes crash build_context()
# ======================================================================


class TestBug3EntityMalformedAttributes:
    def test_non_dict_attributes_handled(self) -> None:
        entity = Entity(
            name="Test",
            entity_type="person",
            attributes="not a dict",  # type: ignore[arg-type]
        )
        mem = EntityMemory(provider=_make_mock_provider(), max_entities=10)
        mem._entities["test"] = entity
        ctx = mem.build_context()
        assert "[Known Entities]" in ctx
        assert "Test" in ctx


# ======================================================================
# Bug 4: knowledge.py midnight race (two datetime.now() calls)
# ======================================================================


class TestBug4KnowledgeMidnightRace:
    def test_remember_uses_consistent_date(self, tmp_path: "os.PathLike[str]") -> None:
        km = KnowledgeMemory(directory=str(tmp_path))
        km.remember("test entry", category="general")
        log_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".log")]
        assert len(log_files) == 1
        with open(os.path.join(str(tmp_path), log_files[0]), "r") as f:
            content = f.read()
        date_in_filename = log_files[0].replace(".log", "")
        assert date_in_filename in content


# ======================================================================
# Bug 5: knowledge.py truncation exceeds max_context_chars
# ======================================================================


class TestBug5KnowledgeTruncationLimit:
    def test_build_context_respects_max_chars(self, tmp_path: "os.PathLike[str]") -> None:
        km = KnowledgeMemory(directory=str(tmp_path), max_context_chars=100)
        km.remember("A" * 200, persistent=True)
        ctx = km.build_context()
        assert len(ctx) <= 100
        assert ctx.endswith("(truncated)")

    def test_truncation_includes_suffix_within_limit(self, tmp_path: "os.PathLike[str]") -> None:
        limit = 50
        km = KnowledgeMemory(directory=str(tmp_path), max_context_chars=limit)
        km.remember("X" * 200, persistent=True)
        ctx = km.build_context()
        assert len(ctx) <= limit


# ======================================================================
# Bug 6: sessions.py list() doesn't delete expired sessions
# ======================================================================


class TestBug6SessionListCleansExpired:
    def test_json_list_deletes_expired_files(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        mem = ConversationMemory(max_messages=50)
        mem.add(Message(role=Role.USER, content="hello"))
        store.save("old", mem)

        time.sleep(1.1)
        results = store.list()
        assert len(results) == 0
        assert not os.path.exists(os.path.join(str(tmp_path), "old.json"))

    def test_sqlite_list_deletes_expired_rows(self, tmp_path: "os.PathLike[str]") -> None:
        db_path = os.path.join(str(tmp_path), "sessions.db")
        store = SQLiteSessionStore(db_path=db_path, default_ttl=1)
        mem = ConversationMemory(max_messages=50)
        mem.add(Message(role=Role.USER, content="hello"))
        store.save("old", mem)

        time.sleep(1.1)
        results = store.list()
        assert len(results) == 0
        assert not store.exists("old")


# ======================================================================
# Helpers
# ======================================================================


def _make_mock_provider():
    from unittest.mock import MagicMock

    from selectools.usage import UsageStats

    provider = MagicMock()
    provider.complete.return_value = (
        Message(role=Role.ASSISTANT, content="mock response"),
        UsageStats(10, 10, 20, 0.001, "mock", "mock"),
    )
    return provider
