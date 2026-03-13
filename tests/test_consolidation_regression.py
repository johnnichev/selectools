"""
Regression tests for v0.16.1 consolidation fixes.

Each test targets a specific bug found during the pre-v0.17 audit:
1. arun() missing tool_usage stats for sequential execution
2. astream() dead code — memory/hooks after yield
3. from_dict() producing invalid tool-pair boundaries
4. EntityMemory thread safety
5. SQLiteTripleStore WAL mode
6. KnowledgeMemory thread safety
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from unittest.mock import MagicMock

import pytest

from selectools.entity_memory import Entity, EntityMemory
from selectools.knowledge import KnowledgeMemory
from selectools.knowledge_graph import SQLiteTripleStore, Triple
from selectools.memory import ConversationMemory
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# ======================================================================
# 1. arun() tool_usage stats (regression for run/arun asymmetry)
# ======================================================================


class FakeProvider:
    name = "fake"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses=None):
        self._responses = responses or []
        self._idx = 0

    def _next_response(self, model):
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
        else:
            resp = Message(role=Role.ASSISTANT, content="Done")
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            model=model or "fake",
            provider="fake",
        )
        return resp, usage

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        return self._next_response(model)

    async def acomplete(self, *, model, system_prompt, messages, tools=None, **kw):
        return self._next_response(model)


@pytest.mark.asyncio
class TestArunToolUsageStats:
    async def test_arun_records_tool_usage_like_run(self) -> None:
        """arun() should populate tool_usage and tool_tokens for sequential tool calls."""
        from selectools.agent import Agent, AgentConfig
        from selectools.tools import Tool

        echo_tool = Tool(
            name="echo",
            description="Echo back",
            parameters=[],
            function=lambda: "echoed",
        )

        # Provider returns a tool call first, then a final response
        tc = ToolCall(tool_name="echo", parameters={}, id="tc1")
        provider = FakeProvider(
            responses=[
                Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                Message(role=Role.ASSISTANT, content="All done"),
            ]
        )

        agent = Agent(
            tools=[echo_tool],
            provider=provider,
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(max_iterations=3),
        )
        result = await agent.arun("Test tool usage")
        assert "echo" in agent.usage.tool_usage
        assert agent.usage.tool_usage["echo"] >= 1

    async def test_run_and_arun_tool_usage_symmetric(self) -> None:
        """run() and arun() should produce the same tool_usage."""
        from selectools.agent import Agent, AgentConfig
        from selectools.tools import Tool

        echo_tool = Tool(
            name="echo",
            description="Echo back",
            parameters=[],
            function=lambda: "echoed",
        )

        tc = ToolCall(tool_name="echo", parameters={}, id="tc1")

        # Sync
        provider_sync = FakeProvider(
            responses=[
                Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                Message(role=Role.ASSISTANT, content="Done sync"),
            ]
        )
        agent_sync = Agent(
            tools=[echo_tool],
            provider=provider_sync,
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(max_iterations=3),
        )
        agent_sync.run("Test")

        # Async
        provider_async = FakeProvider(
            responses=[
                Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
                Message(role=Role.ASSISTANT, content="Done async"),
            ]
        )
        agent_async = Agent(
            tools=[echo_tool],
            provider=provider_async,
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(max_iterations=3),
        )
        await agent_async.arun("Test")

        assert agent_sync.usage.tool_usage == agent_async.usage.tool_usage


# ======================================================================
# 2. from_dict() boundary fix (regression for invalid state on restore)
# ======================================================================


class TestFromDictBoundaryFix:
    def test_from_dict_fixes_orphaned_tool_messages(self) -> None:
        """from_dict() should strip orphaned TOOL messages at the start."""
        data = {
            "max_messages": 20,
            "max_tokens": None,
            "messages": [
                {
                    "role": "tool",
                    "content": "orphan result",
                    "tool_name": "search",
                    "tool_call_id": "tc1",
                },
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        mem = ConversationMemory.from_dict(data)
        history = mem.get_history()
        assert history[0].role == Role.USER
        assert len(history) == 2  # orphan removed

    def test_from_dict_fixes_orphaned_assistant_tool_calls(self) -> None:
        """from_dict() should strip assistant messages with tool_calls at start."""
        data = {
            "max_messages": 20,
            "max_tokens": None,
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search", "parameters": {}, "id": "tc1"}],
                },
                {"role": "tool", "content": "result", "tool_name": "search", "tool_call_id": "tc1"},
                {"role": "user", "content": "Thanks"},
            ],
        }
        mem = ConversationMemory.from_dict(data)
        history = mem.get_history()
        assert history[0].role == Role.USER

    def test_from_dict_preserves_valid_conversation(self) -> None:
        """from_dict() should not touch properly formed conversations."""
        data = {
            "max_messages": 20,
            "max_tokens": None,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        mem = ConversationMemory.from_dict(data)
        assert len(mem) == 2
        assert mem.get_history()[0].role == Role.USER

    def test_from_dict_preserves_summary(self) -> None:
        """from_dict() should preserve summary even with boundary fix."""
        data = {
            "max_messages": 20,
            "max_tokens": None,
            "messages": [
                {"role": "tool", "content": "orphan", "tool_name": "x", "tool_call_id": "tc1"},
                {"role": "user", "content": "Hello"},
            ],
            "summary": "Previous conversation summary",
        }
        mem = ConversationMemory.from_dict(data)
        assert mem.summary == "Previous conversation summary"
        assert mem.get_history()[0].role == Role.USER


# ======================================================================
# 3. EntityMemory thread safety
# ======================================================================


class TestEntityMemoryThreadSafety:
    def test_concurrent_updates_dont_crash(self) -> None:
        """Multiple threads updating EntityMemory simultaneously should not crash."""
        provider = MagicMock()
        em = EntityMemory(provider=provider, max_entities=100)

        errors: list = []

        def update_entities(thread_id: int) -> None:
            try:
                for i in range(50):
                    entity = Entity(
                        name=f"Entity-{thread_id}-{i}",
                        entity_type="test",
                        attributes={"thread": str(thread_id)},
                        first_mentioned=time.time(),
                        last_mentioned=time.time(),
                        mention_count=1,
                    )
                    em.update([entity])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update_entities, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(em.entities) > 0

    def test_concurrent_read_write(self) -> None:
        """Reading entities while writing should not crash."""
        provider = MagicMock()
        em = EntityMemory(provider=provider, max_entities=50)

        errors: list = []
        stop = threading.Event()

        def writer() -> None:
            try:
                for i in range(100):
                    em.update(
                        [
                            Entity(
                                name=f"E{i}",
                                entity_type="test",
                                attributes={},
                                first_mentioned=time.time(),
                                last_mentioned=time.time(),
                                mention_count=1,
                            )
                        ]
                    )
                stop.set()
            except Exception as exc:
                errors.append(exc)
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    _ = em.entities
                    _ = em.build_context()
            except Exception as exc:
                errors.append(exc)

        writer_t = threading.Thread(target=writer)
        reader_t = threading.Thread(target=reader)
        writer_t.start()
        reader_t.start()
        writer_t.join()
        reader_t.join()

        assert len(errors) == 0


# ======================================================================
# 4. SQLiteTripleStore WAL mode
# ======================================================================


class TestSQLiteTripleStoreWAL:
    def test_wal_mode_enabled(self, tmp_path: "os.PathLike[str]") -> None:
        """SQLiteTripleStore should use WAL journal mode."""
        db = os.path.join(str(tmp_path), "triples.db")
        store = SQLiteTripleStore(db_path=db)

        conn = sqlite3.connect(db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_concurrent_writes_dont_crash(self, tmp_path: "os.PathLike[str]") -> None:
        """Multiple threads writing triples should not raise sqlite3 errors."""
        db = os.path.join(str(tmp_path), "triples.db")
        store = SQLiteTripleStore(db_path=db, max_triples=500)

        errors: list = []

        def write_triples(thread_id: int) -> None:
            try:
                for i in range(20):
                    store.add(
                        Triple(
                            subject=f"S{thread_id}",
                            relation=f"rel{i}",
                            object=f"O{thread_id}-{i}",
                            confidence=0.9,
                            source_turn=i,
                            created_at=time.time(),
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_triples, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 100  # 5 threads × 20 triples


# ======================================================================
# 5. KnowledgeMemory thread safety
# ======================================================================


class TestKnowledgeMemoryThreadSafety:
    def test_concurrent_remember_calls(self, tmp_path: "os.PathLike[str]") -> None:
        """Multiple threads calling remember() simultaneously should not corrupt files."""
        km = KnowledgeMemory(directory=str(tmp_path))

        errors: list = []

        def remember_items(thread_id: int) -> None:
            try:
                for i in range(20):
                    km.remember(f"Fact {thread_id}-{i}", category="test")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=remember_items, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # All entries should be present (100 total)
        logs = km.get_recent_logs(days=1)
        entry_count = logs.count("[test]")
        assert entry_count == 100

    def test_concurrent_remember_persistent(self, tmp_path: "os.PathLike[str]") -> None:
        """Concurrent persistent writes should not interleave."""
        km = KnowledgeMemory(directory=str(tmp_path))

        errors: list = []

        def remember_persistent(thread_id: int) -> None:
            try:
                for i in range(10):
                    km.remember(
                        f"Persistent-{thread_id}-{i}",
                        category="fact",
                        persistent=True,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=remember_persistent, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        facts = km.get_persistent_facts()
        line_count = len([ln for ln in facts.strip().split("\n") if ln.strip()])
        assert line_count == 30  # 3 threads × 10 facts
