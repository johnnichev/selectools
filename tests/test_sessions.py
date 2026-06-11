"""
Comprehensive tests for persistent session storage (sessions.py).

Tests cover:
- JsonFileSessionStore: save/load, TTL, delete, list, exists
- SQLiteSessionStore: save/load, TTL, delete, list, exists
- Agent integration: auto-load on init, auto-save after run
"""

from __future__ import annotations

import json
import os
import time

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import JsonFileSessionStore, SessionMetadata, SQLiteSessionStore
from selectools.types import Message, Role, ToolCall


def _memory_with_messages(*contents: str) -> ConversationMemory:
    mem = ConversationMemory(max_messages=50)
    for c in contents:
        mem.add(Message(role=Role.USER, content=c))
    return mem


# ======================================================================
# JsonFileSessionStore
# ======================================================================


class TestJsonFileSessionStoreSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = _memory_with_messages("Hello", "World")
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded.get_history()[0].content == "Hello"

    def test_load_nonexistent_returns_none(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        assert store.load("nonexistent") is None

    def test_save_overwrites_existing(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _memory_with_messages("v1"))
        store.save("s1", _memory_with_messages("v2"))

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.get_history()[0].content == "v2"

    def test_preserves_created_at_on_overwrite(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _memory_with_messages("v1"))

        path = os.path.join(str(tmp_path), "s1.json")
        with open(path, "r") as f:
            first_created = json.load(f)["created_at"]

        time.sleep(0.01)
        store.save("s1", _memory_with_messages("v2"))

        with open(path, "r") as f:
            data = json.load(f)
        assert data["created_at"] == first_created
        assert data["updated_at"] > first_created

    def test_preserves_tool_calls(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = ConversationMemory()
        mem.add(Message(role=Role.USER, content="Search for AI"))
        tc = ToolCall(tool_name="search", parameters={"q": "ai"}, id="tc1")
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc]))
        mem.add(Message(role=Role.TOOL, content="result", tool_name="search", tool_call_id="tc1"))
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        history = loaded.get_history()
        assert history[1].tool_calls[0].tool_name == "search"
        assert history[2].tool_name == "search"

    def test_preserves_summary(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = _memory_with_messages("Hello")
        mem.summary = "User said hello"
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.summary == "User said hello"


class TestJsonFileSessionStoreTTL:
    def test_expired_session_returns_none(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        store.save("s1", _memory_with_messages("Hello"))

        # Manually backdate updated_at
        path = os.path.join(str(tmp_path), "s1.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 10
        with open(path, "w") as f:
            json.dump(data, f)

        assert store.load("s1") is None

    def test_no_ttl_never_expires(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=None)
        store.save("s1", _memory_with_messages("Hello"))

        path = os.path.join(str(tmp_path), "s1.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = 0  # ancient
        with open(path, "w") as f:
            json.dump(data, f)

        assert store.load("s1") is not None


class TestJsonFileSessionStoreDeleteListExists:
    def test_delete_existing(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _memory_with_messages("Hello"))
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        assert store.delete("nope") is False

    def test_exists(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        assert store.exists("s1") is False
        store.save("s1", _memory_with_messages("Hello"))
        assert store.exists("s1") is True

    def test_exists_expired(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        store.save("s1", _memory_with_messages("Hello"))

        path = os.path.join(str(tmp_path), "s1.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 10
        with open(path, "w") as f:
            json.dump(data, f)

        assert store.exists("s1") is False

    def test_list_sessions(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}
        for s in sessions:
            assert isinstance(s, SessionMetadata)

    def test_list_excludes_expired(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        store.save("fresh", _memory_with_messages("ok"))
        store.save("stale", _memory_with_messages("old"))

        path = os.path.join(str(tmp_path), "stale.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 10
        with open(path, "w") as f:
            json.dump(data, f)

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].session_id == "fresh"

    def test_list_empty(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        assert store.list() == []

    def test_creates_directory(self, tmp_path: "os.PathLike[str]") -> None:
        d = os.path.join(str(tmp_path), "nested", "dir")
        store = JsonFileSessionStore(directory=d)
        assert os.path.isdir(d)
        store.save("s1", _memory_with_messages("ok"))
        assert store.exists("s1")

    def test_empty_session_id_raises(self, tmp_path: "os.PathLike[str]") -> None:
        """Regression: empty session_id created a file named '.json'.

        An empty string passed all validation checks (basename('') == '',
        no '..' or null bytes) and resulted in a '.json' file that could
        collide with other hidden files or cause confusion.
        """
        store = JsonFileSessionStore(directory=str(tmp_path))
        with pytest.raises(ValueError, match="must not be empty"):
            store.save("", _memory_with_messages("Hello"))


# ======================================================================
# SQLiteSessionStore
# ======================================================================


class TestSQLiteSessionStoreSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        mem = _memory_with_messages("Hello", "World")
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded.get_history()[0].content == "Hello"

    def test_load_nonexistent_returns_none(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        assert store.load("nonexistent") is None

    def test_save_overwrites_existing(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        store.save("s1", _memory_with_messages("v1"))
        store.save("s1", _memory_with_messages("v2"))

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.get_history()[0].content == "v2"

    def test_preserves_tool_calls(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        mem = ConversationMemory()
        mem.add(Message(role=Role.USER, content="Calculate"))
        tc = ToolCall(tool_name="calc", parameters={"x": 1}, id="tc1")
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc]))
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.get_history()[1].tool_calls[0].tool_name == "calc"

    def test_preserves_summary(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        mem = _memory_with_messages("Hello")
        mem.summary = "Greeted"
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.summary == "Greeted"


class TestSQLiteSessionStoreTTL:
    def test_expired_session_returns_none(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db, default_ttl=1)
        store.save("s1", _memory_with_messages("Hello"))

        # Backdate updated_at
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time() - 10, "s1"),
        )
        conn.commit()
        conn.close()

        assert store.load("s1") is None

    def test_no_ttl_never_expires(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db, default_ttl=None)
        store.save("s1", _memory_with_messages("Hello"))

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE sessions SET updated_at = 0 WHERE session_id = ?",
            ("s1",),
        )
        conn.commit()
        conn.close()

        assert store.load("s1") is not None


class TestSQLiteSessionStoreDeleteListExists:
    def test_delete_existing(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        store.save("s1", _memory_with_messages("Hello"))
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        assert store.delete("nope") is False

    def test_exists(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        assert store.exists("s1") is False
        store.save("s1", _memory_with_messages("Hello"))
        assert store.exists("s1") is True

    def test_list_sessions(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}

    def test_list_excludes_expired(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db, default_ttl=1)
        store.save("fresh", _memory_with_messages("ok"))
        store.save("stale", _memory_with_messages("old"))

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time() - 10, "stale"),
        )
        conn.commit()
        conn.close()

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].session_id == "fresh"

    def test_list_empty(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        assert store.list() == []


# ======================================================================
# Agent integration
# ======================================================================


class TestAgentSessionIntegration:
    """Test session auto-load and auto-save through the agent."""

    def _make_fake_provider(self) -> object:
        from selectools.types import Message, Role
        from selectools.usage import UsageStats

        class FakeProvider:
            name = "fake"
            supports_streaming = False
            supports_async = False

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                usage = UsageStats(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost_usd=0.0001,
                    model=model or "fake",
                    provider="fake",
                )
                return Message(role=Role.ASSISTANT, content="response"), usage

        return FakeProvider()

    def _make_tool(self):
        from selectools.tools import Tool

        return Tool(
            name="echo",
            description="Echo input",
            parameters=[],
            function=lambda: "ok",
        )

    def test_auto_load_on_init(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig

        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = _memory_with_messages("previous message")
        store.save("session-1", mem)

        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            config=AgentConfig(session_store=store, session_id="session-1"),
        )
        assert agent.memory is not None
        assert len(agent.memory) == 1
        assert agent.memory.get_history()[0].content == "previous message"

    def test_auto_load_no_existing_session(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig

        store = JsonFileSessionStore(directory=str(tmp_path))

        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            config=AgentConfig(session_store=store, session_id="nonexistent"),
        )
        # No session to load, memory stays None
        assert agent.memory is None

    def test_auto_save_after_run(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig

        store = JsonFileSessionStore(directory=str(tmp_path))

        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="session-1"),
        )
        agent.run("Hello")

        # Verify session was saved
        loaded = store.load("session-1")
        assert loaded is not None
        assert len(loaded) >= 1

    def test_history_persists_across_instantiations(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig

        store = JsonFileSessionStore(directory=str(tmp_path))

        # First agent run
        agent1 = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="persist-test"),
        )
        agent1.run("Turn 1")

        # Second agent run — loads from session store
        agent2 = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            config=AgentConfig(session_store=store, session_id="persist-test"),
        )
        assert agent2.memory is not None
        history = agent2.memory.get_history()
        assert any(m.content == "Turn 1" for m in history)

    def test_auto_save_with_sqlite(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)

        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="sqlite-test"),
        )
        agent.run("Hello SQLite")

        loaded = store.load("sqlite-test")
        assert loaded is not None
        assert len(loaded) >= 1

    def test_session_observer_events(self, tmp_path: "os.PathLike[str]") -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.observer import AgentObserver

        store = JsonFileSessionStore(directory=str(tmp_path))
        # Pre-save a session so load fires
        store.save("obs-test", _memory_with_messages("prior"))

        events: list = []

        class RecordingObserver(AgentObserver):
            def on_session_load(self, run_id, session_id, message_count):
                events.append(("load", session_id, message_count))

            def on_session_save(self, run_id, session_id, message_count):
                events.append(("save", session_id, message_count))

        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            config=AgentConfig(
                session_store=store,
                session_id="obs-test",
                observers=[RecordingObserver()],
            ),
        )
        agent.run("Hello")

        load_events = [e for e in events if e[0] == "load"]
        save_events = [e for e in events if e[0] == "save"]
        assert len(load_events) == 1
        assert load_events[0][1] == "obs-test"
        assert len(save_events) == 1
        assert save_events[0][1] == "obs-test"

    def test_no_session_config_no_side_effects(self) -> None:
        """Agent without session config should work exactly as before."""
        from selectools.agent import Agent, AgentConfig

        mem = ConversationMemory()
        agent = Agent(
            tools=[self._make_tool()],
            provider=self._make_fake_provider(),
            memory=mem,
            config=AgentConfig(),
        )
        result = agent.run("Hello")
        assert result.content == "response"


# ======================================================================
# Cross-session search
# ======================================================================


def _search_memory(*contents: str) -> ConversationMemory:
    """Build a memory alternating USER/ASSISTANT roles over *contents*."""
    mem = ConversationMemory(max_messages=50)
    for i, c in enumerate(contents):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        mem.add(Message(role=role, content=c))
    return mem


class TestJsonFileSessionStoreSearch:
    def test_match_in_one_of_several_sessions(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _search_memory("The weather is sunny", "Indeed it is"))
        store.save("s2", _search_memory("I found a billing discrepancy", "Looking into the bill"))
        store.save("s3", _search_memory("Tell me about dragons", "Dragons breathe fire"))

        results = store.search("billing discrepancy")
        assert len(results) == 1
        assert results[0].session_id == "s2"
        assert results[0].score > 0
        assert any("billing" in m.lower() for m in results[0].matched_messages)

    def test_namespace_filtering(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _search_memory("quartz crystal"), namespace="agent-a")
        store.save("s1", _search_memory("quartz pebble"), namespace="agent-b")
        store.save("s2", _search_memory("quartz boulder"))

        results = store.search("quartz", namespace="agent-a")
        assert len(results) == 1
        assert results[0].session_id == "s1"
        # Returned id loads with the same namespace
        assert store.load(results[0].session_id, namespace="agent-a") is not None

    def test_no_match_returns_empty_list(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _search_memory("hello world"))
        assert store.search("xylophone") == []

    def test_limit(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        for i in range(6):
            store.save(f"s{i}", _search_memory(f"quartz number {i}"))
        assert len(store.search("quartz", limit=3)) == 3
        assert len(store.search("quartz", limit=10)) == 6

    def test_score_ordering_more_hits_ranks_higher(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("one-hit", _search_memory("quartz here", "nothing", "nothing else"))
        store.save("three-hits", _search_memory("quartz here", "quartz there", "quartz everywhere"))

        results = store.search("quartz")
        assert [r.session_id for r in results] == ["three-hits", "one-hit"]
        assert results[0].score > results[1].score

    def test_case_insensitive(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _search_memory("QUARTZ Crystal"))
        results = store.search("quartz")
        assert len(results) == 1

    def test_tool_messages_not_searched(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        mem = ConversationMemory(max_messages=50)
        mem.add(Message(role=Role.USER, content="run the tool"))
        mem.add(Message(role=Role.TOOL, content="quartz in tool output", tool_name="t"))
        store.save("s1", mem)
        assert store.search("quartz") == []

    def test_snippet_length_capped(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        long_msg = ("lorem ipsum " * 100) + "quartz" + (" dolor sit" * 100)
        store.save("s1", _search_memory(long_msg))
        results = store.search("quartz")
        assert len(results) == 1
        snippet = results[0].matched_messages[0]
        assert len(snippet) <= 200
        assert "quartz" in snippet


class TestSQLiteSessionStoreSearch:
    def _store(self, tmp_path: "os.PathLike[str]") -> SQLiteSessionStore:
        return SQLiteSessionStore(db_path=str(os.path.join(str(tmp_path), "sessions.db")))

    def test_match_in_one_of_several_sessions(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        store.save("s1", _search_memory("The weather is sunny", "Indeed it is"))
        store.save("s2", _search_memory("I found a billing discrepancy", "Looking into the bill"))
        store.save("s3", _search_memory("Tell me about dragons", "Dragons breathe fire"))

        results = store.search("billing discrepancy")
        assert results[0].session_id == "s2"
        assert results[0].score > 0
        assert any("billing" in m.lower() for m in results[0].matched_messages)

    def test_namespace_filtering(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        store.save("s1", _search_memory("quartz crystal"), namespace="agent-a")
        store.save("s1", _search_memory("quartz pebble"), namespace="agent-b")
        store.save("s2", _search_memory("quartz boulder"))

        results = store.search("quartz", namespace="agent-a")
        assert len(results) == 1
        assert results[0].session_id == "s1"
        assert store.load(results[0].session_id, namespace="agent-a") is not None

    def test_no_match_returns_empty_list(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        store.save("s1", _search_memory("hello world"))
        assert store.search("xylophone") == []

    def test_limit(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        for i in range(6):
            store.save(f"s{i}", _search_memory(f"quartz number {i}"))
        assert len(store.search("quartz", limit=3)) == 3
        assert len(store.search("quartz", limit=10)) == 6

    def test_fts5_ranking_sanity(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        assert store._fts_enabled is True
        store.save("one-hit", _search_memory("quartz here", "nothing at all", "nothing else"))
        store.save("three-hits", _search_memory("quartz here", "quartz there", "quartz everywhere"))

        results = store.search("quartz")
        assert [r.session_id for r in results] == ["three-hits", "one-hit"]
        assert results[0].score > results[1].score

    def test_old_database_upgrade_path(self, tmp_path: "os.PathLike[str]") -> None:
        """A DB created before the search feature must be searchable and intact."""
        import sqlite3

        db_path = os.path.join(str(tmp_path), "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                memory_json TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        now = time.time()
        old_mem = _search_memory("legacy quartz conversation", "noted")
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            ("legacy", json.dumps(old_mem.to_dict()), len(old_mem), now, now),
        )
        conn.commit()
        conn.close()

        store = SQLiteSessionStore(db_path=db_path)
        results = store.search("quartz")
        assert [r.session_id for r in results] == ["legacy"]

        # Existing data untouched: load still round-trips
        loaded = store.load("legacy")
        assert loaded is not None
        assert loaded.get_history()[0].content == "legacy quartz conversation"

        # New saves are searchable alongside lazily indexed old rows
        store.save("fresh", _search_memory("fresh quartz too"))
        ids = {r.session_id for r in store.search("quartz")}
        assert ids == {"legacy", "fresh"}

    def test_like_fallback_warns_and_matches(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        store.save("s1", _search_memory("quartz crystal", "nothing"))
        store.save("s2", _search_memory("plain talk"))
        store._fts_enabled = False

        with pytest.warns(RuntimeWarning, match="FTS5"):
            results = store.search("quartz")
        assert [r.session_id for r in results] == ["s1"]

    def test_fts5_detection_cross_build_reuse(
        self, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DB created on an FTS5 build, reopened on a non-FTS5 build, must degrade.

        SQLite short-circuits ``CREATE VIRTUAL TABLE IF NOT EXISTS`` on
        table-name existence BEFORE resolving the module, so against an
        existing table the create "succeeds" even when fts5 is unavailable;
        only later reads/writes of the virtual table raise.  The connection
        wrapper below mirrors exactly that verified behavior, so the store
        exercises its real code paths.
        """
        import sqlite3
        from typing import Any

        db_path = os.path.join(str(tmp_path), "sessions.db")
        fts_store = SQLiteSessionStore(db_path=db_path)
        assert fts_store._fts_enabled is True
        fts_store.save("old", _search_memory("legacy quartz conversation"))

        class NoFts5Connection:
            """Simulates a sqlite build without FTS5 against an existing DB."""

            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def _check(self, sql: str) -> None:
                if "__fts5_probe__" in sql:
                    raise sqlite3.OperationalError("no such module: fts5")
                if "session_messages_fts" in sql and "IF NOT EXISTS" not in sql:
                    # Real access to the virtual table fails; IF NOT EXISTS
                    # short-circuits silently on the existing table name.
                    raise sqlite3.OperationalError("no such module: fts5")

            def execute(self, sql: str, *args: Any) -> Any:
                self._check(sql)
                return self._conn.execute(sql, *args)

            def executemany(self, sql: str, *args: Any) -> Any:
                self._check(sql)
                return self._conn.executemany(sql, *args)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._conn, name)

        real_connect = sqlite3.connect
        monkeypatch.setattr(
            sqlite3, "connect", lambda *a, **kw: NoFts5Connection(real_connect(*a, **kw))
        )

        store = SQLiteSessionStore(db_path=db_path)
        # Pre-fix this failed open: _fts_enabled stayed True and
        # save()/delete()/search() raised "no such module: fts5".
        assert store._fts_enabled is False

        store.save("new", _search_memory("fresh quartz too"))
        with pytest.warns(RuntimeWarning, match="FTS5"):
            results = store.search("quartz")
        assert {r.session_id for r in results} == {"old", "new"}
        assert store.delete("new") is True

    def test_bm25_bounded_hit_count_dominates(self, tmp_path: "os.PathLike[str]") -> None:
        """One rare-term hit must not outscore three common-term hits.

        bm25's IDF makes a single rare-term match score ~ln(N) while a
        common term scores near zero; summing the raw rank unbounded
        inverted the documented "more matches ranks higher" invariant on
        IDF-skewed corpora.  The bounded per-message contribution
        ``1 + r/(1+r)`` keeps hit count strictly dominant.
        """
        store = self._store(tmp_path)
        # Inflate the corpus so "commonword" has rock-bottom IDF and the
        # rare term a huge one (the old formula needs ~1000 messages to
        # invert: 1 rare hit ~10 points vs 3 common hits ~3 points).
        for i in range(40):
            store.save(
                f"filler-{i}",
                _search_memory(*[f"commonword filler {i} {j}" for j in range(25)]),
            )
        store.save("rare-once", _search_memory("the zyzzyva appeared once"))
        store.save(
            "common-thrice",
            _search_memory("commonword one", "commonword two", "commonword three"),
        )

        results = store.search("zyzzyva commonword", limit=100)
        ids = [r.session_id for r in results]
        assert "common-thrice" in ids and "rare-once" in ids
        assert ids.index("common-thrice") < ids.index("rare-once")

    def test_backfill_reindexes_stale_rows(self, tmp_path: "os.PathLike[str]") -> None:
        """Rows rewritten by a search-unaware (older) writer get re-indexed.

        Mixed-version protection: an older library version updates
        ``sessions`` without maintaining the FTS index, leaving
        ``indexed_at < updated_at``.  The next search must re-index.
        """
        import sqlite3

        db_path = os.path.join(str(tmp_path), "sessions.db")
        store = SQLiteSessionStore(db_path=db_path)
        store.save("s1", _search_memory("original topaz topic"))
        assert [r.session_id for r in store.search("topaz")] == ["s1"]

        new_mem = _search_memory("now about obsidian instead")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE sessions SET memory_json = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(new_mem.to_dict()), time.time() + 1, "s1"),
        )
        conn.commit()
        conn.close()

        assert store.search("topaz") == []
        assert [r.session_id for r in store.search("obsidian")] == ["s1"]

    def test_delete_removes_from_search(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        store.save("s1", _search_memory("quartz crystal"))
        assert store.search("quartz")
        store.delete("s1")
        assert store.search("quartz") == []

    def test_tool_messages_not_searched(self, tmp_path: "os.PathLike[str]") -> None:
        store = self._store(tmp_path)
        mem = ConversationMemory(max_messages=50)
        mem.add(Message(role=Role.USER, content="run the tool"))
        mem.add(Message(role=Role.TOOL, content="quartz in tool output", tool_name="t"))
        store.save("s1", mem)
        assert store.search("quartz") == []


class TestSessionStoreProtocolSearch:
    def test_default_search_raises_not_implemented(self) -> None:
        from selectools.sessions import SessionStore

        class LegacyStore(SessionStore):
            pass

        with pytest.raises(NotImplementedError):
            LegacyStore().search("anything")

    def test_search_result_is_frozen(self) -> None:
        from selectools.sessions import SessionSearchResult

        result = SessionSearchResult(session_id="s", score=1.0, matched_messages=["m"])
        with pytest.raises(Exception):
            result.score = 2.0  # type: ignore[misc]
