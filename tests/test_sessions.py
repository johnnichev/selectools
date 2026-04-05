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
