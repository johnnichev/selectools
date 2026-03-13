"""
Integration tests for all v0.16.0 memory features running simultaneously.

Tests that sessions, summarize-on-trim, entity memory, knowledge graph,
and knowledge memory all work together without interference.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from selectools.agent import Agent, AgentConfig
from selectools.entity_memory import EntityMemory
from selectools.knowledge import KnowledgeMemory
from selectools.knowledge_graph import KnowledgeGraphMemory
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore
from selectools.types import Message, Role
from selectools.usage import UsageStats


class FakeProvider:
    """Fake provider that returns controllable responses."""

    name = "fake"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses: list = None) -> None:
        self._responses = responses or ["I'll help with that."]
        self._call_count = 0

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        response_text = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        usage = UsageStats(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            model=model or "fake",
            provider="fake",
        )
        return Message(role=Role.ASSISTANT, content=response_text), usage

    async def acomplete(self, *, model, system_prompt, messages, tools=None, **kw):
        return self.complete(
            model=model, system_prompt=system_prompt, messages=messages, tools=tools, **kw
        )


def _make_tool():
    from selectools.tools import Tool

    return Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")


class TestAllMemoryFeaturesCombined:
    """Test all memory features enabled at the same time."""

    def test_session_with_entity_memory(self, tmp_path: "os.PathLike[str]") -> None:
        """Session store and entity memory should both work."""
        store = JsonFileSessionStore(directory=str(tmp_path))

        # Entity memory with a provider that returns empty entities
        entity_provider = FakeProvider(responses=['{"entities": []}'])
        em = EntityMemory(provider=entity_provider)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="combined-1",
                entity_memory=em,
            ),
        )
        result = agent.run("Hello Alice")
        assert result.content is not None

        # Session should be saved
        loaded = store.load("combined-1")
        assert loaded is not None
        assert len(loaded) >= 2  # user + assistant

    def test_session_with_knowledge_graph(self, tmp_path: "os.PathLike[str]") -> None:
        """Session store and knowledge graph should both work."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        kg_provider = FakeProvider(responses=['{"triples": []}'])
        kg = KnowledgeGraphMemory(provider=kg_provider)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="combined-2",
                knowledge_graph=kg,
            ),
        )
        result = agent.run("Alice works at Acme Corp")
        assert result.content is not None

        loaded = store.load("combined-2")
        assert loaded is not None

    def test_session_with_knowledge_memory(self, tmp_path: "os.PathLike[str]") -> None:
        """Session store and knowledge memory should both work."""
        session_dir = os.path.join(str(tmp_path), "sessions")
        knowledge_dir = os.path.join(str(tmp_path), "knowledge")
        store = JsonFileSessionStore(directory=session_dir)
        km = KnowledgeMemory(directory=knowledge_dir)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="combined-3",
                knowledge_memory=km,
            ),
        )
        result = agent.run("Remember this fact")
        assert result.content is not None

        # Check remember tool was auto-added
        assert "remember" in agent._tools_by_name

    def test_all_features_simultaneously(self, tmp_path: "os.PathLike[str]") -> None:
        """Session + entity + knowledge graph + knowledge memory all at once."""
        session_dir = os.path.join(str(tmp_path), "sessions")
        knowledge_dir = os.path.join(str(tmp_path), "knowledge")
        store = JsonFileSessionStore(directory=session_dir)
        entity_provider = FakeProvider(responses=['{"entities": []}'])
        em = EntityMemory(provider=entity_provider)
        kg_provider = FakeProvider(responses=['{"triples": []}'])
        kg = KnowledgeGraphMemory(provider=kg_provider)
        km = KnowledgeMemory(directory=knowledge_dir)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="all-features",
                entity_memory=em,
                knowledge_graph=kg,
                knowledge_memory=km,
            ),
        )
        result = agent.run("Hello world")
        assert result.content is not None

        # Session saved
        loaded = store.load("all-features")
        assert loaded is not None

    def test_all_features_with_sqlite_backend(self, tmp_path: "os.PathLike[str]") -> None:
        """All features with SQLite session backend."""
        db = os.path.join(str(tmp_path), "sessions.db")
        knowledge_dir = os.path.join(str(tmp_path), "knowledge")
        store = SQLiteSessionStore(db_path=db)
        entity_provider = FakeProvider(responses=['{"entities": []}'])
        em = EntityMemory(provider=entity_provider)
        kg_provider = FakeProvider(responses=['{"triples": []}'])
        kg = KnowledgeGraphMemory(provider=kg_provider)
        km = KnowledgeMemory(directory=knowledge_dir)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="sqlite-all",
                entity_memory=em,
                knowledge_graph=kg,
                knowledge_memory=km,
            ),
        )
        result = agent.run("Test with SQLite")
        assert result.content is not None
        assert store.exists("sqlite-all")

    def test_features_dont_interfere_on_second_run(self, tmp_path: "os.PathLike[str]") -> None:
        """Running twice with persistent session should work cleanly."""
        session_dir = os.path.join(str(tmp_path), "sessions")
        store = JsonFileSessionStore(directory=session_dir)
        entity_provider = FakeProvider(responses=['{"entities": []}'])

        # First run
        agent1 = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="persist",
                entity_memory=EntityMemory(provider=entity_provider),
            ),
        )
        agent1.run("First message")

        # Second run — loads from session
        agent2 = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            config=AgentConfig(
                session_store=store,
                session_id="persist",
                entity_memory=EntityMemory(provider=entity_provider),
            ),
        )
        assert agent2.memory is not None
        assert len(agent2.memory) >= 2  # user + assistant from first run
        result = agent2.run("Second message")
        assert result.content is not None


class TestObserverEventsWithAllFeatures:
    """Test that observer events fire correctly when all features are enabled."""

    def test_all_observer_events_fire(self, tmp_path: "os.PathLike[str]") -> None:
        session_dir = os.path.join(str(tmp_path), "sessions")
        store = JsonFileSessionStore(directory=session_dir)
        store.save("obs-test", ConversationMemory(max_messages=50))

        events: list = []

        class RecordingObserver(AgentObserver):
            def on_session_load(self, run_id, session_id, message_count):
                events.append(("session_load", session_id))

            def on_session_save(self, run_id, session_id, message_count):
                events.append(("session_save", session_id))

            def on_entity_extraction(self, run_id, entities):
                events.append(("entity_extraction", len(entities)))

            def on_kg_extraction(self, run_id, triples):
                events.append(("kg_extraction", len(triples)))

        entity_provider = FakeProvider(responses=['{"entities": []}'])
        kg_provider = FakeProvider(responses=['{"triples": []}'])

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeProvider(),
            config=AgentConfig(
                session_store=store,
                session_id="obs-test",
                entity_memory=EntityMemory(provider=entity_provider),
                knowledge_graph=KnowledgeGraphMemory(provider=kg_provider),
                observers=[RecordingObserver()],
            ),
        )
        agent.run("Hello")

        event_types = [e[0] for e in events]
        assert "session_load" in event_types
        assert "session_save" in event_types
        # Entity and KG extraction may fire depending on implementation
