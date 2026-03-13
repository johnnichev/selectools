"""
Async path tests for v0.16.0 memory features.

Tests that arun() properly handles session auto-save, entity extraction,
knowledge graph extraction, and summarize-on-trim.
"""

from __future__ import annotations

import os

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


class FakeAsyncProvider:
    """Fake provider with async support."""

    name = "fake"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses: list = None) -> None:
        self._responses = responses or ["Async response"]
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


@pytest.mark.asyncio
class TestAsyncSessionAutoSave:
    async def test_arun_auto_saves_session(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="async-1"),
        )
        result = await agent.arun("Hello async")
        assert result.content is not None

        loaded = store.load("async-1")
        assert loaded is not None
        assert len(loaded) >= 2

    async def test_arun_auto_loads_session(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        # Pre-save a session
        mem = ConversationMemory(max_messages=50)
        mem.add(Message(role=Role.USER, content="Previous"))
        mem.add(Message(role=Role.ASSISTANT, content="Previous response"))
        store.save("async-2", mem)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            config=AgentConfig(session_store=store, session_id="async-2"),
        )
        assert agent.memory is not None
        assert len(agent.memory) == 2

        result = await agent.arun("Continue")
        assert result.content is not None

        # Session should be updated
        loaded = store.load("async-2")
        assert loaded is not None
        assert len(loaded) >= 4  # 2 previous + user + assistant

    async def test_arun_with_sqlite_session(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="async-sqlite"),
        )
        result = await agent.arun("Hello SQLite async")
        assert result.content is not None
        assert store.exists("async-sqlite")


@pytest.mark.asyncio
class TestAsyncEntityMemory:
    async def test_arun_with_entity_memory(self, tmp_path: "os.PathLike[str]") -> None:
        entity_provider = FakeAsyncProvider(responses=['{"entities": []}'])
        em = EntityMemory(provider=entity_provider)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(entity_memory=em),
        )
        result = await agent.arun("Alice met Bob in Paris")
        assert result.content is not None


@pytest.mark.asyncio
class TestAsyncKnowledgeGraph:
    async def test_arun_with_knowledge_graph(self) -> None:
        kg_provider = FakeAsyncProvider(responses=['{"triples": []}'])
        kg = KnowledgeGraphMemory(provider=kg_provider)

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(knowledge_graph=kg),
        )
        result = await agent.arun("Bob works at Acme Corp")
        assert result.content is not None


@pytest.mark.asyncio
class TestAsyncKnowledgeMemory:
    async def test_arun_with_knowledge_memory(self, tmp_path: "os.PathLike[str]") -> None:
        km = KnowledgeMemory(directory=str(tmp_path))

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(knowledge_memory=km),
        )
        result = await agent.arun("Remember this fact")
        assert result.content is not None
        assert "remember" in agent._tools_by_name


@pytest.mark.asyncio
class TestAsyncCombinedFeatures:
    async def test_arun_all_features(self, tmp_path: "os.PathLike[str]") -> None:
        session_dir = os.path.join(str(tmp_path), "sessions")
        knowledge_dir = os.path.join(str(tmp_path), "knowledge")
        store = JsonFileSessionStore(directory=session_dir)
        entity_provider = FakeAsyncProvider(responses=['{"entities": []}'])
        kg_provider = FakeAsyncProvider(responses=['{"triples": []}'])

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(
                session_store=store,
                session_id="async-all",
                entity_memory=EntityMemory(provider=entity_provider),
                knowledge_graph=KnowledgeGraphMemory(provider=kg_provider),
                knowledge_memory=KnowledgeMemory(directory=knowledge_dir),
            ),
        )
        result = await agent.arun("Hello with all features")
        assert result.content is not None
        assert store.exists("async-all")

    async def test_arun_observer_events(self, tmp_path: "os.PathLike[str]") -> None:
        session_dir = os.path.join(str(tmp_path), "sessions")
        store = JsonFileSessionStore(directory=session_dir)
        store.save("obs-async", ConversationMemory(max_messages=50))

        events: list = []

        class RecordingObserver(AgentObserver):
            def on_session_load(self, run_id, session_id, message_count):
                events.append("session_load")

            def on_session_save(self, run_id, session_id, message_count):
                events.append("session_save")

        agent = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            config=AgentConfig(
                session_store=store,
                session_id="obs-async",
                observers=[RecordingObserver()],
            ),
        )
        await agent.arun("Hello async observer")
        assert "session_load" in events
        assert "session_save" in events

    async def test_arun_persists_across_instantiations(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))

        # First async run
        agent1 = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            memory=ConversationMemory(max_messages=50),
            config=AgentConfig(session_store=store, session_id="persist-async"),
        )
        await agent1.arun("First async")

        # Second async run — loads from store
        agent2 = Agent(
            tools=[_make_tool()],
            provider=FakeAsyncProvider(),
            config=AgentConfig(session_store=store, session_id="persist-async"),
        )
        assert agent2.memory is not None
        assert len(agent2.memory) >= 2
        result = await agent2.arun("Second async")
        assert result.content is not None
