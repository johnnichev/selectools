"""
Tests for EntityMemory (entity_memory.py).

Tests cover:
- Entity dataclass serialization
- EntityMemory extraction, update, deduplication, pruning
- build_context output
- Round-trip serialization
- Agent integration (context injection, extraction after run)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from selectools.entity_memory import Entity, EntityMemory
from selectools.types import Message, Role
from selectools.usage import UsageStats


def _usage(model: str = "fake") -> UsageStats:
    return UsageStats(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.0001,
        model=model,
        provider="fake",
    )


class FakeExtractionProvider:
    """Returns a canned JSON array of entities."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, entities_json: str = "[]") -> None:
        self._response = entities_json
        self.calls: List[Dict[str, Any]] = []

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        self.calls.append({"model": model, "messages": messages})
        return Message(role=Role.ASSISTANT, content=self._response), _usage(model)


class FailingProvider:
    name = "failing"
    supports_streaming = False
    supports_async = False

    def complete(self, **kw):
        raise RuntimeError("down")


# ======================================================================
# Entity dataclass
# ======================================================================


class TestEntity:
    def test_to_dict_round_trip(self) -> None:
        e = Entity(
            name="Alice",
            entity_type="person",
            attributes={"role": "engineer"},
            first_mentioned=100.0,
            last_mentioned=200.0,
            mention_count=3,
        )
        d = e.to_dict()
        restored = Entity.from_dict(d)
        assert restored.name == "Alice"
        assert restored.entity_type == "person"
        assert restored.attributes == {"role": "engineer"}
        assert restored.first_mentioned == 100.0
        assert restored.last_mentioned == 200.0
        assert restored.mention_count == 3

    def test_from_dict_defaults(self) -> None:
        e = Entity.from_dict({"name": "Bob"})
        assert e.entity_type == "unknown"
        assert e.attributes == {}
        assert e.mention_count == 1


# ======================================================================
# EntityMemory — extraction
# ======================================================================


class TestEntityMemoryExtraction:
    def test_extracts_entities_from_messages(self) -> None:
        entities_json = json.dumps(
            [
                {"name": "Alice", "entity_type": "person", "attributes": {"role": "CEO"}},
                {"name": "Acme Corp", "entity_type": "organization", "attributes": {}},
            ]
        )
        provider = FakeExtractionProvider(entities_json)
        em = EntityMemory(provider=provider)

        msgs = [
            Message(role=Role.USER, content="Alice from Acme Corp called."),
            Message(role=Role.ASSISTANT, content="I see, what did she say?"),
        ]
        entities = em.extract_entities(msgs)

        assert len(entities) == 2
        assert entities[0].name == "Alice"
        assert entities[1].name == "Acme Corp"
        assert len(provider.calls) == 1

    def test_empty_messages_returns_empty(self) -> None:
        provider = FakeExtractionProvider()
        em = EntityMemory(provider=provider)
        assert em.extract_entities([]) == []
        assert len(provider.calls) == 0

    def test_provider_failure_returns_empty(self) -> None:
        em = EntityMemory(provider=FailingProvider())
        result = em.extract_entities([Message(role=Role.USER, content="test")])
        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        provider = FakeExtractionProvider("not json at all")
        em = EntityMemory(provider=provider)
        result = em.extract_entities([Message(role=Role.USER, content="test")])
        assert result == []

    def test_respects_relevance_window(self) -> None:
        entities_json = json.dumps([{"name": "X", "entity_type": "concept"}])
        provider = FakeExtractionProvider(entities_json)
        em = EntityMemory(provider=provider, relevance_window=2)

        msgs = [Message(role=Role.USER, content=f"msg-{i}") for i in range(10)]
        em.extract_entities(msgs)

        # The provider should only receive 2 recent messages
        call_msgs = provider.calls[0]["messages"]
        content = call_msgs[0].content
        assert "msg-8" in content
        assert "msg-9" in content

    def test_strips_code_fences(self) -> None:
        response = '```json\n[{"name": "Python", "entity_type": "technology"}]\n```'
        provider = FakeExtractionProvider(response)
        em = EntityMemory(provider=provider)
        result = em.extract_entities([Message(role=Role.USER, content="I love Python")])
        assert len(result) == 1
        assert result[0].name == "Python"


# ======================================================================
# EntityMemory — update and deduplication
# ======================================================================


class TestEntityMemoryUpdate:
    def test_adds_new_entities(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider())
        em.update(
            [
                Entity(name="Alice", entity_type="person"),
                Entity(name="Bob", entity_type="person"),
            ]
        )
        assert len(em.entities) == 2

    def test_deduplicates_by_name(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider())
        em.update([Entity(name="Alice", entity_type="person")])
        em.update([Entity(name="alice", entity_type="person")])  # same, different case
        assert len(em.entities) == 1
        assert em.entities[0].mention_count == 2

    def test_merges_attributes(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider())
        em.update([Entity(name="Alice", entity_type="person", attributes={"role": "CEO"})])
        em.update([Entity(name="Alice", entity_type="person", attributes={"age": "30"})])

        alice = em.entities[0]
        assert alice.attributes == {"role": "CEO", "age": "30"}

    def test_lru_pruning(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider(), max_entities=3)
        em.update(
            [
                Entity(name="A", entity_type="x"),
                Entity(name="B", entity_type="x"),
                Entity(name="C", entity_type="x"),
            ]
        )
        # Adding one more should prune the oldest-mentioned
        em.update([Entity(name="D", entity_type="x")])
        assert len(em.entities) == 3
        names = {e.name for e in em.entities}
        assert "D" in names


# ======================================================================
# EntityMemory — build_context
# ======================================================================


class TestEntityMemoryBuildContext:
    def test_empty_returns_empty_string(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider())
        assert em.build_context() == ""

    def test_formats_entities(self) -> None:
        em = EntityMemory(provider=FakeExtractionProvider())
        em.update(
            [
                Entity(name="Alice", entity_type="person", attributes={"role": "CEO"}),
                Entity(name="Python", entity_type="technology"),
            ]
        )
        ctx = em.build_context()
        assert "[Known Entities]" in ctx
        assert "Alice [person]" in ctx
        assert "role: CEO" in ctx
        assert "Python [technology]" in ctx


# ======================================================================
# EntityMemory — serialization
# ======================================================================


class TestEntityMemorySerialization:
    def test_round_trip(self) -> None:
        provider = FakeExtractionProvider()
        em = EntityMemory(provider=provider, max_entities=25, relevance_window=5)
        em.update(
            [
                Entity(name="Alice", entity_type="person", attributes={"x": "y"}),
            ]
        )

        d = em.to_dict()
        restored = EntityMemory.from_dict(d, provider)
        assert len(restored.entities) == 1
        assert restored.entities[0].name == "Alice"
        assert restored._max_entities == 25
        assert restored._relevance_window == 5

    def test_empty_round_trip(self) -> None:
        provider = FakeExtractionProvider()
        em = EntityMemory(provider=provider)
        d = em.to_dict()
        restored = EntityMemory.from_dict(d, provider)
        assert len(restored.entities) == 0


# ======================================================================
# Agent integration
# ======================================================================


class TestEntityMemoryAgentIntegration:
    def _make_tool(self):
        from selectools.tools import Tool

        return Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")

    def test_entity_context_injected(self) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class RecordingProvider:
            name = "recording"
            supports_streaming = False
            supports_async = False

            def __init__(self):
                self.last_messages: List[Message] = []

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                self.last_messages = list(messages)
                return Message(role=Role.ASSISTANT, content="ok"), _usage(model)

        recording = RecordingProvider()
        em = EntityMemory(provider=FakeExtractionProvider())
        em.update([Entity(name="Alice", entity_type="person")])

        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(entity_memory=em),
        )
        agent.run("Tell me about Alice")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert any("[Known Entities]" in m.content for m in system_msgs)

    def test_no_entity_memory_no_injection(self) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        class RecordingProvider:
            name = "recording"
            supports_streaming = False
            supports_async = False

            def __init__(self):
                self.last_messages: List[Message] = []

            def complete(self, *, model, system_prompt, messages, tools=None, **kw):
                self.last_messages = list(messages)
                return Message(role=Role.ASSISTANT, content="ok"), _usage(model)

        recording = RecordingProvider()
        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(),
        )
        agent.run("Hello")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert not any("[Known Entities]" in m.content for m in system_msgs)

    def test_extraction_failure_doesnt_crash(self) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        em = EntityMemory(provider=FailingProvider())

        agent = Agent(
            tools=[self._make_tool()],
            provider=FakeExtractionProvider(
                json.dumps([])
            ),  # hack: this returns (Message, Usage) from complete
            memory=ConversationMemory(),
            config=AgentConfig(entity_memory=em),
        )

        # Override provider to return proper format
        class SimpleProvider:
            name = "simple"
            supports_streaming = False
            supports_async = False

            def complete(self, **kw):
                return Message(role=Role.ASSISTANT, content="ok"), _usage()

        agent.provider = SimpleProvider()
        result = agent.run("Hello")
        assert result.content == "ok"
