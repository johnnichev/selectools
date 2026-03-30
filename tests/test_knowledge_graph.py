"""
Tests for KnowledgeGraphMemory (knowledge_graph.py).

Tests cover:
- Triple dataclass serialization
- InMemoryTripleStore: add, query, pruning, clear
- SQLiteTripleStore: add, query, pruning, clear, persistence
- KnowledgeGraphMemory: extraction, query_relevant, build_context
- Round-trip serialization
- Agent integration (context injection, extraction after run)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from selectools.knowledge_graph import (
    InMemoryTripleStore,
    KnowledgeGraphMemory,
    SQLiteTripleStore,
    Triple,
)
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
    """Returns a canned JSON array of triples."""

    name = "fake"
    supports_streaming = False
    supports_async = False

    def __init__(self, triples_json: str = "[]") -> None:
        self._response = triples_json
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
# Triple dataclass
# ======================================================================


class TestTriple:
    def test_to_dict_round_trip(self) -> None:
        t = Triple(
            subject="Alice",
            relation="works_at",
            object="Acme Corp",
            confidence=0.9,
            source_turn=5,
            created_at=100.0,
        )
        d = t.to_dict()
        restored = Triple.from_dict(d)
        assert restored.subject == "Alice"
        assert restored.relation == "works_at"
        assert restored.object == "Acme Corp"
        assert restored.confidence == 0.9
        assert restored.source_turn == 5
        assert restored.created_at == 100.0

    def test_from_dict_defaults(self) -> None:
        t = Triple.from_dict({"subject": "A", "relation": "knows", "object": "B"})
        assert t.confidence == 1.0
        assert t.source_turn == 0


# ======================================================================
# InMemoryTripleStore
# ======================================================================


class TestInMemoryTripleStore:
    def test_add_and_count(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="knows", object="B"))
        assert store.count() == 1

    def test_add_many(self) -> None:
        store = InMemoryTripleStore()
        store.add_many(
            [
                Triple(subject="A", relation="knows", object="B"),
                Triple(subject="C", relation="likes", object="D"),
            ]
        )
        assert store.count() == 2

    def test_all_returns_copies(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="r", object="B"))
        all_triples = store.all()
        assert len(all_triples) == 1
        assert all_triples[0].subject == "A"

    def test_query_finds_by_subject(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="Alice", relation="works_at", object="Acme"))
        store.add(Triple(subject="Bob", relation="works_at", object="Widget Co"))
        results = store.query(["alice"])
        assert len(results) == 1
        assert results[0].subject == "Alice"

    def test_query_finds_by_relation(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="likes", object="B"))
        store.add(Triple(subject="C", relation="dislikes", object="D"))
        results = store.query(["likes"])
        assert len(results) == 2  # "likes" appears in both "likes" and "dislikes"

    def test_query_finds_by_object(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="r", object="Python"))
        results = store.query(["python"])
        assert len(results) == 1

    def test_query_empty_keywords(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="r", object="B"))
        assert store.query([]) == []

    def test_clear(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="r", object="B"))
        store.clear()
        assert store.count() == 0

    def test_pruning(self) -> None:
        store = InMemoryTripleStore(max_triples=3)
        for i in range(5):
            store.add(Triple(subject=f"S{i}", relation="r", object=f"O{i}"))
        assert store.count() == 3
        # Oldest should be pruned
        subjects = {t.subject for t in store.all()}
        assert "S0" not in subjects
        assert "S1" not in subjects
        assert "S4" in subjects

    def test_to_list(self) -> None:
        store = InMemoryTripleStore()
        store.add(Triple(subject="A", relation="r", object="B"))
        lst = store.to_list()
        assert len(lst) == 1
        assert lst[0]["subject"] == "A"


# ======================================================================
# SQLiteTripleStore
# ======================================================================


class TestSQLiteTripleStore:
    def test_add_and_count(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="A", relation="knows", object="B"))
        assert store.count() == 1

    def test_add_many(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add_many(
            [
                Triple(subject="A", relation="knows", object="B"),
                Triple(subject="C", relation="likes", object="D"),
            ]
        )
        assert store.count() == 2

    def test_query_finds_by_keyword(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="Alice", relation="works_at", object="Acme"))
        store.add(Triple(subject="Bob", relation="works_at", object="Widget"))
        results = store.query(["alice"])
        assert len(results) == 1
        assert results[0].subject == "Alice"

    def test_query_case_insensitive(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="Alice", relation="knows", object="Bob"))
        results = store.query(["ALICE"])
        assert len(results) == 1

    def test_query_empty_keywords(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="A", relation="r", object="B"))
        assert store.query([]) == []

    def test_all_returns_ordered(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="First", relation="r", object="B", created_at=1.0))
        store.add(Triple(subject="Second", relation="r", object="B", created_at=2.0))
        all_triples = store.all()
        assert all_triples[0].subject == "First"
        assert all_triples[1].subject == "Second"

    def test_clear(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="A", relation="r", object="B"))
        store.clear()
        assert store.count() == 0

    def test_pruning(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"), max_triples=3)
        for i in range(5):
            store.add(Triple(subject=f"S{i}", relation="r", object=f"O{i}", created_at=float(i)))
        assert store.count() == 3

    def test_to_list(self, tmp_path) -> None:
        store = SQLiteTripleStore(db_path=str(tmp_path / "kg.db"))
        store.add(Triple(subject="A", relation="r", object="B"))
        lst = store.to_list()
        assert len(lst) == 1
        assert lst[0]["subject"] == "A"

    def test_persistence_across_instances(self, tmp_path) -> None:
        db_path = str(tmp_path / "kg.db")
        store1 = SQLiteTripleStore(db_path=db_path)
        store1.add(Triple(subject="A", relation="r", object="B"))
        store2 = SQLiteTripleStore(db_path=db_path)
        assert store2.count() == 1


# ======================================================================
# KnowledgeGraphMemory — extraction
# ======================================================================


class TestKGExtraction:
    def test_extracts_triples_from_messages(self) -> None:
        triples_json = json.dumps(
            [
                {"subject": "Alice", "relation": "works_at", "object": "Acme", "confidence": 0.95},
                {"subject": "Bob", "relation": "knows", "object": "Alice", "confidence": 0.8},
            ]
        )
        provider = FakeExtractionProvider(triples_json)
        kg = KnowledgeGraphMemory(provider=provider)

        msgs = [
            Message(role=Role.USER, content="Alice works at Acme. Bob knows Alice."),
            Message(role=Role.ASSISTANT, content="I see."),
        ]
        triples = kg.extract_triples(msgs)

        assert len(triples) == 2
        assert triples[0].subject == "Alice"
        assert triples[0].relation == "works_at"
        assert triples[0].object == "Acme"
        assert triples[0].confidence == 0.95
        assert triples[1].subject == "Bob"
        assert len(provider.calls) == 1

    def test_empty_messages_returns_empty(self) -> None:
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider)
        assert kg.extract_triples([]) == []
        assert len(provider.calls) == 0

    def test_provider_failure_returns_empty(self) -> None:
        kg = KnowledgeGraphMemory(provider=FailingProvider())
        result = kg.extract_triples([Message(role=Role.USER, content="test")])
        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        provider = FakeExtractionProvider("not json at all")
        kg = KnowledgeGraphMemory(provider=provider)
        result = kg.extract_triples([Message(role=Role.USER, content="test")])
        assert result == []

    def test_respects_relevance_window(self) -> None:
        triples_json = json.dumps([{"subject": "X", "relation": "r", "object": "Y"}])
        provider = FakeExtractionProvider(triples_json)
        kg = KnowledgeGraphMemory(provider=provider, relevance_window=2)

        msgs = [Message(role=Role.USER, content=f"msg-{i}") for i in range(10)]
        kg.extract_triples(msgs)

        call_msgs = provider.calls[0]["messages"]
        content = call_msgs[0].content
        assert "msg-8" in content
        assert "msg-9" in content

    def test_strips_code_fences(self) -> None:
        response = '```json\n[{"subject": "Python", "relation": "is_a", "object": "language"}]\n```'
        provider = FakeExtractionProvider(response)
        kg = KnowledgeGraphMemory(provider=provider)
        result = kg.extract_triples([Message(role=Role.USER, content="I love Python")])
        assert len(result) == 1
        assert result[0].subject == "Python"

    def test_skips_incomplete_triples(self) -> None:
        triples_json = json.dumps(
            [
                {"subject": "A", "relation": "r", "object": "B"},
                {"subject": "C"},  # missing relation and object
                {"relation": "r", "object": "D"},  # missing subject
            ]
        )
        provider = FakeExtractionProvider(triples_json)
        kg = KnowledgeGraphMemory(provider=provider)
        result = kg.extract_triples([Message(role=Role.USER, content="test")])
        assert len(result) == 1
        assert result[0].subject == "A"


# ======================================================================
# KnowledgeGraphMemory — query_relevant
# ======================================================================


class TestKGQueryRelevant:
    def test_returns_matching_triples(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add_many(
            [
                Triple(subject="Alice", relation="works_at", object="Acme"),
                Triple(subject="Bob", relation="lives_in", object="NYC"),
            ]
        )
        results = kg.query_relevant("Tell me about Alice")
        assert len(results) == 1
        assert results[0].subject == "Alice"

    def test_respects_max_context_triples(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider(), max_context_triples=2)
        kg.store.add_many(
            [Triple(subject=f"S{i}", relation="r", object=f"O{i}") for i in range(10)]
        )
        results = kg.query_relevant("S0 S1 S2 S3 S4")
        assert len(results) <= 2

    def test_empty_query_returns_empty(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add(Triple(subject="A", relation="r", object="B"))
        assert kg.query_relevant("") == []

    def test_short_words_filtered(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add(Triple(subject="A", relation="is", object="B"))
        # "is" and "a" are 2 chars or less, filtered out
        assert kg.query_relevant("is a") == []


# ======================================================================
# KnowledgeGraphMemory — build_context
# ======================================================================


class TestKGBuildContext:
    def test_empty_returns_empty_string(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        assert kg.build_context() == ""

    def test_formats_triples_without_query(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add_many(
            [
                Triple(subject="Alice", relation="works_at", object="Acme"),
                Triple(subject="Bob", relation="knows", object="Alice", confidence=0.8),
            ]
        )
        ctx = kg.build_context()
        assert "[Known Relationships]" in ctx
        assert "Alice --[works_at]--> Acme" in ctx
        assert "Bob --[knows]--> Alice (confidence: 0.8)" in ctx

    def test_formats_triples_with_query(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add_many(
            [
                Triple(subject="Alice", relation="works_at", object="Acme"),
                Triple(subject="Bob", relation="lives_in", object="NYC"),
            ]
        )
        ctx = kg.build_context(query="Where does Alice work?")
        assert "[Known Relationships]" in ctx
        assert "Alice" in ctx

    def test_respects_max_context_triples(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider(), max_context_triples=2)
        kg.store.add_many(
            [Triple(subject=f"S{i}", relation="r", object=f"O{i}") for i in range(10)]
        )
        ctx = kg.build_context()
        lines = [line for line in ctx.split("\n") if line.startswith("- ")]
        assert len(lines) == 2

    def test_full_confidence_no_suffix(self) -> None:
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add(Triple(subject="A", relation="r", object="B", confidence=1.0))
        ctx = kg.build_context()
        assert "confidence" not in ctx


# ======================================================================
# KnowledgeGraphMemory — serialization
# ======================================================================


class TestKGSerialization:
    def test_round_trip(self) -> None:
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider, max_context_triples=10, relevance_window=5)
        kg.store.add(Triple(subject="A", relation="r", object="B", confidence=0.9))

        d = kg.to_dict()
        restored = KnowledgeGraphMemory.from_dict(d, provider)
        assert restored.store.count() == 1
        triples = restored.store.all()
        assert triples[0].subject == "A"
        assert triples[0].confidence == 0.9
        assert restored._max_context_triples == 10
        assert restored._relevance_window == 5

    def test_empty_round_trip(self) -> None:
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider)
        d = kg.to_dict()
        restored = KnowledgeGraphMemory.from_dict(d, provider)
        assert restored.store.count() == 0

    def test_max_triples_preserved_in_round_trip(self) -> None:
        """Regression: max_triples must survive to_dict/from_dict round-trip.

        Before the fix, to_dict() did not include max_triples, so from_dict()
        always created an InMemoryTripleStore with default max_triples=200,
        ignoring the user's custom limit.  This caused data loss when
        len(stored_triples) > 200 on restore, and the pruning limit was wrong
        for stores created with max_triples < 200.
        """
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider, max_triples=50)
        for i in range(30):
            kg.store.add(Triple(subject=f"S{i}", relation="r", object=f"O{i}"))

        d = kg.to_dict()
        assert "max_triples" in d
        assert d["max_triples"] == 50

        restored = KnowledgeGraphMemory.from_dict(d, provider)
        # The restored store must have the correct max_triples limit
        assert restored._max_triples == 50
        # All 30 triples should survive (under the 50 limit)
        assert restored.store.count() == 30

    def test_custom_max_triples_enforced_after_round_trip(self) -> None:
        """Regression: pruning limit is correct after from_dict().

        After restoring with max_triples=5, adding a 6th triple must prune
        to 5, not to the old default of 200.
        """
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider, max_triples=5)
        for i in range(4):
            kg.store.add(Triple(subject=f"S{i}", relation="r", object=f"O{i}"))

        d = kg.to_dict()
        restored = KnowledgeGraphMemory.from_dict(d, provider)
        # Add one more to bring to the limit, then one more to trigger pruning
        restored.store.add(Triple(subject="S4", relation="r", object="O4"))
        restored.store.add(Triple(subject="S5", relation="r", object="O5"))
        # Should be pruned back to 5
        assert restored.store.count() == 5

    def test_model_preserved_in_round_trip(self) -> None:
        """Regression: model must survive to_dict/from_dict round-trip.

        Before the fix, to_dict() did not include model, so from_dict()
        always created a KnowledgeGraphMemory with model=None, ignoring
        the user's custom extraction model.
        """
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider, model="gpt-4o")
        d = kg.to_dict()
        assert "model" in d
        assert d["model"] == "gpt-4o"

        restored = KnowledgeGraphMemory.from_dict(d, provider)
        assert restored._model == "gpt-4o"

    def test_model_none_preserved_in_round_trip(self) -> None:
        """Regression: model=None round-trips correctly."""
        provider = FakeExtractionProvider()
        kg = KnowledgeGraphMemory(provider=provider, model=None)
        d = kg.to_dict()
        restored = KnowledgeGraphMemory.from_dict(d, provider)
        assert restored._model is None


# ======================================================================
# Agent integration
# ======================================================================


class TestKGAgentIntegration:
    def _make_tool(self):
        from selectools.tools import Tool

        return Tool(name="echo", description="Echo", parameters=[], function=lambda: "ok")

    def test_kg_context_injected(self) -> None:
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
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add(Triple(subject="Alice", relation="works_at", object="Acme"))

        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_graph=kg),
        )
        agent.run("Tell me about Alice")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        assert any("[Known Relationships]" in m.content for m in system_msgs)

    def test_no_kg_no_injection(self) -> None:
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
        assert not any("[Known Relationships]" in m.content for m in system_msgs)

    def test_extraction_failure_doesnt_crash(self) -> None:
        from selectools.agent import Agent, AgentConfig
        from selectools.memory import ConversationMemory

        kg = KnowledgeGraphMemory(provider=FailingProvider())

        class SimpleProvider:
            name = "simple"
            supports_streaming = False
            supports_async = False

            def complete(self, **kw):
                return Message(role=Role.ASSISTANT, content="ok"), _usage()

        agent = Agent(
            tools=[self._make_tool()],
            provider=SimpleProvider(),
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_graph=kg),
        )
        result = agent.run("Hello")
        assert result.content == "ok"

    def test_kg_query_uses_user_message(self) -> None:
        """KG context injection should use user's message as query."""
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
        kg = KnowledgeGraphMemory(provider=FakeExtractionProvider())
        kg.store.add(Triple(subject="Alice", relation="works_at", object="Acme"))
        kg.store.add(Triple(subject="Bob", relation="lives_in", object="NYC"))

        agent = Agent(
            tools=[self._make_tool()],
            provider=recording,
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_graph=kg),
        )
        # Query about Alice - should get Alice-related triples
        agent.run("Tell me about Alice and her work")

        system_msgs = [m for m in recording.last_messages if m.role == Role.SYSTEM]
        kg_msgs = [m for m in system_msgs if "[Known Relationships]" in m.content]
        assert len(kg_msgs) == 1
        assert "Alice" in kg_msgs[0].content


# ======================================================================
# Regression tests
# ======================================================================


class TestKGExtractionRegressions:
    """Regression tests for extract_triples() edge cases."""

    def test_null_confidence_does_not_discard_all_triples(self) -> None:
        """Regression: float(None) raised TypeError which was caught by the outer
        except-all, silently discarding ALL triples whenever a single item had
        confidence=null in the LLM response.

        The fix wraps the float() cast per-item with a try/except that falls
        back to confidence=1.0 for None or non-numeric values, so other triples
        in the same response are preserved.
        """
        triples_json = json.dumps(
            [
                # One triple with null confidence
                {"subject": "Alice", "relation": "works_at", "object": "Acme", "confidence": None},
                # One with a valid confidence
                {"subject": "Bob", "relation": "knows", "object": "Alice", "confidence": 0.9},
                # One with a non-numeric string confidence
                {
                    "subject": "Acme",
                    "relation": "is_a",
                    "object": "company",
                    "confidence": "high",
                },
            ]
        )
        provider = FakeExtractionProvider(triples_json)
        kg = KnowledgeGraphMemory(provider=provider)
        result = kg.extract_triples([Message(role=Role.USER, content="context")])

        # All three triples must be extracted — null/invalid confidence must not
        # abort the entire extraction.
        assert len(result) == 3

        by_subject = {t.subject: t for t in result}
        # null confidence falls back to 1.0
        assert by_subject["Alice"].confidence == 1.0
        # valid float confidence is preserved
        assert by_subject["Bob"].confidence == 0.9
        # non-numeric string falls back to 1.0
        assert by_subject["Acme"].confidence == 1.0

    def test_missing_confidence_key_defaults_to_one(self) -> None:
        """Regression: triple without 'confidence' key must default to 1.0."""
        triples_json = json.dumps(
            [{"subject": "A", "relation": "r", "object": "B"}]  # no confidence key
        )
        provider = FakeExtractionProvider(triples_json)
        kg = KnowledgeGraphMemory(provider=provider)
        result = kg.extract_triples([Message(role=Role.USER, content="test")])
        assert len(result) == 1
        assert result[0].confidence == 1.0
