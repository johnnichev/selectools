"""Tests for UnifiedMemory — tiered memory with auto-promotion."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, List, Tuple

import pytest

from selectools.entity_memory import Entity, EntityMemory
from selectools.knowledge import KnowledgeMemory
from selectools.types import Message, Role
from selectools.unified_memory import (
    DEFAULT_IMPORTANCE_RULES,
    EpisodicMemory,
    ImportanceRule,
    InMemoryKnowledgeStore,
    UnifiedMemory,
    score_importance,
)


def make_memory(tmp_path: Any, **kwargs: Any) -> UnifiedMemory:
    long_term = kwargs.pop("long_term", None)
    if long_term is None:
        long_term = KnowledgeMemory(
            directory=str(tmp_path / "lt"),
            store=InMemoryKnowledgeStore(),
            max_entries=kwargs.pop("long_term_limit", 1000),
        )
    return UnifiedMemory(long_term=long_term, **kwargs)


def lt_contents(memory: UnifiedMemory) -> List[str]:
    return [e.content for e in memory.long_term.store.query(limit=1000)]


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestConstruction:
    def test_roadmap_config_shape(self) -> None:
        memory = UnifiedMemory(
            importance_threshold=0.7,
            short_term_limit=100,
            long_term_limit=1000,
            episodic_retention_days=30,
            auto_promote=True,
        )
        assert memory.short_term.max_messages == 100

    def test_zero_arg_default(self) -> None:
        memory = UnifiedMemory()
        memory.add_turn("hello", "hi there")
        assert len(memory.short_term) == 2
        assert len(memory.episodic) == 1

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            UnifiedMemory(importance_threshold=1.5)
        with pytest.raises(ValueError):
            UnifiedMemory(short_term_limit=0)
        with pytest.raises(ValueError):
            UnifiedMemory(episodic_retention_days=0)
        with pytest.raises(ValueError):
            UnifiedMemory(compaction_threshold=0.0)

    def test_dependency_injection(self, tmp_path: Any) -> None:
        from selectools.memory import ConversationMemory

        stm = ConversationMemory(max_messages=4)
        lt = KnowledgeMemory(directory=str(tmp_path / "lt"), store=InMemoryKnowledgeStore())
        epi = EpisodicMemory()
        memory = UnifiedMemory(short_term=stm, long_term=lt, episodic=epi)
        assert memory.short_term is stm
        assert memory.long_term is lt
        assert memory.episodic is epi


# ----------------------------------------------------------------------
# Importance scoring
# ----------------------------------------------------------------------


class TestImportanceScoring:
    def test_name_scores_high(self) -> None:
        assert score_importance("My name is John Niche") == pytest.approx(0.9)

    def test_preference_scores_mid(self) -> None:
        assert score_importance("I prefer dark roast coffee") == pytest.approx(0.75)

    def test_location_scores_lower(self) -> None:
        assert score_importance("I live in Brusque, Brazil") == pytest.approx(0.6)

    def test_mundane_text_scores_base(self) -> None:
        assert score_importance("the weather is mild today") < 0.5

    def test_case_insensitive(self) -> None:
        assert score_importance("MY NAME IS ALICE") == pytest.approx(0.9)

    def test_custom_rule_table_overrides_defaults(self) -> None:
        rules = [ImportanceRule(name="project", pattern=r"selectools", score=0.95)]
        assert score_importance("selectools ships today", rules=rules) == pytest.approx(0.95)
        # Custom table replaces defaults: name rule no longer applies.
        assert score_importance("My name is John", rules=rules) < 0.5

    def test_default_rules_are_documented(self) -> None:
        for rule in DEFAULT_IMPORTANCE_RULES:
            assert rule.description, f"rule {rule.name} missing description"


# ----------------------------------------------------------------------
# Promotion lifecycle
# ----------------------------------------------------------------------


class TestPromotion:
    def test_aged_out_item_at_threshold_promoted(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=2, importance_threshold=0.9)
        memory.add_turn("My name is John Niche", "Nice to meet you")
        memory.add_turn("ok", "ok")  # ages out the first turn
        contents = lt_contents(memory)
        assert "My name is John Niche" in contents

    def test_aged_out_item_below_threshold_not_promoted(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=2, importance_threshold=0.7)
        memory.add_turn("I live in Brusque", "Cool")  # location = 0.6 < 0.7
        memory.add_turn("ok", "ok")
        assert lt_contents(memory) == []

    def test_auto_promote_false_requires_consolidate(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=2, auto_promote=False)
        memory.add_turn("My name is John Niche", "Hello")
        memory.add_turn("ok", "ok")
        assert lt_contents(memory) == []
        promoted = memory.consolidate()
        # Only items still in STM are considered by consolidate().
        assert promoted == 0
        memory.add_turn("I prefer tabs over spaces", "Noted")
        promoted = memory.consolidate()
        assert promoted == 1
        assert "I prefer tabs over spaces" in lt_contents(memory)

    def test_consolidate_promotes_items_still_in_stm(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=100)
        memory.add_turn("My name is John Niche", "Hi John")
        assert lt_contents(memory) == []
        promoted = memory.consolidate()
        assert promoted == 1
        assert "My name is John Niche" in lt_contents(memory)

    def test_promotion_is_idempotent(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=100)
        memory.add_turn("My name is John Niche", "Hi John")
        assert memory.consolidate() == 1
        assert memory.consolidate() == 0
        assert lt_contents(memory).count("My name is John Niche") == 1

    def test_no_duplicate_promotion_on_age_out_after_consolidate(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=2)
        memory.add_turn("My name is John Niche", "Hi John")
        memory.consolidate()
        memory.add_turn("ok", "ok")  # ages out the already-promoted turn
        assert lt_contents(memory).count("My name is John Niche") == 1

    def test_promotion_carries_category_and_importance(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=100)
        memory.add_turn("I prefer dark roast coffee", "Noted")
        memory.consolidate()
        entries = memory.long_term.store.query(limit=10)
        assert len(entries) == 1
        assert entries[0].category == "preference"
        assert entries[0].importance == pytest.approx(0.75)

    def test_custom_scorer_hook_invoked(self, tmp_path: Any) -> None:
        calls: List[str] = []

        def scorer(text: str) -> float:
            calls.append(text)
            return 0.99

        memory = make_memory(tmp_path, short_term_limit=100, scorer=scorer)
        memory.add_turn("totally mundane text", "ok")
        promoted = memory.consolidate()
        assert promoted == 2  # scorer promotes everything
        assert calls

    def test_scorer_failure_falls_back_to_rules(self, tmp_path: Any) -> None:
        def scorer(text: str) -> float:
            raise RuntimeError("LLM down")

        memory = make_memory(tmp_path, short_term_limit=100, scorer=scorer)
        memory.add_turn("My name is John Niche", "ok")
        assert memory.consolidate() == 1


# ----------------------------------------------------------------------
# Episodic memory
# ----------------------------------------------------------------------


class TestEpisodicMemory:
    def test_retention_pruning(self) -> None:
        epi = EpisodicMemory()
        old = datetime.now(timezone.utc) - timedelta(days=40)
        epi.add("old question", "old answer", when=old)
        epi.add("new question", "new answer")
        assert len(epi) == 2
        removed = epi.prune(30)
        assert removed == 1
        assert len(epi) == 1
        assert epi.recent(days=7)[0].user == "new question"

    def test_add_turn_prunes_old_episodes(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, episodic_retention_days=30)
        old = datetime.now(timezone.utc) - timedelta(days=40)
        memory.episodic.add("ancient", "history", when=old)
        memory.add_turn("hello", "hi")
        assert all(e.user != "ancient" for e in memory.episodic.recent(days=365))

    def test_json_serializable_roundtrip(self) -> None:
        epi = EpisodicMemory()
        epi.add("question", "answer")
        payload = json.dumps(epi.to_dict())
        restored = EpisodicMemory.from_dict(json.loads(payload))
        assert len(restored) == 1
        episode = restored.recent(days=7)[0]
        assert episode.user == "question"
        assert episode.assistant == "answer"
        assert episode.timestamp.tzinfo is not None


# ----------------------------------------------------------------------
# Context assembly + compaction
# ----------------------------------------------------------------------


class TestContextAssembly:
    def test_no_compaction_under_budget(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, token_counter=lambda t: len(t))
        memory.add_turn("hello", "hi there")
        context = memory.assemble_context(max_tokens=100_000)
        assert "hello" in context
        assert "compacted" not in context
        assert "truncated" not in context

    def test_compaction_triggers_at_token_boundary(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, token_counter=lambda t: len(t))
        for i in range(10):
            memory.add_turn(f"question number {i} with some padding text", f"answer number {i}")
        full = memory.assemble_context(max_tokens=100_000)
        # Set the budget just below the full size: 70% of max_tokens must trip.
        max_tokens = int(len(full) / 0.7) - 10
        context = memory.assemble_context(max_tokens=max_tokens)
        assert "compacted" in context or "truncated" in context
        # The most recent turn must survive compaction.
        assert "answer number 9" in context

    def test_compaction_respects_budget_for_small_limits(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, token_counter=lambda t: len(t))
        for i in range(20):
            memory.add_turn(f"q{i} " + "x" * 50, f"a{i} " + "y" * 50)
        context = memory.assemble_context(max_tokens=400)
        assert len(context) <= 400

    def test_summarizer_hook_used_for_compaction(self, tmp_path: Any) -> None:
        calls: List[str] = []

        def summarizer(text: str) -> str:
            calls.append(text)
            return "SUMMARY_MARKER"

        memory = make_memory(tmp_path, token_counter=lambda t: len(t), summarizer=summarizer)
        for i in range(10):
            memory.add_turn(f"question number {i} with some padding text", f"answer number {i}")
        full = memory.assemble_context(max_tokens=100_000)
        context = memory.assemble_context(max_tokens=int(len(full) / 0.7) - 10)
        assert calls
        assert "SUMMARY_MARKER" in context

    def test_assemble_includes_all_tiers(self, tmp_path: Any) -> None:
        em = EntityMemory(provider=None)
        em.update([Entity(name="Alice", entity_type="person", attributes={"role": "daughter"})])
        memory = make_memory(tmp_path, entity_memory=em)
        memory.long_term.remember("User's daughter is Alice", importance=0.9)
        memory.add_turn("hello", "hi")
        context = memory.assemble_context(max_tokens=100_000)
        assert "User's daughter is Alice" in context
        assert "Alice [person]" in context
        assert "[Recent Episodes]" in context
        assert "[Conversation]" in context

    def test_invalid_max_tokens(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        with pytest.raises(ValueError):
            memory.assemble_context(max_tokens=0)


# ----------------------------------------------------------------------
# Recall federation
# ----------------------------------------------------------------------


class TestRecall:
    def test_federated_ordering(self, tmp_path: Any) -> None:
        em = EntityMemory(provider=None)
        em.update([Entity(name="Alice", entity_type="person", attributes={"relation": "daughter"})])
        memory = make_memory(tmp_path, entity_memory=em)
        memory.long_term.remember(
            "User's daughter is named Alice", category="relationship", importance=0.9
        )
        memory.add_turn("We discussed Alice's school today", "Sounds good")
        results = memory.recall("Alice daughter name")
        sources = [r.source for r in results]
        assert sources[0] == "long_term"
        assert "entity" in sources
        assert "episodic" in sources
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_no_match_returns_empty(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        memory.add_turn("hello", "hi")
        assert memory.recall("quantum chromodynamics") == []

    def test_recall_limit(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        for i in range(10):
            memory.long_term.remember(f"fact about cats number {i}", importance=0.8)
        results = memory.recall("cats", limit=3)
        assert len(results) == 3

    def test_recall_episodic_date_filter(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        old = datetime.now(timezone.utc) - timedelta(days=10)
        memory.episodic.add("old cats discussion", "meow", when=old)
        memory.episodic.add("new cats discussion", "purr")
        recent_only = memory.recall("cats", days=2)
        assert all("old cats" not in r.content for r in recent_only)
        both = memory.recall("cats", days=30)
        assert any("old cats" in r.content for r in both)


# ----------------------------------------------------------------------
# Entity tier fed per turn
# ----------------------------------------------------------------------


class _EntityStubProvider:
    """Stub provider returning a canned entity-extraction JSON payload."""

    name = "stub"

    def complete(self, **kwargs: Any) -> Tuple[Message, None]:
        payload = json.dumps(
            [{"name": "Alice", "entity_type": "person", "attributes": {"relation": "daughter"}}]
        )
        return Message(role=Role.ASSISTANT, content=payload), None


class TestEntityTier:
    def test_entities_fed_per_turn(self, tmp_path: Any) -> None:
        em = EntityMemory(provider=_EntityStubProvider())
        memory = make_memory(tmp_path, entity_memory=em)
        memory.add_turn("My daughter Alice starts school tomorrow", "Exciting!")
        names = [e.name for e in em.entities]
        assert "Alice" in names

    def test_no_entity_memory_is_fine(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        memory.add_turn("My daughter Alice starts school tomorrow", "Exciting!")
        assert memory.entity_memory is None


# ----------------------------------------------------------------------
# Thread safety
# ----------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_turn_smoke(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path, short_term_limit=20)
        errors: List[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(20):
                    memory.add_turn(
                        f"My name is Worker{worker_id} turn {i}", f"hello {worker_id}-{i}"
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(memory.short_term) <= 20
        memory.assemble_context(max_tokens=4000)
        memory.recall("Worker1")


# ----------------------------------------------------------------------
# clear()
# ----------------------------------------------------------------------


class TestClear:
    def test_clear_resets_short_term_and_episodic(self, tmp_path: Any) -> None:
        memory = make_memory(tmp_path)
        memory.add_turn("My name is John Niche", "Hi")
        memory.consolidate()
        memory.clear()
        assert len(memory.short_term) == 0
        assert len(memory.episodic) == 0
        # Long-term survives clear() by default.
        assert "My name is John Niche" in lt_contents(memory)
        # Promotion dedup state is reset: same content can be re-learned
        # but the store still holds only one copy (same content, new entry).
        memory.add_turn("My name is John Niche", "Hi")
        assert memory.consolidate() == 1
