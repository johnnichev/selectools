"""Tests for knowledge memory enhancement (R3) — stores, entries, eviction."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from selectools.knowledge import (
    FileKnowledgeStore,
    KnowledgeEntry,
    KnowledgeMemory,
    SQLiteKnowledgeStore,
)


class TestKnowledgeEntry:
    """KnowledgeEntry dataclass and TTL logic."""

    def test_default_fields(self):
        e = KnowledgeEntry(content="hello")
        assert e.content == "hello"
        assert e.category == "general"
        assert e.importance == 0.5
        assert not e.persistent
        assert e.ttl_days is None
        assert not e.is_expired

    def test_expired_entry(self):
        e = KnowledgeEntry(
            content="old",
            ttl_days=1,
            created_at=datetime.utcnow() - timedelta(days=2),
        )
        assert e.is_expired

    def test_not_expired_entry(self):
        e = KnowledgeEntry(content="fresh", ttl_days=7)
        assert not e.is_expired

    def test_no_ttl_never_expires(self):
        e = KnowledgeEntry(
            content="eternal",
            ttl_days=None,
            created_at=datetime.utcnow() - timedelta(days=365),
        )
        assert not e.is_expired


class TestFileKnowledgeStore:
    """FileKnowledgeStore CRUD and query operations."""

    @pytest.fixture
    def store(self, tmp_path):
        return FileKnowledgeStore(directory=str(tmp_path / "knowledge"))

    def test_save_and_get(self, store):
        entry = KnowledgeEntry(content="test fact", importance=0.8)
        entry_id = store.save(entry)
        retrieved = store.get(entry_id)
        assert retrieved is not None
        assert retrieved.content == "test fact"
        assert retrieved.importance == 0.8

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_delete(self, store):
        entry = KnowledgeEntry(content="to delete")
        store.save(entry)
        assert store.delete(entry.id)
        assert store.get(entry.id) is None

    def test_delete_nonexistent(self, store):
        assert not store.delete("nonexistent")

    def test_count(self, store):
        assert store.count() == 0
        store.save(KnowledgeEntry(content="a"))
        store.save(KnowledgeEntry(content="b"))
        assert store.count() == 2

    def test_query_by_category(self, store):
        store.save(KnowledgeEntry(content="a", category="fact"))
        store.save(KnowledgeEntry(content="b", category="preference"))
        store.save(KnowledgeEntry(content="c", category="fact"))
        results = store.query(category="fact")
        assert len(results) == 2
        assert all(e.category == "fact" for e in results)

    def test_query_by_min_importance(self, store):
        store.save(KnowledgeEntry(content="low", importance=0.2))
        store.save(KnowledgeEntry(content="high", importance=0.9))
        results = store.query(min_importance=0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    def test_query_filters_expired(self, store):
        store.save(
            KnowledgeEntry(
                content="expired",
                ttl_days=1,
                created_at=datetime.utcnow() - timedelta(days=2),
            )
        )
        store.save(KnowledgeEntry(content="fresh"))
        results = store.query()
        assert len(results) == 1
        assert results[0].content == "fresh"

    def test_query_ordered_by_importance(self, store):
        store.save(KnowledgeEntry(content="low", importance=0.2))
        store.save(KnowledgeEntry(content="mid", importance=0.5))
        store.save(KnowledgeEntry(content="high", importance=0.9))
        results = store.query()
        assert results[0].content == "high"
        assert results[-1].content == "low"

    def test_prune_expired(self, store):
        store.save(
            KnowledgeEntry(
                content="old",
                ttl_days=1,
                created_at=datetime.utcnow() - timedelta(days=2),
            )
        )
        store.save(KnowledgeEntry(content="fresh"))
        removed = store.prune()
        assert removed == 1
        assert store.count() == 1

    def test_prune_persistent_survives(self, store):
        store.save(
            KnowledgeEntry(
                content="persistent",
                persistent=True,
                ttl_days=1,
                created_at=datetime.utcnow() - timedelta(days=2),
            )
        )
        removed = store.prune()
        assert removed == 0
        assert store.count() == 1

    def test_update_existing(self, store):
        entry = KnowledgeEntry(content="original")
        store.save(entry)
        entry.content = "updated"
        store.save(entry)
        assert store.count() == 1
        assert store.get(entry.id).content == "updated"


class TestSQLiteKnowledgeStore:
    """SQLiteKnowledgeStore CRUD and query operations."""

    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteKnowledgeStore(db_path=str(tmp_path / "test.db"))

    def test_save_and_get(self, store):
        entry = KnowledgeEntry(content="sqlite fact", importance=0.7)
        store.save(entry)
        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "sqlite fact"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_delete(self, store):
        entry = KnowledgeEntry(content="to delete")
        store.save(entry)
        assert store.delete(entry.id)
        assert store.get(entry.id) is None

    def test_count(self, store):
        store.save(KnowledgeEntry(content="a"))
        store.save(KnowledgeEntry(content="b"))
        assert store.count() == 2

    def test_query_by_category(self, store):
        store.save(KnowledgeEntry(content="a", category="fact"))
        store.save(KnowledgeEntry(content="b", category="pref"))
        results = store.query(category="fact")
        assert len(results) == 1

    def test_query_by_min_importance(self, store):
        store.save(KnowledgeEntry(content="low", importance=0.1))
        store.save(KnowledgeEntry(content="high", importance=0.9))
        results = store.query(min_importance=0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    def test_query_filters_expired(self, store):
        store.save(
            KnowledgeEntry(
                content="expired",
                ttl_days=1,
                created_at=datetime.utcnow() - timedelta(days=2),
            )
        )
        store.save(KnowledgeEntry(content="fresh"))
        results = store.query()
        assert len(results) == 1

    def test_prune_expired(self, store):
        store.save(
            KnowledgeEntry(
                content="old",
                ttl_days=1,
                created_at=datetime.utcnow() - timedelta(days=2),
            )
        )
        store.save(KnowledgeEntry(content="fresh"))
        removed = store.prune()
        assert removed == 1


class TestKnowledgeMemoryEnhanced:
    """KnowledgeMemory with new store-based features."""

    @pytest.fixture
    def memory(self, tmp_path):
        return KnowledgeMemory(directory=str(tmp_path / "mem"))

    def test_backward_compat_directory_only(self, tmp_path):
        """Original constructor API still works."""
        mem = KnowledgeMemory(directory=str(tmp_path / "legacy"))
        result = mem.remember("hello")
        assert result  # returns entry ID

    def test_remember_with_importance(self, memory):
        entry_id = memory.remember("important fact", importance=0.9)
        assert entry_id
        entry = memory.store.get(entry_id)
        assert entry.importance == 0.9

    def test_remember_with_ttl(self, memory):
        entry_id = memory.remember("temporary", ttl_days=7)
        entry = memory.store.get(entry_id)
        assert entry.ttl_days == 7

    def test_remember_with_category(self, memory):
        entry_id = memory.remember("user likes dark mode", category="preference")
        entry = memory.store.get(entry_id)
        assert entry.category == "preference"

    def test_max_entries_eviction(self, tmp_path):
        mem = KnowledgeMemory(directory=str(tmp_path / "evict"), max_entries=3)
        mem.remember("low1", importance=0.1)
        mem.remember("low2", importance=0.2)
        mem.remember("high", importance=0.9)
        mem.remember("new", importance=0.5)
        # Should have evicted lowest importance
        assert mem.store.count() <= 3

    def test_persistent_survives_eviction(self, tmp_path):
        mem = KnowledgeMemory(directory=str(tmp_path / "persist"), max_entries=2)
        mem.remember("persistent", persistent=True, importance=0.1)
        mem.remember("normal1", importance=0.5)
        mem.remember("normal2", importance=0.6)
        # Persistent entry should survive even with low importance
        entries = mem.store.query(limit=10)
        contents = {e.content for e in entries}
        assert "persistent" in contents

    def test_build_context_uses_store(self, memory):
        memory.remember("critical fact", importance=0.9, category="fact")
        memory.remember("minor detail", importance=0.3)
        ctx = memory.build_context()
        assert "critical fact" in ctx

    def test_build_context_truncation(self, tmp_path):
        mem = KnowledgeMemory(directory=str(tmp_path / "trunc"), max_context_chars=50)
        mem.remember("x" * 100, importance=0.9)
        ctx = mem.build_context()
        assert len(ctx) <= 50

    def test_with_sqlite_store(self, tmp_path):
        """KnowledgeMemory works with SQLiteKnowledgeStore."""
        store = SQLiteKnowledgeStore(db_path=str(tmp_path / "km.db"))
        mem = KnowledgeMemory(directory=str(tmp_path / "km"), store=store)
        entry_id = mem.remember("sqlite entry", importance=0.8)
        assert store.get(entry_id).content == "sqlite entry"

    def test_legacy_logs_still_written(self, memory):
        """Legacy .log files are still written for backward compat."""
        memory.remember("test entry")
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(memory.directory, f"{today}.log")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            assert "test entry" in f.read()

    def test_prune_old_logs(self, memory):
        """Legacy prune_old_logs still works."""
        removed = memory.prune_old_logs(keep_days=0)
        assert isinstance(removed, int)
