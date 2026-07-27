"""Tests for knowledge memory enhancement (R3) — stores, entries, eviction."""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from selectools.knowledge import (
    FileKnowledgeStore,
    KnowledgeEntry,
    KnowledgeMemory,
    SQLiteKnowledgeStore,
    entry_from_jsonable,
    entry_to_jsonable,
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
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        assert e.is_expired

    def test_not_expired_entry(self):
        e = KnowledgeEntry(content="fresh", ttl_days=7)
        assert not e.is_expired

    def test_no_ttl_never_expires(self):
        e = KnowledgeEntry(
            content="eternal",
            ttl_days=None,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        assert not e.is_expired

    def test_is_expired_with_naive_datetime_does_not_crash(self):
        """Regression: is_expired must not raise TypeError when created_at is naive.

        KnowledgeEntry accepts datetime for created_at.  If a caller passes a
        naive datetime (no tzinfo) and ttl_days is set, the internal comparison
        ``datetime.now(timezone.utc) > naive_dt + timedelta(...)`` would raise
        TypeError.  The property now normalises naive datetimes to UTC before
        comparing.
        """
        naive_old = datetime(2020, 1, 1)  # no tzinfo
        e = KnowledgeEntry(
            content="legacy entry",
            ttl_days=1,
            created_at=naive_old,
        )
        # Must not raise TypeError
        assert e.is_expired is True

    def test_is_expired_naive_fresh_does_not_crash(self):
        """Regression: is_expired with naive future datetime should return False."""
        from datetime import timedelta

        naive_future = datetime.now() + timedelta(days=365)  # naive, in the future
        e = KnowledgeEntry(
            content="fresh",
            ttl_days=999,
            created_at=naive_future,
        )
        assert e.is_expired is False


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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
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
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
        )
        store.save(KnowledgeEntry(content="fresh"))
        removed = store.prune()
        assert removed == 1


class TestPruneMaxAgeDaysZero:
    """Regression: prune(max_age_days=0) treated 0 as falsy, skipping age-based pruning."""

    def test_file_store_prune_max_age_zero_removes_all_non_persistent(self, tmp_path):
        """Regression: FileKnowledgeStore.prune(max_age_days=0) used ``if max_age_days``
        instead of ``if max_age_days is not None``, so 0 was treated as no age limit.

        With max_age_days=0, all non-persistent entries should be pruned
        (cutoff = now - 0 days = now, everything is older than now).
        """
        store = FileKnowledgeStore(directory=str(tmp_path / "prune0"))
        store.save(
            KnowledgeEntry(
                content="yesterday",
                created_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        )
        removed = store.prune(max_age_days=0)
        assert removed == 1, "prune(max_age_days=0) must remove entries older than now"

    def test_sqlite_store_prune_max_age_zero_removes_all_non_persistent(self, tmp_path):
        """Regression: same 0-is-falsy bug in SQLiteKnowledgeStore."""
        store = SQLiteKnowledgeStore(db_path=str(tmp_path / "prune0.db"))
        store.save(
            KnowledgeEntry(
                content="yesterday",
                created_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        )
        removed = store.prune(max_age_days=0)
        assert removed == 1, "prune(max_age_days=0) must remove entries older than now"


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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = os.path.join(memory.directory, f"{today}.log")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            assert "test entry" in f.read()

    def test_prune_old_logs(self, memory):
        """Legacy prune_old_logs still works."""
        removed = memory.prune_old_logs(keep_days=0)
        assert isinstance(removed, int)


# ======================================================================
# Regression tests for timezone-naive datetime bugs
# ======================================================================


class TestNaiveDatetimeRegressions:
    """Regression tests: naive datetime comparisons must not raise TypeError."""

    def test_file_store_query_with_naive_created_at(self, tmp_path):
        """Regression: FileKnowledgeStore._load_all() must normalize naive datetimes.

        Writing a naive ISO string (no timezone) to the JSONL file and then
        calling query() with a timezone-aware 'since' filter must not raise
        TypeError: can't compare offset-naive and offset-aware datetimes.
        """
        import json as json_mod

        store = FileKnowledgeStore(directory=str(tmp_path / "fks"))
        # Write a JSONL entry manually with a naive ISO timestamp
        entries_path = store._entries_path
        os.makedirs(os.path.dirname(entries_path), exist_ok=True)
        naive_entry = {
            "id": "naive001",
            "content": "old fact",
            "category": "general",
            "importance": 0.5,
            "persistent": False,
            "ttl_days": None,
            "created_at": "2020-01-01T00:00:00",  # naive (no +00:00)
            "updated_at": "2020-01-01T00:00:00",  # naive
            "metadata": {},
        }
        with open(entries_path, "w", encoding="utf-8") as f:
            f.write(json_mod.dumps(naive_entry) + "\n")

        # query() with since= must not raise TypeError
        since = datetime.now(timezone.utc) - timedelta(days=30)
        results = store.query(since=since)
        # The naive entry (year 2020) is older than 30 days, so not returned
        assert all(e.created_at >= since for e in results)

    def test_file_store_is_expired_naive(self, tmp_path):
        """Regression: is_expired must work for entries loaded with naive datetimes."""
        import json as json_mod

        store = FileKnowledgeStore(directory=str(tmp_path / "fks2"))
        entries_path = store._entries_path
        os.makedirs(os.path.dirname(entries_path), exist_ok=True)
        # Entry with ttl_days=1 but created 2 years ago (naive)
        expired_entry = {
            "id": "naive002",
            "content": "expired fact",
            "category": "general",
            "importance": 0.5,
            "persistent": False,
            "ttl_days": 1,
            "created_at": "2020-06-01T12:00:00",  # naive, definitely expired
            "updated_at": "2020-06-01T12:00:00",
            "metadata": {},
        }
        with open(entries_path, "w", encoding="utf-8") as f:
            f.write(json_mod.dumps(expired_entry) + "\n")

        # query() filters expired entries — must not raise TypeError
        results = store.query()
        assert all(not e.is_expired or e.content != "expired fact" for e in results)

    def test_sqlite_store_prune_naive_created_at(self, tmp_path):
        """Regression: SQLiteKnowledgeStore.prune() must handle naive ISO strings."""
        import sqlite3 as sqlite3_mod

        db_path = str(tmp_path / "naive.db")
        store = SQLiteKnowledgeStore(db_path=db_path)

        # Insert a row manually with a naive ISO timestamp (simulates old data)
        conn = sqlite3_mod.connect(db_path)
        conn.execute(
            """INSERT INTO knowledge (id, content, category, importance, persistent,
               ttl_days, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "naive-id-01",
                "old content",
                "general",
                0.5,
                0,
                1,  # ttl_days=1
                "2020-01-01T00:00:00",  # naive — no +00:00
                "2020-01-01T00:00:00",
                "{}",
            ),
        )
        conn.commit()
        conn.close()

        # prune() with ttl-expired entry must not raise TypeError
        removed = store.prune()
        assert removed >= 1  # the naive-dated entry should be pruned

    def test_sqlite_store_row_to_entry_naive(self, tmp_path):
        """Regression: _row_to_entry normalizes naive datetimes to UTC."""
        import sqlite3 as sqlite3_mod

        db_path = str(tmp_path / "naive2.db")
        store = SQLiteKnowledgeStore(db_path=db_path)

        conn = sqlite3_mod.connect(db_path)
        conn.execute(
            """INSERT INTO knowledge (id, content, category, importance, persistent,
               ttl_days, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "naive-id-02",
                "some content",
                "general",
                0.5,
                0,
                None,
                "2025-01-01T12:00:00",  # naive
                "2025-01-01T12:00:00",  # naive
                "{}",
            ),
        )
        conn.commit()
        conn.close()

        entry = store.get("naive-id-02")
        assert entry is not None
        # Must be timezone-aware after normalization
        assert entry.created_at.tzinfo is not None
        assert entry.updated_at.tzinfo is not None
        # is_expired must not raise TypeError
        _ = entry.is_expired

    def test_file_store_query_naive_since_does_not_crash(self, tmp_path):
        """Regression: FileKnowledgeStore.query(since=naive) must not raise TypeError.

        entry.created_at is always UTC-aware (normalized in _load_all()).
        If a naive since is passed, comparison raises TypeError without the fix.
        """
        store = FileKnowledgeStore(directory=str(tmp_path / "fks3"))
        store.save(KnowledgeEntry(content="some fact"))

        # Pass a naive datetime — must not crash
        naive_since = datetime(2020, 1, 1)  # no tzinfo
        results = store.query(since=naive_since)
        # The entry was created recently (after 2020), so it should be returned
        assert len(results) == 1

    def test_sqlite_store_query_naive_since_does_not_crash(self, tmp_path):
        """Regression: SQLiteKnowledgeStore.query(since=naive) must not crash.

        Stored ISO strings include +00:00; a naive since.isoformat() lacks it,
        making SQLite string comparison incorrect.  With the fix, naive since is
        normalized to UTC-aware before calling isoformat().
        """
        store = SQLiteKnowledgeStore(db_path=str(tmp_path / "naive_since.db"))
        store.save(KnowledgeEntry(content="recent fact"))

        # Pass a naive datetime — must not crash and must return the entry
        naive_since = datetime(2020, 1, 1)  # no tzinfo
        results = store.query(since=naive_since)
        assert len(results) == 1

    def test_file_store_load_all_invalid_isoformat_date(self, tmp_path):
        """Regression: FileKnowledgeStore._load_all() must skip entries with
        invalid ISO datetime strings without crashing (ValueError from
        datetime.fromisoformat() was not caught before the fix).
        """
        import json as json_mod

        store = FileKnowledgeStore(directory=str(tmp_path / "fks_invalid"))
        entries_path = store._entries_path
        os.makedirs(os.path.dirname(entries_path), exist_ok=True)

        # Write one corrupt entry with an invalid ISO date string, followed
        # by one valid entry.
        bad_entry = {
            "id": "bad-date-id",
            "content": "corrupt entry",
            "category": "general",
            "importance": 0.5,
            "persistent": False,
            "ttl_days": None,
            "created_at": "not-a-valid-date",  # invalid ISO format
            "updated_at": "also-not-valid",
            "metadata": {},
        }
        good_entry = {
            "id": "good-date-id",
            "content": "valid entry",
            "category": "general",
            "importance": 0.5,
            "persistent": False,
            "ttl_days": None,
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "metadata": {},
        }
        with open(entries_path, "w", encoding="utf-8") as f:
            f.write(json_mod.dumps(bad_entry) + "\n")
            f.write(json_mod.dumps(good_entry) + "\n")

        # _load_all() must not raise ValueError; the bad entry must be skipped.
        results = store.query()
        assert len(results) == 1
        assert results[0].id == "good-date-id"


# ======================================================================
# Regression tests for _enforce_max_entries counting bug
# ======================================================================


class TestEnforceMaxEntriesRegressions:
    """Regression: _enforce_max_entries must use live (non-expired) count."""

    def test_expired_entries_do_not_trigger_eviction_of_live_entries(self, tmp_path):
        """Regression: count() included expired entries, causing unnecessary eviction.

        With max_entries=5 and 3 live + 3 expired entries (total count=6),
        calling _enforce_max_entries() used to evict 1 live entry (6-5=1) even
        though only 3 live entries exist — well within the 5-entry limit.
        """
        from datetime import datetime, timedelta, timezone

        mem = KnowledgeMemory(directory=str(tmp_path / "evict_bug"), max_entries=5)
        # Add 3 non-expired entries
        for i in range(3):
            mem.remember(f"live-{i}", importance=0.5)

        # Manually inject 3 expired entries directly into the store
        from selectools.knowledge import KnowledgeEntry

        for i in range(3):
            e = KnowledgeEntry(
                content=f"expired-{i}",
                ttl_days=1,
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
            mem.store.save(e)

        # Live entries before: 3, expired entries: 3, total count: 6
        assert mem.store.count() == 6

        # Adding one more live entry should NOT evict any live entries
        # because 4 live entries < max_entries=5
        mem.remember("live-3", importance=0.5)

        live_entries = mem.store.query(limit=100)
        live_contents = {e.content for e in live_entries}
        # All 4 live entries must still be present
        assert "live-0" in live_contents
        assert "live-1" in live_contents
        assert "live-2" in live_contents
        assert "live-3" in live_contents
        assert len(live_entries) == 4

    def test_enforce_max_entries_still_evicts_when_truly_over_limit(self, tmp_path):
        """_enforce_max_entries must still evict when live entries exceed max_entries."""
        mem = KnowledgeMemory(directory=str(tmp_path / "evict_ok"), max_entries=3)
        for i in range(4):
            mem.remember(f"entry-{i}", importance=float(i) * 0.1)

        # Should have evicted the lowest-importance entry
        live_entries = mem.store.query(limit=100)
        assert len(live_entries) <= 3

    def test_enforce_max_entries_with_sqlite_store(self, tmp_path):
        """Regression also applies to SQLiteKnowledgeStore backend."""
        from datetime import datetime, timedelta, timezone

        from selectools.knowledge import KnowledgeEntry, SQLiteKnowledgeStore

        store = SQLiteKnowledgeStore(db_path=str(tmp_path / "evict.db"))
        mem = KnowledgeMemory(directory=str(tmp_path / "evict_sqlite"), store=store, max_entries=5)

        # Add 3 live entries
        for i in range(3):
            mem.remember(f"live-{i}", importance=0.5)

        # Inject 3 expired entries
        for i in range(3):
            e = KnowledgeEntry(
                content=f"expired-{i}",
                ttl_days=1,
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
            store.save(e)

        assert store.count() == 6

        # Add one live entry — should not evict existing live entries
        mem.remember("live-3", importance=0.5)

        live_entries = store.query(limit=100)
        live_contents = {e.content for e in live_entries}
        assert "live-0" in live_contents
        assert "live-3" in live_contents
        assert len(live_entries) == 4


# ======================================================================
# Shared entry serialization (issue #145)
# ======================================================================


class TestEntrySerialization:
    """entry_to_jsonable / entry_from_jsonable — the mapping all backends share."""

    def test_round_trip_preserves_every_field(self):
        original = KnowledgeEntry(
            id="rt-1",
            content="round trip",
            category="fact",
            importance=0.75,
            persistent=True,
            ttl_days=30,
            metadata={"source": "test", "nested": {"n": 1}},
        )
        assert entry_from_jsonable(entry_to_jsonable(original)) == original

    def test_to_jsonable_is_json_serializable(self):
        """Timestamps come out as ISO strings; metadata stays a live dict."""
        data = entry_to_jsonable(KnowledgeEntry(content="x", metadata={"k": "v"}))
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
        assert data["metadata"] == {"k": "v"}
        json.dumps(data)

    @pytest.mark.parametrize("raw", [0.9, "0.9", 1, "1"])
    def test_importance_is_always_a_float(self, raw):
        """Regression #145: every backend must coerce importance identically."""
        entry = entry_from_jsonable(
            {
                "id": "i",
                "content": "c",
                "importance": raw,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert type(entry.importance) is float
        assert entry.importance == float(raw)

    def test_null_fields_fall_back_to_defaults(self):
        """A NULL column must read as the schema default, not as None."""
        entry = entry_from_jsonable(
            {
                "id": "n",
                "content": "c",
                "category": None,
                "importance": None,
                "persistent": None,
                "ttl_days": None,
                "metadata": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert entry.category == "general"
        assert entry.importance == 0.5
        assert entry.persistent is False
        assert entry.ttl_days is None
        assert entry.metadata == {}

    def test_missing_optional_fields_fall_back_to_defaults(self):
        entry = entry_from_jsonable(
            {
                "id": "m",
                "content": "c",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert entry.category == "general"
        assert entry.importance == 0.5
        assert entry.persistent is False
        assert entry.ttl_days is None
        assert entry.metadata == {}

    @pytest.mark.parametrize(
        "raw,expected",
        [({"k": "v"}, {"k": "v"}), ('{"k": "v"}', {"k": "v"}), ("", {}), (None, {})],
    )
    def test_metadata_accepts_dict_json_string_and_empty(self, raw, expected):
        """JSONB backends pass a dict; text backends pass a (possibly empty) string."""
        entry = entry_from_jsonable(
            {
                "id": "md",
                "content": "c",
                "metadata": raw,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert entry.metadata == expected

    def test_naive_timestamps_are_normalized_to_utc(self):
        entry = entry_from_jsonable(
            {
                "id": "tz",
                "content": "c",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        )
        assert entry.created_at.tzinfo is timezone.utc
        assert entry.updated_at.tzinfo is timezone.utc

    def test_z_suffixed_timestamps_are_accepted(self):
        entry = entry_from_jsonable(
            {
                "id": "z",
                "content": "c",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
        assert entry.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_malformed_importance_raises_at_the_boundary(self):
        """Garbage must fail loudly here, not as a TypeError several frames later."""
        with pytest.raises(ValueError):
            entry_from_jsonable(
                {
                    "id": "bad",
                    "content": "c",
                    "importance": "high",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )


class TestSQLiteImportanceCoercion:
    """Regression #145: SQLiteKnowledgeStore._row_to_entry skipped the float() cast."""

    @staticmethod
    def _insert_raw(db_path, entry_id, importance):
        """Write a row directly, bypassing save(), the way legacy/external data arrives."""
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    "legacy row",
                    "general",
                    importance,
                    0,
                    None,
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                    "{}",
                ),
            )

    def test_null_importance_reads_as_default(self, tmp_path):
        """Previously: None from SQLite, 0.5 from Redis, TypeError from Supabase."""
        db = str(tmp_path / "null_imp.db")
        store = SQLiteKnowledgeStore(db)
        self._insert_raw(db, "nul", None)

        entry = store.get("nul")
        assert entry is not None
        assert type(entry.importance) is float
        assert entry.importance == 0.5

    def test_non_numeric_importance_raises_instead_of_returning_a_str(self, tmp_path):
        """SQLite's REAL affinity keeps non-numeric text as TEXT.

        Before the fix ``get()`` handed back the raw ``str``, which only blew up
        later inside ``KnowledgeMemory.build_context()`` as ``TypeError: '>=' not
        supported between instances of 'str' and 'float'``.  Redis and Supabase
        already raised ValueError at the boundary.
        """
        db = str(tmp_path / "bad_imp.db")
        store = SQLiteKnowledgeStore(db)
        self._insert_raw(db, "bad", "high")

        with pytest.raises(ValueError):
            store.get("bad")

    def test_corrupt_importance_surfaces_at_the_boundary_not_in_build_context(self, tmp_path):
        """The end-to-end symptom reported in issue #145.

        A non-numeric importance survives the ``importance >= ?`` filter (SQLite
        sorts TEXT above REAL), so before the fix it reached ``build_context()``
        as a ``str`` and blew up there with ``TypeError: '>=' not supported
        between instances of 'str' and 'float'`` — a message that names neither
        the store nor the offending row.  It now raises ``ValueError`` from the
        store, where the bad value is.
        """
        db = str(tmp_path / "mixed.db")
        store = SQLiteKnowledgeStore(db)
        store.save(KnowledgeEntry(id="ok", content="normal", importance=0.6))
        self._insert_raw(db, "bad", "high")

        mem = KnowledgeMemory(directory=str(tmp_path / "mixed_mem"), store=store)
        with pytest.raises(ValueError):
            mem.build_context()

    def test_build_context_unaffected_for_well_formed_rows(self, tmp_path):
        db = str(tmp_path / "clean.db")
        store = SQLiteKnowledgeStore(db)
        store.save(KnowledgeEntry(id="hi", content="important thing", importance=0.9))
        store.save(KnowledgeEntry(id="lo", content="minor thing", importance=0.2))

        mem = KnowledgeMemory(directory=str(tmp_path / "clean_mem"), store=store)
        context = mem.build_context()
        assert "important thing" in context
        assert "minor thing" in context
        assert all(type(e.importance) is float for e in store.query())

    def test_integer_importance_reads_back_as_float(self, tmp_path):
        db = str(tmp_path / "int_imp.db")
        store = SQLiteKnowledgeStore(db)
        self._insert_raw(db, "one", 1)

        entry = store.get("one")
        assert entry is not None
        assert type(entry.importance) is float
        assert entry.importance == 1.0

    def test_existing_rows_still_round_trip(self, tmp_path):
        """The on-disk schema is unchanged, so rows written before the fix still load."""
        db = str(tmp_path / "compat.db")
        store = SQLiteKnowledgeStore(db)
        original = KnowledgeEntry(
            id="compat",
            content="written the old way",
            category="fact",
            importance=0.8,
            persistent=True,
            ttl_days=7,
            metadata={"a": 1},
        )
        store.save(original)

        assert store.get("compat") == original

    def test_file_store_importance_is_coerced(self, tmp_path):
        """FileKnowledgeStore also skipped the cast; a JSONL string must coerce."""
        store = FileKnowledgeStore(directory=str(tmp_path / "fks_imp"))
        with open(store._entries_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": "str-imp",
                        "content": "from jsonl",
                        "importance": "0.9",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    }
                )
                + "\n"
            )

        entry = store.get("str-imp")
        assert entry is not None
        assert type(entry.importance) is float
        assert entry.importance == 0.9
