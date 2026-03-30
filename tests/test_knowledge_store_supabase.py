"""Tests for SupabaseKnowledgeStore — mock-based, no real Supabase required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from selectools.knowledge import KnowledgeEntry
from selectools.knowledge_store_supabase import SupabaseKnowledgeStore

# ---------------------------------------------------------------------------
# Helpers — Supabase client mock builder
# ---------------------------------------------------------------------------


class FakeResponse:
    """Mimics a Supabase API response."""

    def __init__(self, data: Optional[List[Dict[str, Any]]] = None, count: Optional[int] = None):
        self.data = data or []
        self.count = count


class FakeQueryBuilder:
    """Chainable query builder that stores entries in memory."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self._filters: List[tuple] = []
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._limit_val: Optional[int] = None
        self._select_cols: Optional[str] = None
        self._count_mode: Optional[str] = None
        self._is_delete: bool = False

    def select(self, cols: str = "*", count: Optional[str] = None) -> "FakeQueryBuilder":
        self._select_cols = cols
        self._count_mode = count
        return self

    def eq(self, col: str, val: Any) -> "FakeQueryBuilder":
        self._filters.append(("eq", col, val))
        return self

    def gte(self, col: str, val: Any) -> "FakeQueryBuilder":
        self._filters.append(("gte", col, val))
        return self

    def order(self, col: str, desc: bool = False) -> "FakeQueryBuilder":
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int) -> "FakeQueryBuilder":
        self._limit_val = n
        return self

    def upsert(self, row: Dict[str, Any]) -> "FakeQueryBuilder":
        # Remove existing row with same id
        self._rows[:] = [r for r in self._rows if r["id"] != row["id"]]
        self._rows.append(row)
        return self

    def delete(self) -> "FakeQueryBuilder":
        self._is_delete = True
        return self

    def execute(self) -> FakeResponse:
        if self._is_delete:
            to_delete: List[Dict[str, Any]] = []
            for row in self._rows:
                if self._matches(row):
                    to_delete.append(row)
            for r in to_delete:
                self._rows.remove(r)
            return FakeResponse(data=to_delete)

        result = [r for r in self._rows if self._matches(r)]

        if self._order_col:
            result.sort(
                key=lambda r: r.get(self._order_col, 0),  # type: ignore[arg-type]
                reverse=self._order_desc,
            )

        if self._limit_val is not None:
            result = result[: self._limit_val]

        count_val = len(self._rows) if self._count_mode == "exact" else None
        return FakeResponse(data=result, count=count_val)

    def _matches(self, row: Dict[str, Any]) -> bool:
        for op, col, val in self._filters:
            rv = row.get(col)
            if op == "eq":
                if rv != val:
                    return False
            elif op == "gte":
                if rv is None or rv < val:
                    return False
        return True


class FakeSupabaseClient:
    """Mimics supabase.Client with in-memory storage per table."""

    def __init__(self) -> None:
        self._tables: Dict[str, List[Dict[str, Any]]] = {}

    def table(self, name: str) -> FakeQueryBuilder:
        self._tables.setdefault(name, [])
        return FakeQueryBuilder(self._tables[name])


@pytest.fixture()
def store() -> SupabaseKnowledgeStore:
    return SupabaseKnowledgeStore(client=FakeSupabaseClient(), table_name="knowledge")


def _make_entry(**overrides: Any) -> KnowledgeEntry:
    defaults: Dict[str, Any] = {
        "content": "test knowledge",
        "category": "fact",
        "importance": 0.7,
    }
    defaults.update(overrides)
    return KnowledgeEntry(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_returns_id(self, store: SupabaseKnowledgeStore) -> None:
        entry = _make_entry()
        result = store.save(entry)
        assert result == entry.id

    def test_save_and_get_roundtrip(self, store: SupabaseKnowledgeStore) -> None:
        entry = _make_entry(content="Supabase works", importance=0.8)
        store.save(entry)
        fetched = store.get(entry.id)
        assert fetched is not None
        assert fetched.content == "Supabase works"
        assert fetched.importance == 0.8


class TestGet:
    def test_get_missing_returns_none(self, store: SupabaseKnowledgeStore) -> None:
        assert store.get("nonexistent") is None


class TestDelete:
    def test_delete_existing_returns_true(self, store: SupabaseKnowledgeStore) -> None:
        entry = _make_entry()
        store.save(entry)
        assert store.delete(entry.id) is True

    def test_delete_missing_returns_false(self, store: SupabaseKnowledgeStore) -> None:
        assert store.delete("nonexistent") is False


class TestCount:
    def test_empty_store(self, store: SupabaseKnowledgeStore) -> None:
        assert store.count() == 0

    def test_count_after_saves(self, store: SupabaseKnowledgeStore) -> None:
        store.save(_make_entry(id="a"))
        store.save(_make_entry(id="b"))
        assert store.count() == 2


class TestQuery:
    def test_query_by_category(self, store: SupabaseKnowledgeStore) -> None:
        store.save(_make_entry(id="1", category="fact"))
        store.save(_make_entry(id="2", category="preference"))
        results = store.query(category="fact")
        assert len(results) == 1
        assert results[0].category == "fact"

    def test_query_min_importance(self, store: SupabaseKnowledgeStore) -> None:
        store.save(_make_entry(id="low", importance=0.2))
        store.save(_make_entry(id="high", importance=0.9))
        results = store.query(min_importance=0.5)
        assert len(results) == 1
        assert results[0].id == "high"

    def test_query_ordered_by_importance_desc(self, store: SupabaseKnowledgeStore) -> None:
        store.save(_make_entry(id="a", importance=0.3))
        store.save(_make_entry(id="b", importance=0.8))
        store.save(_make_entry(id="c", importance=0.5))
        results = store.query()
        importances = [e.importance for e in results]
        assert importances == sorted(importances, reverse=True)

    def test_query_respects_limit(self, store: SupabaseKnowledgeStore) -> None:
        for i in range(5):
            store.save(_make_entry(id=str(i)))
        results = store.query(limit=2)
        assert len(results) == 2


class TestPrune:
    def test_prune_expired_entries(self, store: SupabaseKnowledgeStore) -> None:
        expired = _make_entry(
            id="expired",
            ttl_days=1,
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        fresh = _make_entry(id="fresh", ttl_days=30)
        store.save(expired)
        store.save(fresh)

        removed = store.prune()
        assert removed == 1
        assert store.get("expired") is None
        assert store.get("fresh") is not None

    def test_prune_skips_persistent(self, store: SupabaseKnowledgeStore) -> None:
        entry = _make_entry(id="pinned", persistent=True, importance=0.1)
        store.save(entry)
        removed = store.prune(min_importance=0.5)
        assert removed == 0
        assert store.get("pinned") is not None


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestNaiveDatetimeRegression:
    """Regression: naive datetimes from Supabase must not crash on is_expired comparisons."""

    def test_naive_timestamp_from_supabase_does_not_crash_is_expired(
        self, store: SupabaseKnowledgeStore
    ) -> None:
        """Regression: _row_to_entry must normalize naive datetimes to UTC-aware.

        Supabase may return TIMESTAMPTZ values as strings without timezone info
        (e.g. '2020-01-01T00:00:00').  Without normalization, datetime.fromisoformat()
        returns a naive datetime, and comparing it with datetime.now(timezone.utc)
        in is_expired raises TypeError.
        """
        # Inject a row with naive timestamps directly into the fake store
        row = {
            "id": "naive_test",
            "content": "test",
            "category": "fact",
            "importance": 0.7,
            "persistent": False,
            "ttl_days": 1,
            "created_at": "2020-01-01T00:00:00",  # naive, no tz
            "updated_at": "2020-01-01T00:00:00",
            "metadata": {},
        }
        store._client._tables.setdefault(store._table, []).append(row)

        # Must not raise TypeError
        result = store.get("naive_test")
        assert result is not None
        assert result.is_expired  # created in 2020 with ttl_days=1 is expired

    def test_naive_since_does_not_crash_query(self, store: SupabaseKnowledgeStore) -> None:
        """Regression: query(since=naive_datetime) must not raise TypeError."""
        store.save(_make_entry(id="entry1"))
        # Pass a naive datetime as since — should not crash
        naive_since = datetime(2020, 1, 1)  # no tzinfo
        results = store.query(since=naive_since)
        assert len(results) == 1
