"""Tests for RedisKnowledgeStore — mock-based, no real Redis required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import pytest

from selectools.knowledge import KnowledgeEntry
from selectools.knowledge_store_redis import RedisKnowledgeStore

# ---------------------------------------------------------------------------
# Helpers — in-memory fake Redis primitives
# ---------------------------------------------------------------------------


class FakeRedis:
    """Minimal in-memory Redis double that supports hashes, sorted sets, and sets."""

    def __init__(self) -> None:
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}
        self._sets: Dict[str, Set[str]] = {}

    # -- hash commands -----------------------------------------------------

    def hset(self, name: str, mapping: Dict[str, str] | None = None, **kwargs: str) -> int:
        if mapping is None:
            mapping = kwargs
        self._hashes.setdefault(name, {}).update(mapping)
        return len(mapping)

    def hget(self, name: str, key: str) -> Optional[str]:
        return self._hashes.get(name, {}).get(key)

    def hgetall(self, name: str) -> Dict[str, str]:
        return dict(self._hashes.get(name, {}))

    # -- sorted set commands -----------------------------------------------

    def zadd(self, name: str, mapping: Dict[str, float]) -> int:
        self._sorted_sets.setdefault(name, {}).update(mapping)
        return len(mapping)

    def zrem(self, name: str, *members: str) -> int:
        zset = self._sorted_sets.get(name, {})
        removed = 0
        for m in members:
            if m in zset:
                del zset[m]
                removed += 1
        return removed

    def zrevrangebyscore(self, name: str, max_score: str, min_score: str) -> List[str]:
        zset = self._sorted_sets.get(name, {})
        max_val = float("inf") if max_score == "+inf" else float(max_score)
        min_val = float(min_score)
        items = [(m, s) for m, s in zset.items() if min_val <= s <= max_val]
        items.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in items]

    # -- set commands ------------------------------------------------------

    def sadd(self, name: str, *members: str) -> int:
        self._sets.setdefault(name, set()).update(members)
        return len(members)

    def srem(self, name: str, *members: str) -> int:
        s = self._sets.get(name, set())
        removed = 0
        for m in members:
            if m in s:
                s.discard(m)
                removed += 1
        return removed

    def smembers(self, name: str) -> Set[str]:
        return set(self._sets.get(name, set()))

    def scard(self, name: str) -> int:
        return len(self._sets.get(name, set()))

    # -- key commands ------------------------------------------------------

    def delete(self, *names: str) -> int:
        removed = 0
        for n in names:
            if n in self._hashes:
                del self._hashes[n]
                removed += 1
        return removed

    # -- pipeline ----------------------------------------------------------

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """Collects commands and executes them sequentially."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: List[tuple] = []

    def hset(self, name: str, mapping: Dict[str, str] | None = None, **kw: str) -> "FakePipeline":
        self._commands.append(("hset", name, mapping or kw))
        return self

    def zadd(self, name: str, mapping: Dict[str, float]) -> "FakePipeline":
        self._commands.append(("zadd", name, mapping))
        return self

    def sadd(self, name: str, *members: str) -> "FakePipeline":
        self._commands.append(("sadd", name, members))
        return self

    def zrem(self, name: str, *members: str) -> "FakePipeline":
        self._commands.append(("zrem", name, members))
        return self

    def srem(self, name: str, *members: str) -> "FakePipeline":
        self._commands.append(("srem", name, members))
        return self

    def delete(self, *names: str) -> "FakePipeline":
        self._commands.append(("delete", names))
        return self

    def execute(self) -> List[Any]:
        results: List[Any] = []
        for cmd in self._commands:
            if cmd[0] == "hset":
                results.append(self._redis.hset(cmd[1], mapping=cmd[2]))
            elif cmd[0] == "zadd":
                results.append(self._redis.zadd(cmd[1], cmd[2]))
            elif cmd[0] == "sadd":
                results.append(self._redis.sadd(cmd[1], *cmd[2]))
            elif cmd[0] == "zrem":
                results.append(self._redis.zrem(cmd[1], *cmd[2]))
            elif cmd[0] == "srem":
                results.append(self._redis.srem(cmd[1], *cmd[2]))
            elif cmd[0] == "delete":
                results.append(self._redis.delete(*cmd[1]))
        self._commands.clear()
        return results


@pytest.fixture()
def store() -> RedisKnowledgeStore:
    return RedisKnowledgeStore(redis_client=FakeRedis(), prefix="test")


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
    def test_save_returns_id(self, store: RedisKnowledgeStore) -> None:
        entry = _make_entry()
        result = store.save(entry)
        assert result == entry.id

    def test_save_and_get_roundtrip(self, store: RedisKnowledgeStore) -> None:
        entry = _make_entry(content="The sky is blue", importance=0.9)
        store.save(entry)
        fetched = store.get(entry.id)
        assert fetched is not None
        assert fetched.content == "The sky is blue"
        assert fetched.importance == 0.9
        assert fetched.category == "fact"


class TestGet:
    def test_get_missing_returns_none(self, store: RedisKnowledgeStore) -> None:
        assert store.get("nonexistent") is None


class TestDelete:
    def test_delete_existing_returns_true(self, store: RedisKnowledgeStore) -> None:
        entry = _make_entry()
        store.save(entry)
        assert store.delete(entry.id) is True

    def test_delete_missing_returns_false(self, store: RedisKnowledgeStore) -> None:
        assert store.delete("nonexistent") is False

    def test_delete_removes_from_count(self, store: RedisKnowledgeStore) -> None:
        entry = _make_entry()
        store.save(entry)
        assert store.count() == 1
        store.delete(entry.id)
        assert store.count() == 0


class TestCount:
    def test_empty_store(self, store: RedisKnowledgeStore) -> None:
        assert store.count() == 0

    def test_count_after_saves(self, store: RedisKnowledgeStore) -> None:
        store.save(_make_entry(id="a"))
        store.save(_make_entry(id="b"))
        assert store.count() == 2


class TestQueryByCategory:
    def test_query_filters_by_category(self, store: RedisKnowledgeStore) -> None:
        store.save(_make_entry(id="1", category="fact"))
        store.save(_make_entry(id="2", category="preference"))
        store.save(_make_entry(id="3", category="fact"))

        results = store.query(category="fact")
        assert len(results) == 2
        assert all(e.category == "fact" for e in results)

    def test_query_empty_category(self, store: RedisKnowledgeStore) -> None:
        store.save(_make_entry(id="1", category="fact"))
        results = store.query(category="preference")
        assert results == []


class TestQueryByImportance:
    def test_query_min_importance(self, store: RedisKnowledgeStore) -> None:
        store.save(_make_entry(id="low", importance=0.2))
        store.save(_make_entry(id="mid", importance=0.5))
        store.save(_make_entry(id="high", importance=0.9))

        results = store.query(min_importance=0.5)
        assert len(results) == 2
        assert results[0].importance >= results[1].importance

    def test_query_ordered_by_importance_desc(self, store: RedisKnowledgeStore) -> None:
        store.save(_make_entry(id="a", importance=0.3))
        store.save(_make_entry(id="b", importance=0.8))
        store.save(_make_entry(id="c", importance=0.5))

        results = store.query()
        importances = [e.importance for e in results]
        assert importances == sorted(importances, reverse=True)


class TestQueryLimit:
    def test_query_respects_limit(self, store: RedisKnowledgeStore) -> None:
        for i in range(5):
            store.save(_make_entry(id=str(i)))
        results = store.query(limit=2)
        assert len(results) == 2


class TestPrune:
    def test_prune_expired_entries(self, store: RedisKnowledgeStore) -> None:
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

    def test_prune_by_max_age(self, store: RedisKnowledgeStore) -> None:
        old = _make_entry(
            id="old",
            created_at=datetime.now(timezone.utc) - timedelta(days=100),
        )
        recent = _make_entry(id="recent")
        store.save(old)
        store.save(recent)

        removed = store.prune(max_age_days=30)
        assert removed == 1
        assert store.get("old") is None
        assert store.get("recent") is not None

    def test_prune_by_min_importance(self, store: RedisKnowledgeStore) -> None:
        low = _make_entry(id="low", importance=0.1)
        high = _make_entry(id="high", importance=0.9)
        store.save(low)
        store.save(high)

        removed = store.prune(min_importance=0.5)
        assert removed == 1
        assert store.get("low") is None
        assert store.get("high") is not None

    def test_prune_skips_persistent(self, store: RedisKnowledgeStore) -> None:
        entry = _make_entry(
            id="pinned",
            persistent=True,
            importance=0.1,
        )
        store.save(entry)

        removed = store.prune(min_importance=0.5)
        assert removed == 0
        assert store.get("pinned") is not None


class TestQuerySince:
    def test_query_since_filter(self, store: RedisKnowledgeStore) -> None:
        old = _make_entry(
            id="old",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        recent = _make_entry(id="recent")
        store.save(old)
        store.save(recent)

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        results = store.query(since=cutoff)
        assert len(results) == 1
        assert results[0].id == "recent"
