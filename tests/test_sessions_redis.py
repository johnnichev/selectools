"""
Tests for RedisSessionStore backend.

Uses a mock Redis client to test all 5 SessionStore protocol methods
without requiring a real Redis server.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import RedisSessionStore, SessionMetadata
from selectools.types import Message, Role, ToolCall

# ======================================================================
# Mock Redis client
# ======================================================================


class FakeRedis:
    """In-memory fake Redis client for testing."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._ttls and time.time() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            return None
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self._ttls.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = time.time() + ttl

    def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                self._ttls.pop(k, None)
                removed += 1
        return removed

    def exists(self, key: str) -> int:
        if key in self._ttls and time.time() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            return 0
        return 1 if key in self._store else 0

    def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> tuple:
        import fnmatch

        keys = [k for k in self._store if fnmatch.fnmatch(k, match)]
        return (0, keys)

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """Fake Redis pipeline that queues operations."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list = []

    def set(self, key: str, value: str) -> "FakePipeline":
        self._ops.append(("set", key, value))
        return self

    def setex(self, key: str, ttl: int, value: str) -> "FakePipeline":
        self._ops.append(("setex", key, ttl, value))
        return self

    def execute(self) -> list:
        results = []
        for op in self._ops:
            if op[0] == "set":
                self._redis.set(op[1], op[2])
                results.append(True)
            elif op[0] == "setex":
                self._redis.setex(op[1], op[2], op[3])
                results.append(True)
        self._ops.clear()
        return results


def _memory_with_messages(*contents: str) -> ConversationMemory:
    mem = ConversationMemory(max_messages=50)
    for c in contents:
        mem.add(Message(role=Role.USER, content=c))
    return mem


def _make_redis_store(
    fake_redis: FakeRedis,
    prefix: str = "selectools:session:",
    default_ttl: Optional[int] = None,
) -> RedisSessionStore:
    """Create a RedisSessionStore with a mocked redis import."""
    fake_module = MagicMock()
    fake_module.from_url = MagicMock(return_value=fake_redis)

    with patch.dict("sys.modules", {"redis": fake_module}):
        store = RedisSessionStore(
            url="redis://localhost:6379/0",
            prefix=prefix,
            default_ttl=default_ttl,
        )
    return store


# ======================================================================
# Save / Load
# ======================================================================


class TestRedisSessionStoreSaveLoad:
    def test_save_and_load_round_trip(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        mem = _memory_with_messages("Hello", "World")
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded.get_history()[0].content == "Hello"
        assert loaded.get_history()[1].content == "World"

    def test_load_nonexistent_returns_none(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        assert store.load("nonexistent") is None

    def test_save_overwrites_existing(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("v1"))
        store.save("s1", _memory_with_messages("v2"))

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.get_history()[0].content == "v2"

    def test_preserves_created_at_on_overwrite(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("v1"))

        meta_key = "selectools:session:s1:meta"
        first_meta = json.loads(fake.get(meta_key))
        first_created = first_meta["created_at"]

        time.sleep(0.01)
        store.save("s1", _memory_with_messages("v2"))

        second_meta = json.loads(fake.get(meta_key))
        assert second_meta["created_at"] == first_created
        assert second_meta["updated_at"] > first_created

    def test_preserves_tool_calls(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        mem = ConversationMemory()
        mem.add(Message(role=Role.USER, content="Search"))
        tc = ToolCall(tool_name="search", parameters={"q": "ai"}, id="tc1")
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc]))
        mem.add(Message(role=Role.TOOL, content="result", tool_name="search", tool_call_id="tc1"))
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        history = loaded.get_history()
        assert history[1].tool_calls[0].tool_name == "search"
        assert history[2].tool_name == "search"

    def test_preserves_summary(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        mem = _memory_with_messages("Hello")
        mem.summary = "User said hello"
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.summary == "User said hello"


# ======================================================================
# TTL
# ======================================================================


class TestRedisSessionStoreTTL:
    def test_save_with_ttl_uses_setex(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake, default_ttl=3600)
        store.save("s1", _memory_with_messages("Hello"))

        key = "selectools:session:s1"
        assert key in fake._store
        assert key in fake._ttls

    def test_save_without_ttl_uses_set(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake, default_ttl=None)
        store.save("s1", _memory_with_messages("Hello"))

        key = "selectools:session:s1"
        assert key in fake._store
        assert key not in fake._ttls


# ======================================================================
# Delete / List / Exists
# ======================================================================


class TestRedisSessionStoreDeleteListExists:
    def test_delete_existing(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("Hello"))
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        assert store.delete("nope") is False

    def test_delete_removes_both_keys(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("Hello"))
        store.delete("s1")
        assert "selectools:session:s1" not in fake._store
        assert "selectools:session:s1:meta" not in fake._store

    def test_exists_true(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("Hello"))
        assert store.exists("s1") is True

    def test_exists_false(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        assert store.exists("s1") is False

    def test_list_sessions(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}
        for s in sessions:
            assert isinstance(s, SessionMetadata)

    def test_list_empty(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        assert store.list() == []

    def test_list_message_counts(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        by_id = {s.session_id: s for s in sessions}
        assert by_id["s1"].message_count == 1
        assert by_id["s2"].message_count == 2

    def test_list_ignores_corrupt_meta(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("ok"))
        # Corrupt the meta key
        fake._store["selectools:session:s1:meta"] = "not json{{"

        sessions = store.list()
        assert len(sessions) == 0

    def test_custom_prefix(self) -> None:
        fake = FakeRedis()
        store = _make_redis_store(fake, prefix="myapp:")
        store.save("s1", _memory_with_messages("Hello"))
        assert "myapp:s1" in fake._store
        assert "myapp:s1:meta" in fake._store


# ======================================================================
# Import error
# ======================================================================


class TestRedisSessionStoreImportError:
    def test_missing_redis_raises_import_error(self) -> None:
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(ImportError, match="redis"):
                RedisSessionStore()


# ======================================================================
# Edge cases: corrupt meta on save
# ======================================================================


class TestRedisSessionStoreEdgeCases:
    def test_save_with_corrupt_existing_meta(self) -> None:
        """If existing meta is corrupt JSON, save should still work."""
        fake = FakeRedis()
        store = _make_redis_store(fake)
        # Place corrupt meta
        fake._store["selectools:session:s1:meta"] = "not valid json"
        store.save("s1", _memory_with_messages("Hello"))

        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 1

    def test_list_skips_missing_meta_value(self) -> None:
        """If meta key exists but get returns None (race condition), skip it."""
        fake = FakeRedis()
        store = _make_redis_store(fake)
        store.save("s1", _memory_with_messages("Hello"))
        # Simulate race: meta key in scan but value deleted
        fake._store["selectools:session:s1:meta"] = None  # type: ignore[assignment]

        # Should not crash
        sessions = store.list()
        # The None value should be handled
        assert isinstance(sessions, list)
