"""
Tests for RedisCache (cache_redis.py) using mocks.

Tests cover:
- Import error when redis is not installed
- get() hit/miss tracking
- set() with default and custom TTL
- delete() existing and nonexistent keys
- clear() prefixed keys
- Key prefixing (_full_key)
- Stats property
- __repr__
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.cache import CacheStats


class FakeRedisClient:
    """In-memory mock that mimics the subset of redis.Redis used by RedisCache."""

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        self._store[key] = value

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                count += 1
        return count

    def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> tuple:
        matched = [k for k in self._store if k.startswith(match.rstrip("*"))]
        return (0, matched)


def _make_redis_cache(**kwargs: Any) -> Any:
    """Create a RedisCache with a mocked redis connection."""
    fake_client = FakeRedisClient()

    fake_redis_module = MagicMock()
    fake_redis_module.from_url = MagicMock(return_value=fake_client)

    with patch.dict("sys.modules", {"redis": fake_redis_module}):
        from selectools.cache_redis import RedisCache

        cache = RedisCache(**kwargs)
        cache._client = fake_client
        return cache


class TestRedisCacheImportError:
    """Test import error handling."""

    def test_import_error_without_redis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(__import__("sys").modules, "redis", None)
        with pytest.raises(ImportError, match="redis"):
            from selectools.cache_redis import RedisCache

            RedisCache()


class TestRedisCacheGetSet:
    """Tests for get() and set() operations."""

    def test_get_miss(self) -> None:
        cache = _make_redis_cache()
        result = cache.get("nonexistent")

        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_set_and_get(self) -> None:
        cache = _make_redis_cache()
        value = ({"content": "hello"}, {"tokens": 10})
        cache.set("key1", value)
        result = cache.get("key1")

        assert result is not None
        assert result[0] == {"content": "hello"}
        assert result[1] == {"tokens": 10}
        assert cache.stats.hits == 1

    def test_overwrite_existing_key(self) -> None:
        cache = _make_redis_cache()
        cache.set("key1", ("first", {}))
        cache.set("key1", ("second", {}))
        result = cache.get("key1")

        assert result is not None
        assert result[0] == "second"

    def test_multiple_keys(self) -> None:
        cache = _make_redis_cache()
        cache.set("a", ("val_a", {}))
        cache.set("b", ("val_b", {}))

        assert cache.get("a")[0] == "val_a"
        assert cache.get("b")[0] == "val_b"

    def test_stats_accumulate(self) -> None:
        cache = _make_redis_cache()
        cache.set("key1", ("val", {}))

        cache.get("key1")
        cache.get("key1")
        cache.get("missing")

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1


class TestRedisCacheDelete:
    """Tests for delete() operation."""

    def test_delete_existing(self) -> None:
        cache = _make_redis_cache()
        cache.set("key1", ("val", {}))

        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self) -> None:
        cache = _make_redis_cache()
        result = cache.delete("nonexistent")
        assert result is False


class TestRedisCacheClear:
    """Tests for clear() operation."""

    def test_clear_removes_all(self) -> None:
        cache = _make_redis_cache()
        cache.set("key1", ("a", {}))
        cache.set("key2", ("b", {}))

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_resets_stats(self) -> None:
        cache = _make_redis_cache()
        cache.set("key1", ("a", {}))
        cache.get("key1")
        cache.get("missing")

        cache.clear()

        assert cache.stats.hits == 0
        assert cache.stats.misses == 0


class TestRedisCacheKeyPrefixing:
    """Tests for key prefixing behavior."""

    def test_default_prefix(self) -> None:
        cache = _make_redis_cache()
        assert cache._prefix == "selectools:"

    def test_custom_prefix(self) -> None:
        cache = _make_redis_cache(prefix="myapp:")
        assert cache._prefix == "myapp:"

    def test_full_key_adds_prefix(self) -> None:
        cache = _make_redis_cache(prefix="test:")
        assert cache._full_key("abc") == "test:abc"

    def test_full_key_no_double_prefix(self) -> None:
        cache = _make_redis_cache(prefix="test:")
        assert cache._full_key("test:abc") == "test:abc"


class TestRedisCacheRepr:
    """Tests for __repr__."""

    def test_repr_default(self) -> None:
        cache = _make_redis_cache()
        r = repr(cache)
        assert "RedisCache" in r
        assert "selectools:" in r
        assert "300" in r

    def test_repr_custom(self) -> None:
        cache = _make_redis_cache(prefix="custom:", default_ttl=600)
        r = repr(cache)
        assert "custom:" in r
        assert "600" in r


class TestRedisCacheTTL:
    """Tests for TTL handling."""

    def test_default_ttl(self) -> None:
        cache = _make_redis_cache(default_ttl=300)
        assert cache._default_ttl == 300

    def test_none_ttl(self) -> None:
        cache = _make_redis_cache(default_ttl=None)
        assert cache._default_ttl is None

        cache.set("key1", ("val", {}))
        result = cache.get("key1")
        assert result is not None

    def test_custom_ttl_override(self) -> None:
        cache = _make_redis_cache(default_ttl=300)
        cache.set("key1", ("val", {}), ttl=60)
        result = cache.get("key1")
        assert result is not None


class TestRedisCacheStats:
    """Tests for stats property."""

    def test_stats_returns_cache_stats(self) -> None:
        cache = _make_redis_cache()
        assert isinstance(cache.stats, CacheStats)

    def test_initial_stats(self) -> None:
        cache = _make_redis_cache()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
