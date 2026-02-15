"""Unit tests for the caching module."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from selectools.cache import CacheKeyBuilder, CacheStats, InMemoryCache
from selectools.types import Message, Role, ToolCall

# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_default_values(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_hit_rate_zero_requests(self) -> None:
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == pytest.approx(0.3)

    def test_hit_rate_all_hits(self) -> None:
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == pytest.approx(1.0)

    def test_total_requests(self) -> None:
        stats = CacheStats(hits=5, misses=3)
        assert stats.total_requests == 8

    def test_repr(self) -> None:
        stats = CacheStats(hits=1, misses=1)
        r = repr(stats)
        assert "hits=1" in r
        assert "misses=1" in r
        assert "hit_rate=" in r


# ---------------------------------------------------------------------------
# InMemoryCache: basic get / set
# ---------------------------------------------------------------------------


class TestInMemoryCacheBasic:
    def test_get_miss(self) -> None:
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        cache = InMemoryCache()
        msg = Message(role=Role.ASSISTANT, content="hello")
        stats_obj = MagicMock()
        cache.set("key1", (msg, stats_obj))
        result = cache.get("key1")
        assert result is not None
        assert result[0] is msg
        assert result[1] is stats_obj

    def test_overwrite_existing_key(self) -> None:
        cache = InMemoryCache()
        msg1 = Message(role=Role.ASSISTANT, content="first")
        msg2 = Message(role=Role.ASSISTANT, content="second")
        cache.set("k", (msg1, None))
        cache.set("k", (msg2, None))
        result = cache.get("k")
        assert result is not None
        assert result[0].content == "second"

    def test_size_property(self) -> None:
        cache = InMemoryCache()
        assert cache.size == 0
        cache.set("a", ("x", "y"))
        assert cache.size == 1
        cache.set("b", ("x", "y"))
        assert cache.size == 2

    def test_delete_existing(self) -> None:
        cache = InMemoryCache()
        cache.set("k", ("a", "b"))
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_nonexistent(self) -> None:
        cache = InMemoryCache()
        assert cache.delete("nope") is False

    def test_clear(self) -> None:
        cache = InMemoryCache()
        cache.set("a", ("1", "2"))
        cache.set("b", ("3", "4"))
        cache.get("a")  # hit
        cache.get("missing")  # miss
        cache.clear()
        assert cache.size == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_negative_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            InMemoryCache(max_size=-1)

    def test_repr(self) -> None:
        cache = InMemoryCache(max_size=50, default_ttl=60)
        r = repr(cache)
        assert "max_size=50" in r
        assert "default_ttl=60" in r


# ---------------------------------------------------------------------------
# InMemoryCache: TTL
# ---------------------------------------------------------------------------


class TestInMemoryCacheTTL:
    def test_entry_expires_after_ttl(self) -> None:
        cache = InMemoryCache(default_ttl=1)
        cache.set("k", ("val", "stats"))
        assert cache.get("k") is not None
        time.sleep(1.1)
        assert cache.get("k") is None

    def test_per_entry_ttl_override(self) -> None:
        cache = InMemoryCache(default_ttl=60)
        cache.set("short", ("v", "s"), ttl=1)
        cache.set("long", ("v", "s"), ttl=60)
        time.sleep(1.1)
        assert cache.get("short") is None  # expired
        assert cache.get("long") is not None  # still valid

    def test_no_ttl_means_no_expiry(self) -> None:
        cache = InMemoryCache(default_ttl=None)
        cache.set("forever", ("v", "s"))
        # Entry has expires_at=0.0 which means "never"
        result = cache.get("forever")
        assert result is not None


# ---------------------------------------------------------------------------
# InMemoryCache: LRU eviction
# ---------------------------------------------------------------------------


class TestInMemoryCacheLRU:
    def test_evicts_oldest_when_full(self) -> None:
        cache = InMemoryCache(max_size=3, default_ttl=None)
        cache.set("a", ("1", "s"))
        cache.set("b", ("2", "s"))
        cache.set("c", ("3", "s"))
        # Cache full with [a, b, c]
        cache.set("d", ("4", "s"))  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") is not None
        assert cache.stats.evictions == 1

    def test_access_refreshes_lru_order(self) -> None:
        cache = InMemoryCache(max_size=3, default_ttl=None)
        cache.set("a", ("1", "s"))
        cache.set("b", ("2", "s"))
        cache.set("c", ("3", "s"))
        # Access "a" to move it to end (most recent)
        cache.get("a")
        # Now adding "d" should evict "b" (oldest)
        cache.set("d", ("4", "s"))
        assert cache.get("b") is None
        assert cache.get("a") is not None  # "a" was refreshed

    def test_eviction_count_accumulates(self) -> None:
        cache = InMemoryCache(max_size=2, default_ttl=None)
        cache.set("a", ("1", "s"))
        cache.set("b", ("2", "s"))
        cache.set("c", ("3", "s"))  # evict a
        cache.set("d", ("4", "s"))  # evict b
        assert cache.stats.evictions == 2


# ---------------------------------------------------------------------------
# InMemoryCache: stats tracking
# ---------------------------------------------------------------------------


class TestInMemoryCacheStats:
    def test_miss_increments(self) -> None:
        cache = InMemoryCache()
        cache.get("x")
        cache.get("y")
        assert cache.stats.misses == 2
        assert cache.stats.hits == 0

    def test_hit_increments(self) -> None:
        cache = InMemoryCache()
        cache.set("k", ("v", "s"))
        cache.get("k")
        cache.get("k")
        assert cache.stats.hits == 2
        assert cache.stats.misses == 0

    def test_expired_entry_counts_as_miss(self) -> None:
        cache = InMemoryCache(default_ttl=1)
        cache.set("k", ("v", "s"))
        cache.get("k")  # hit
        time.sleep(1.1)
        cache.get("k")  # miss (expired)
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_hit_rate_with_mixed(self) -> None:
        cache = InMemoryCache()
        cache.set("a", ("v", "s"))
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.stats.hit_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# InMemoryCache: thread safety
# ---------------------------------------------------------------------------


class TestInMemoryCacheThreadSafety:
    def test_concurrent_set_and_get(self) -> None:
        cache = InMemoryCache(max_size=100, default_ttl=None)
        errors: List[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(50):
                    cache.set(f"key-{start}-{i}", (f"val-{i}", "s"))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(100):
                    cache.get("key-0-0")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# CacheKeyBuilder
# ---------------------------------------------------------------------------


class TestCacheKeyBuilder:
    def test_deterministic(self) -> None:
        msgs = [Message(role=Role.USER, content="hello")]
        key1 = CacheKeyBuilder.build(
            model="gpt-4o",
            system_prompt="You are helpful.",
            messages=msgs,
            temperature=0.0,
        )
        key2 = CacheKeyBuilder.build(
            model="gpt-4o",
            system_prompt="You are helpful.",
            messages=msgs,
            temperature=0.0,
        )
        assert key1 == key2

    def test_starts_with_prefix(self) -> None:
        key = CacheKeyBuilder.build(
            model="m",
            system_prompt="s",
            messages=[],
            temperature=0.0,
        )
        assert key.startswith("selectools:")

    def test_different_model_different_key(self) -> None:
        msgs = [Message(role=Role.USER, content="hello")]
        k1 = CacheKeyBuilder.build(
            model="gpt-4o", system_prompt="s", messages=msgs, temperature=0.0
        )
        k2 = CacheKeyBuilder.build(
            model="gpt-4o-mini", system_prompt="s", messages=msgs, temperature=0.0
        )
        assert k1 != k2

    def test_different_messages_different_key(self) -> None:
        k1 = CacheKeyBuilder.build(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="hello")],
            temperature=0.0,
        )
        k2 = CacheKeyBuilder.build(
            model="m",
            system_prompt="s",
            messages=[Message(role=Role.USER, content="world")],
            temperature=0.0,
        )
        assert k1 != k2

    def test_different_temperature_different_key(self) -> None:
        msgs = [Message(role=Role.USER, content="hi")]
        k1 = CacheKeyBuilder.build(model="m", system_prompt="s", messages=msgs, temperature=0.0)
        k2 = CacheKeyBuilder.build(model="m", system_prompt="s", messages=msgs, temperature=0.5)
        assert k1 != k2

    def test_different_system_prompt_different_key(self) -> None:
        msgs = [Message(role=Role.USER, content="hi")]
        k1 = CacheKeyBuilder.build(
            model="m", system_prompt="prompt A", messages=msgs, temperature=0.0
        )
        k2 = CacheKeyBuilder.build(
            model="m", system_prompt="prompt B", messages=msgs, temperature=0.0
        )
        assert k1 != k2

    def test_with_tools_changes_key(self) -> None:
        from selectools import Tool, ToolParameter

        msgs = [Message(role=Role.USER, content="hi")]
        tool = Tool(
            name="search",
            description="Search the web for information",
            parameters=[ToolParameter(name="query", param_type=str, description="Search query")],
            function=lambda query: "result",
        )
        k1 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=msgs, tools=None, temperature=0.0
        )
        k2 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=msgs, tools=[tool], temperature=0.0
        )
        assert k1 != k2

    def test_with_tool_calls_in_messages(self) -> None:
        tc = ToolCall(tool_name="search", parameters={"query": "test"})
        msg_with = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        msg_without = Message(role=Role.ASSISTANT, content="")
        k1 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=[msg_with], temperature=0.0
        )
        k2 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=[msg_without], temperature=0.0
        )
        assert k1 != k2

    def test_empty_messages(self) -> None:
        key = CacheKeyBuilder.build(model="m", system_prompt="s", messages=[], temperature=0.0)
        assert isinstance(key, str)
        assert len(key) > 20
