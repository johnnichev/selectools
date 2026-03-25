"""Tests for SemanticCache."""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock

import pytest

from selectools.cache_semantic import SemanticCache, _cosine_similarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(embeddings: dict[str, list[float]]) -> MagicMock:
    """Return a mock EmbeddingProvider that maps text → fixed vector."""
    provider = MagicMock()

    def embed_text(text: str) -> list[float]:
        return embeddings.get(text, [0.0, 0.0, 1.0])

    def embed_query(text: str) -> list[float]:
        return embeddings.get(text, [0.0, 0.0, 1.0])

    provider.embed_text.side_effect = embed_text
    provider.embed_query.side_effect = embed_query
    return provider


VALUE_A = ("response_A", None)
VALUE_B = ("response_B", None)

EMBED_NYC = [1.0, 0.0, 0.0]
EMBED_NY = [0.99, 0.141, 0.0]  # close to NYC
EMBED_PARIS = [0.0, 0.0, 1.0]  # far from NYC


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_negative():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# SemanticCache construction
# ---------------------------------------------------------------------------


def test_invalid_threshold_raises():
    provider = MagicMock()
    with pytest.raises(ValueError):
        SemanticCache(provider, similarity_threshold=1.5)


def test_invalid_max_size_raises():
    provider = MagicMock()
    with pytest.raises(ValueError):
        SemanticCache(provider, max_size=-1)


def test_initial_state():
    cache = SemanticCache(MagicMock(), max_size=10)
    assert cache.size == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


# ---------------------------------------------------------------------------
# get / set
# ---------------------------------------------------------------------------


def test_exact_key_hit():
    embeddings = {"nyc": EMBED_NYC}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, similarity_threshold=0.9, max_size=10)

    cache.set("nyc", VALUE_A)
    result = cache.get("nyc")

    assert result == VALUE_A
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


def test_semantic_hit():
    embeddings = {"weather in nyc": EMBED_NYC, "weather in new york": EMBED_NY}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, similarity_threshold=0.9, max_size=10)

    cache.set("weather in nyc", VALUE_A)
    result = cache.get("weather in new york")

    assert result == VALUE_A
    assert cache.stats.hits == 1


def test_semantic_miss_below_threshold():
    embeddings = {"nyc": EMBED_NYC, "paris": EMBED_PARIS}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, similarity_threshold=0.9, max_size=10)

    cache.set("nyc", VALUE_A)
    result = cache.get("paris")

    assert result is None
    assert cache.stats.misses == 1


def test_miss_on_empty_cache():
    provider = _mock_provider({"x": [1.0, 0.0]})
    cache = SemanticCache(provider, max_size=10)
    assert cache.get("x") is None
    assert cache.stats.misses == 1


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


def test_entry_expires():
    embeddings = {"key": [1.0, 0.0]}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, default_ttl=1, max_size=10)

    cache.set("key", VALUE_A)
    assert cache.get("key") == VALUE_A

    time.sleep(1.1)
    assert cache.get("key") is None


def test_no_ttl_entry_persists():
    embeddings = {"key": [1.0, 0.0]}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, default_ttl=None, max_size=10)

    cache.set("key", VALUE_A)
    time.sleep(0.1)
    assert cache.get("key") == VALUE_A


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_lru_eviction():
    provider = MagicMock()
    provider.embed_text.side_effect = lambda t: [float(ord(t[0])), 0.0]
    provider.embed_query.side_effect = lambda t: [float(ord(t[0])), 0.0]
    cache = SemanticCache(provider, similarity_threshold=0.99, max_size=2)

    cache.set("a", ("val_a", None))
    cache.set("b", ("val_b", None))
    # "a" is LRU — adding "c" should evict "a"
    cache.set("c", ("val_c", None))

    assert cache.size == 2
    assert cache.stats.evictions == 1


# ---------------------------------------------------------------------------
# delete / clear
# ---------------------------------------------------------------------------


def test_delete_existing_key():
    embeddings = {"key": [1.0, 0.0]}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, max_size=10)

    cache.set("key", VALUE_A)
    assert cache.delete("key") is True
    assert cache.size == 0


def test_delete_nonexistent_key():
    cache = SemanticCache(MagicMock(), max_size=10)
    assert cache.delete("nope") is False


def test_clear_resets_everything():
    embeddings = {"k": [1.0, 0.0]}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, max_size=10)

    cache.set("k", VALUE_A)
    cache.get("k")
    cache.clear()

    assert cache.size == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_set_and_get():
    embeddings = {str(i): [float(i), 0.0, 0.0] for i in range(20)}
    provider = _mock_provider(embeddings)
    cache = SemanticCache(provider, similarity_threshold=0.99, max_size=100)

    errors: list[Exception] = []

    def writer():
        for i in range(10):
            try:
                cache.set(str(i), (f"val_{i}", None))
            except Exception as e:
                errors.append(e)

    def reader():
        for i in range(10):
            try:
                cache.get(str(i))
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr():
    cache = SemanticCache(MagicMock(), max_size=42)
    assert "SemanticCache" in repr(cache)
    assert "42" in repr(cache)
