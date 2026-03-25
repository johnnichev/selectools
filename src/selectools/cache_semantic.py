"""
Semantic cache for LLM responses using embedding-based similarity lookup.

Provides a drop-in replacement for :class:`~selectools.cache.InMemoryCache` that
serves cached responses for semantically equivalent queries even when the exact
prompt wording differs.  Uses cosine similarity over embedding vectors with a
configurable similarity threshold.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from .cache import CacheStats

if TYPE_CHECKING:
    from .embeddings.provider import EmbeddingProvider


# ---------------------------------------------------------------------------
# Internal entry
# ---------------------------------------------------------------------------


@dataclass
class _SemanticEntry:
    """Internal entry storing the original key, its embedding, and the cached value."""

    key: str
    embedding: List[float]
    value: Tuple[Any, Any]
    expires_at: float  # monotonic deadline; 0.0 = never expires


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Thread-safe semantic cache with embedding-based similarity lookup.

    On :meth:`get`, the query key is embedded and compared against all stored
    embeddings using cosine similarity.  If the best match exceeds
    ``similarity_threshold`` and has not expired, the cached value is returned.
    This lets cache hits serve semantically equivalent queries even when the
    exact text differs.

    Implements the same :class:`~selectools.cache.Cache` protocol as
    :class:`~selectools.cache.InMemoryCache` so it can be passed directly to
    ``AgentConfig(cache=SemanticCache(...))``.

    Args:
        embedding_provider: Provider used to embed cache keys.  Any
            :class:`~selectools.embeddings.provider.EmbeddingProvider` works.
        similarity_threshold: Minimum cosine similarity to treat a lookup as a
            hit.  Range: ``[0.0, 1.0]``.  Higher values require closer matches.
            Default: ``0.92``.
        max_size: Maximum number of entries before LRU eviction.  ``0`` means
            unbounded.  Default: ``1000``.
        default_ttl: Default time-to-live in seconds.  ``None`` disables expiry.
            Default: ``None``.

    Example::

        from selectools.cache_semantic import SemanticCache
        from selectools.embeddings.openai import OpenAIEmbeddingProvider

        cache = SemanticCache(
            embedding_provider=OpenAIEmbeddingProvider(),
            similarity_threshold=0.92,
            max_size=500,
            default_ttl=3600,
        )
        config = AgentConfig(cache=cache)
        # "What's the weather in NYC?" hits cache for "Weather in New York City?"
    """

    def __init__(
        self,
        embedding_provider: "EmbeddingProvider",
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_size < 0:
            raise ValueError("max_size must be >= 0")

        self._embedding_provider = embedding_provider
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._default_ttl = default_ttl

        self._entries: List[_SemanticEntry] = []
        self._stats = CacheStats()
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """Embed *key* and return the best-matching cached value, or ``None`` on miss."""
        query_embedding = self._embedding_provider.embed_query(key)
        now = time.monotonic()

        with self._lock:
            best_score = -1.0
            best_idx = -1

            for i, entry in enumerate(self._entries):
                if entry.expires_at and now > entry.expires_at:
                    continue
                score = _cosine_similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score >= self._threshold:
                # Move to end (most-recently used) by re-inserting
                entry = self._entries.pop(best_idx)
                self._entries.append(entry)
                self._stats.hits += 1
                return entry.value

            self._stats.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Tuple[Any, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Embed *key* and store *value* with optional TTL override."""
        embedding = self._embedding_provider.embed_text(key)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (time.monotonic() + effective_ttl) if effective_ttl else 0.0

        with self._lock:
            # Replace existing entry with exact key match
            for i, entry in enumerate(self._entries):
                if entry.key == key:
                    self._entries[i] = _SemanticEntry(
                        key=key,
                        embedding=embedding,
                        value=value,
                        expires_at=expires_at,
                    )
                    self._entries.append(self._entries.pop(i))  # move to end
                    return

            # Evict LRU if at capacity
            if self._max_size and len(self._entries) >= self._max_size:
                self._entries.pop(0)
                self._stats.evictions += 1

            self._entries.append(
                _SemanticEntry(
                    key=key,
                    embedding=embedding,
                    value=value,
                    expires_at=expires_at,
                )
            )

    def delete(self, key: str) -> bool:
        """Remove an entry by exact original key.  Returns ``True`` if found."""
        with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.key == key:
                    self._entries.pop(i)
                    return True
        return False

    def clear(self) -> None:
        """Remove all entries and reset stats."""
        with self._lock:
            self._entries.clear()
            self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Current performance counters (read-only snapshot)."""
        return self._stats

    @property
    def size(self) -> int:
        """Number of entries currently stored (includes expired but not yet pruned)."""
        return len(self._entries)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SemanticCache(size={self.size}, max_size={self._max_size}, "
            f"threshold={self._threshold}, stats={self._stats!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["SemanticCache"]
