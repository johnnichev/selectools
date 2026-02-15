"""
Redis-backed response cache for distributed deployments.

Requires the ``redis`` package::

    pip install selectools[cache]
"""

from __future__ import annotations

import pickle  # nosec B403 - we only deserialize data we serialized ourselves
from typing import Any, Optional, Tuple

from .cache import CacheStats


class RedisCache:
    """
    Distributed TTL cache backed by Redis.

    Each entry is stored as a pickled ``(Message, UsageStats)`` tuple with a
    server-side TTL managed by Redis.  Stats (hits / misses) are tracked
    in-process; eviction counting is not available since Redis manages
    expiry independently.

    Args:
        url: Redis connection URL, e.g. ``redis://localhost:6379/0``.
        prefix: Key prefix to namespace selectools entries.
        default_ttl: Default time-to-live in seconds.  ``None`` means entries
            persist until explicitly deleted.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "selectools:",
        default_ttl: Optional[int] = 300,
    ) -> None:
        try:
            import redis as redis_lib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RedisCache requires the 'redis' package. "
                "Install it with: pip install selectools[cache]"
            ) from exc

        self._client: Any = redis_lib.from_url(url, decode_responses=False)
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._stats = CacheStats()

    # -- helpers -----------------------------------------------------------

    def _full_key(self, key: str) -> str:
        if key.startswith(self._prefix):
            return key
        return f"{self._prefix}{key}"

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """Retrieve a cached value, returning ``None`` on miss."""
        raw: Optional[bytes] = self._client.get(self._full_key(key))
        if raw is None:
            self._stats.misses += 1
            return None
        self._stats.hits += 1
        return pickle.loads(raw)  # type: ignore[no-any-return]  # nosec B301

    def set(
        self,
        key: str,
        value: Tuple[Any, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value with optional TTL override."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        full_key = self._full_key(key)
        data = pickle.dumps(value)
        if effective_ttl:
            self._client.setex(full_key, effective_ttl, data)
        else:
            self._client.set(full_key, data)

    def delete(self, key: str) -> bool:
        """Remove a key.  Returns ``True`` if it existed."""
        removed: int = self._client.delete(self._full_key(key))
        return removed > 0

    def clear(self) -> None:
        """Remove all selectools-prefixed keys and reset stats."""
        cursor: int = 0
        pattern = f"{self._prefix}*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Current hit / miss counters (in-process only)."""
        return self._stats

    def __repr__(self) -> str:
        """Return a human-readable summary of the Redis cache configuration."""
        return f"RedisCache(prefix={self._prefix!r}, default_ttl={self._default_ttl})"


__all__ = ["RedisCache"]
