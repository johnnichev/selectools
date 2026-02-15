"""
Response caching for LLM provider calls.

Provides a Cache protocol, an in-memory LRU+TTL implementation, and a
deterministic cache-key builder so that identical requests can be served
from cache instead of hitting the LLM API.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from .tools.base import Tool
    from .types import Message
    from .usage import UsageStats


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Tracks cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        """Return a human-readable summary of cache statistics."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# CacheEntry  (internal)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Internal entry wrapping a cached value with an expiry timestamp."""

    value: Tuple[Any, Any]  # (Message, UsageStats)
    expires_at: float  # time.monotonic() deadline; 0.0 = never


# ---------------------------------------------------------------------------
# Cache Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Cache(Protocol):
    """
    Abstract cache interface.

    Any object satisfying this protocol can be passed to ``AgentConfig(cache=...)``.
    """

    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """Return cached (Message, UsageStats) or ``None`` on miss / expiry."""
        ...

    def set(
        self,
        key: str,
        value: Tuple[Any, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a (Message, UsageStats) pair, optionally overriding the default TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Remove a key.  Return ``True`` if the key existed."""
        ...

    def clear(self) -> None:
        """Remove all entries."""
        ...

    @property
    def stats(self) -> CacheStats:
        """Current hit / miss / eviction counters."""
        ...


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


class InMemoryCache:
    """
    Thread-safe in-memory LRU cache with per-entry TTL.

    Uses :class:`collections.OrderedDict` for O(1) LRU operations and
    monotonic timestamps for reliable expiry checks.

    Args:
        max_size: Maximum number of entries.  When exceeded the least-recently
            used entry is evicted.  ``0`` means unbounded (not recommended).
        default_ttl: Default time-to-live in **seconds**.  ``None`` disables
            automatic expiry (entries stay until evicted by LRU).
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 300,
    ) -> None:
        if max_size < 0:
            raise ValueError("max_size must be >= 0")
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Tuple[Any, Any]]:
        """Retrieve a cached value, returning ``None`` on miss or expiry."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL expiry
            if entry.expires_at and time.monotonic() > entry.expires_at:
                del self._data[key]
                self._stats.misses += 1
                return None

            # Move to end (most-recently used)
            self._data.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Tuple[Any, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value, evicting LRU entries if necessary."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (time.monotonic() + effective_ttl) if effective_ttl else 0.0

        with self._lock:
            if key in self._data:
                # Update existing entry and move to end
                self._data[key] = _CacheEntry(value=value, expires_at=expires_at)
                self._data.move_to_end(key)
            else:
                # Evict if at capacity
                if self._max_size and len(self._data) >= self._max_size:
                    self._data.popitem(last=False)  # Remove oldest
                    self._stats.evictions += 1
                self._data[key] = _CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """Remove a specific key.  Returns ``True`` if it was present."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries and reset stats."""
        with self._lock:
            self._data.clear()
            self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Current performance counters (read-only snapshot)."""
        return self._stats

    @property
    def size(self) -> int:
        """Number of entries currently stored (includes expired but not yet pruned)."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a human-readable summary of the cache state."""
        return (
            f"InMemoryCache(size={self.size}, max_size={self._max_size}, "
            f"default_ttl={self._default_ttl}, stats={self._stats!r})"
        )


# ---------------------------------------------------------------------------
# CacheKeyBuilder
# ---------------------------------------------------------------------------


class CacheKeyBuilder:
    """
    Builds deterministic cache keys from LLM request parameters.

    The key is a SHA-256 hex digest of a canonical JSON representation of
    (model, system_prompt, messages, tools, temperature).
    """

    @staticmethod
    def build(
        *,
        model: str,
        system_prompt: str,
        messages: List["Message"],
        tools: Optional[List["Tool"]] = None,
        temperature: float = 0.0,
    ) -> str:
        """Return a deterministic hex-digest key for the given request params."""
        messages_data: List[Dict[str, Any]] = []
        for msg in messages:
            entry: Dict[str, Any] = {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content or "",
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"tool_name": tc.tool_name, "parameters": tc.parameters}
                    for tc in msg.tool_calls
                ]
            messages_data.append(entry)

        tools_data: List[Dict[str, Any]] = []
        if tools:
            for t in tools:
                tools_data.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": (
                                    p.param_type.__name__
                                    if hasattr(p.param_type, "__name__")
                                    else str(p.param_type)
                                ),
                                "required": p.required,
                            }
                            for p in t.parameters
                        ],
                    }
                )

        payload: Dict[str, Any] = {
            "model": model,
            "system_prompt": system_prompt,
            "messages": messages_data,
            "tools": tools_data,
            "temperature": temperature,
        }

        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"selectools:{digest}"


__all__ = [
    "Cache",
    "CacheStats",
    "CacheKeyBuilder",
    "InMemoryCache",
]
