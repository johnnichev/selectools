"""
Redis-backed knowledge store for distributed deployments.

Requires the ``redis`` package::

    pip install selectools[cache]
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeEntry


class RedisKnowledgeStore:
    """Redis-backed knowledge store for distributed or multi-process use.

    Entries are stored as JSON in Redis hashes keyed by ``{prefix}:{id}``.
    A sorted set ``{prefix}:importance`` indexes entries by importance score,
    and per-category sets ``{prefix}:category:{cat}`` enable category-based
    lookups.

    Args:
        redis_client: A ``redis.Redis`` (or compatible) client instance.
        prefix: Key prefix to namespace knowledge entries.
    """

    def __init__(self, redis_client: Any, prefix: str = "knowledge") -> None:
        self._client = redis_client
        self._prefix = prefix

    # -- key helpers -------------------------------------------------------

    def _entry_key(self, entry_id: str) -> str:
        return f"{self._prefix}:{entry_id}"

    def _importance_key(self) -> str:
        return f"{self._prefix}:importance"

    def _category_key(self, category: str) -> str:
        return f"{self._prefix}:category:{category}"

    def _all_ids_key(self) -> str:
        return f"{self._prefix}:all_ids"

    # -- serialization -----------------------------------------------------

    @staticmethod
    def _entry_to_dict(entry: KnowledgeEntry) -> Dict[str, str]:
        return {
            "id": entry.id,
            "content": entry.content,
            "category": entry.category,
            "importance": str(entry.importance),
            "persistent": "1" if entry.persistent else "0",
            "ttl_days": str(entry.ttl_days) if entry.ttl_days is not None else "",
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "metadata": json.dumps(entry.metadata),
        }

    @staticmethod
    def _dict_to_entry(data: Dict[str, str]) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=data["id"],
            content=data["content"],
            category=data.get("category", "general"),
            importance=float(data.get("importance", "0.5")),
            persistent=data.get("persistent") == "1",
            ttl_days=int(data["ttl_days"]) if data.get("ttl_days") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=json.loads(data.get("metadata", "{}")),
        )

    # -- protocol methods --------------------------------------------------

    def save(self, entry: KnowledgeEntry) -> str:
        """Save or update an entry.  Returns the entry ID."""
        key = self._entry_key(entry.id)

        # Read old category before pipeline (reduces TOCTOU window)
        existing_raw: Optional[str] = self._client.hget(key, "category")

        pipe = self._client.pipeline()
        if existing_raw is not None and existing_raw != entry.category:
            pipe.srem(self._category_key(existing_raw), entry.id)
        pipe.hset(key, mapping=self._entry_to_dict(entry))
        pipe.zadd(self._importance_key(), {entry.id: entry.importance})
        pipe.sadd(self._category_key(entry.category), entry.id)
        pipe.sadd(self._all_ids_key(), entry.id)
        pipe.execute()

        return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a single entry by ID."""
        data: Dict[str, str] = self._client.hgetall(self._entry_key(entry_id))
        if not data:
            return None
        return self._dict_to_entry(data)

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Query entries with optional filters, ordered by importance descending."""
        if category is not None:
            candidate_ids: List[str] = list(self._client.smembers(self._category_key(category)))
        else:
            # Use sorted set to get IDs by importance descending
            candidate_ids = self._client.zrevrangebyscore(
                self._importance_key(), "+inf", str(min_importance)
            )

        entries: List[KnowledgeEntry] = []
        for eid in candidate_ids:
            entry = self.get(eid)
            if entry is None:
                continue
            if entry.is_expired:
                continue
            if entry.importance < min_importance:
                continue
            if since is not None and entry.created_at < since:
                continue
            entries.append(entry)

        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:limit]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry.  Returns True if it existed."""
        key = self._entry_key(entry_id)
        data: Dict[str, str] = self._client.hgetall(key)
        if not data:
            return False

        category = data.get("category", "general")
        pipe = self._client.pipeline()
        pipe.delete(key)
        pipe.zrem(self._importance_key(), entry_id)
        pipe.srem(self._category_key(category), entry_id)
        pipe.srem(self._all_ids_key(), entry_id)
        pipe.execute()
        return True

    def count(self) -> int:
        """Total entries.

        May include stale entries not yet pruned.
        Call ``prune()`` for an accurate count.
        """
        result: int = self._client.scard(self._all_ids_key())
        return result

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        """Remove expired and low-importance non-persistent entries.  Returns count removed."""
        all_ids: List[str] = list(self._client.smembers(self._all_ids_key()))
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days) if max_age_days else None
        removed = 0

        for eid in all_ids:
            entry = self.get(eid)
            if entry is None:
                continue
            if entry.persistent:
                continue

            should_remove = False

            # Expired by TTL
            if entry.is_expired:
                should_remove = True
            # Older than max_age_days
            elif cutoff is not None and entry.created_at < cutoff:
                should_remove = True
            # Below minimum importance
            elif min_importance > 0 and entry.importance < min_importance:
                should_remove = True

            if should_remove:
                self.delete(eid)
                removed += 1

        return removed


__all__ = ["RedisKnowledgeStore"]
