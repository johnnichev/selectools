"""
Supabase-backed knowledge store for cloud-native deployments.

Requires the ``supabase`` package::

    pip install supabase
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from .knowledge import KnowledgeEntry, entry_from_jsonable, entry_to_jsonable
from .stability import beta

_logger = logging.getLogger(__name__)


@beta
class SupabaseKnowledgeStore:
    """Supabase-backed knowledge store for cloud-native or multi-service use.

    Entries are stored in a Supabase table whose columns match the
    ``KnowledgeEntry`` fields.  The table must already exist with the
    following schema::

        CREATE TABLE knowledge (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            importance FLOAT DEFAULT 0.5,
            persistent BOOLEAN DEFAULT FALSE,
            ttl_days INTEGER,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            metadata JSONB DEFAULT '{}'
        );

    Args:
        client: A ``supabase.Client`` instance.
        table_name: Name of the Supabase table.  Default: ``"knowledge"``.
    """

    def __init__(self, client: Any, table_name: str = "knowledge") -> None:
        self._client = client
        self._table = table_name

    # -- protocol methods --------------------------------------------------

    def save(self, entry: KnowledgeEntry) -> str:
        """Save or update an entry.  Returns the entry ID."""
        try:
            row = entry_to_jsonable(entry)
            self._client.table(self._table).upsert(row).execute()
            return entry.id
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.save failed: %s", exc)
            return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a single entry by ID."""
        try:
            response = self._client.table(self._table).select("*").eq("id", entry_id).execute()
            if not response.data:
                return None
            return entry_from_jsonable(response.data[0])
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.get failed: %s", exc)
            return None

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Query entries with optional filters, ordered by importance descending."""
        try:
            builder = self._client.table(self._table).select("*")
            builder = builder.gte("importance", min_importance)
            if category is not None:
                builder = builder.eq("category", category)
            if since is not None:
                # Normalize naive since to UTC-aware for correct ISO string comparison.
                since_aware = (
                    since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
                )
                builder = builder.gte("created_at", since_aware.isoformat())
            builder = builder.order("importance", desc=True).limit(limit)
            response = builder.execute()

            entries = [entry_from_jsonable(row) for row in (response.data or [])]
            return [e for e in entries if not e.is_expired]
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.query failed: %s", exc)
            return []

    def delete(self, entry_id: str) -> bool:
        """Delete an entry.  Returns True if it existed."""
        try:
            response = self._client.table(self._table).delete().eq("id", entry_id).execute()
            return bool(response.data)
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.delete failed: %s", exc)
            return False

    def count(self) -> int:
        """Total number of stored entries."""
        try:
            response = self._client.table(self._table).select("id", count="exact").execute()
            return response.count if response.count is not None else 0
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.count failed: %s", exc)
            return 0

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        """Remove expired and low-importance non-persistent entries.  Returns count removed."""
        try:
            # Fetch all non-persistent entries to evaluate locally
            response = self._client.table(self._table).select("*").eq("persistent", False).execute()
            rows = response.data or []

            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=max_age_days) if max_age_days is not None else None
            removed = 0

            for row in rows:
                entry = entry_from_jsonable(row)
                should_remove = False

                if entry.is_expired:
                    should_remove = True
                elif cutoff is not None and entry.created_at < cutoff:
                    should_remove = True
                elif min_importance > 0 and entry.importance < min_importance:
                    should_remove = True

                if should_remove:
                    self._client.table(self._table).delete().eq("id", entry.id).execute()
                    removed += 1

            return removed
        except Exception as exc:
            _logger.warning("SupabaseKnowledgeStore.prune failed: %s", exc)
            return 0


__stability__ = "beta"

__all__ = ["SupabaseKnowledgeStore"]
