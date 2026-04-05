"""
Supabase-backed knowledge store for cloud-native deployments.

Requires the ``supabase`` package::

    pip install supabase
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeEntry

_logger = logging.getLogger(__name__)


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

    # -- serialization -----------------------------------------------------

    @staticmethod
    def _entry_to_row(entry: KnowledgeEntry) -> Dict[str, Any]:
        return {
            "id": entry.id,
            "content": entry.content,
            "category": entry.category,
            "importance": entry.importance,
            "persistent": entry.persistent,
            "ttl_days": entry.ttl_days,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "metadata": entry.metadata,
        }

    @staticmethod
    def _row_to_entry(row: Dict[str, Any]) -> KnowledgeEntry:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        updated_at = datetime.fromisoformat(row["updated_at"])
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return KnowledgeEntry(
            id=row["id"],
            content=row["content"],
            category=row.get("category", "general"),
            importance=float(row.get("importance", 0.5)),
            persistent=bool(row.get("persistent", False)),
            ttl_days=row.get("ttl_days"),
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

    # -- protocol methods --------------------------------------------------

    def save(self, entry: KnowledgeEntry) -> str:
        """Save or update an entry.  Returns the entry ID."""
        try:
            row = self._entry_to_row(entry)
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
            return self._row_to_entry(response.data[0])
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

            entries = [self._row_to_entry(row) for row in (response.data or [])]
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
                entry = self._row_to_entry(row)
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


__all__ = ["SupabaseKnowledgeStore"]
