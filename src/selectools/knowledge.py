"""
Cross-session knowledge memory — daily logs, persistent facts, and pluggable stores.

Provides durable memory that persists across agent sessions.  The original
file-based backend is preserved for backward compatibility; new code can
use the ``KnowledgeStore`` protocol with ``SQLiteKnowledgeStore`` (or Redis /
Supabase backends in separate modules) for richer querying.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# ======================================================================
# KnowledgeEntry — structured entry for the new store-based API
# ======================================================================


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge stored by the agent.

    Attributes:
        id: Unique identifier (UUID).
        content: The knowledge text.
        category: Tag such as ``"fact"``, ``"preference"``, ``"instruction"``, ``"context"``.
        importance: Score from 0.0 (trivial) to 1.0 (critical).
        persistent: If ``True``, the entry survives importance-based eviction.
        ttl_days: Number of days before the entry expires.  ``None`` means no expiry.
        created_at: When the entry was first stored.
        updated_at: When the entry was last modified.
        metadata: Arbitrary key-value pairs for application-specific data.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    category: str = "general"
    importance: float = 0.5
    persistent: bool = False
    ttl_days: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Whether this entry has passed its TTL."""
        if self.ttl_days is None:
            return False
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > created + timedelta(days=self.ttl_days)


# ======================================================================
# KnowledgeStore — protocol for pluggable backends
# ======================================================================


@runtime_checkable
class KnowledgeStore(Protocol):
    """Protocol for knowledge storage backends.

    Implementations must provide CRUD + query + prune operations.
    """

    def save(self, entry: KnowledgeEntry) -> str:
        """Save or update an entry.  Returns the entry ID."""
        ...

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a single entry by ID."""
        ...

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        """Query entries with optional filters, ordered by importance descending."""
        ...

    def delete(self, entry_id: str) -> bool:
        """Delete an entry.  Returns True if it existed."""
        ...

    def count(self) -> int:
        """Total number of stored entries."""
        ...

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        """Remove expired and low-importance non-persistent entries.  Returns count removed."""
        ...


# ======================================================================
# FileKnowledgeStore — wraps existing file-based logic
# ======================================================================


class FileKnowledgeStore:
    """File-based knowledge store (backward-compatible with the original KnowledgeMemory).

    Entries are stored as JSON lines in a ``entries.jsonl`` file, plus the
    legacy daily-log and MEMORY.md formats.

    Args:
        directory: Base directory for knowledge files.  Created if absent.
    """

    def __init__(self, directory: str = "./memory") -> None:
        self._directory = directory
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    @property
    def _entries_path(self) -> str:
        return os.path.join(self._directory, "entries.jsonl")

    def _load_all(self) -> List[KnowledgeEntry]:
        if not os.path.exists(self._entries_path):
            return []
        entries: List[KnowledgeEntry] = []
        with open(self._entries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    created_at = datetime.fromisoformat(d["created_at"])
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    updated_at = datetime.fromisoformat(d["updated_at"])
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    entries.append(
                        KnowledgeEntry(
                            id=d["id"],
                            content=d["content"],
                            category=d.get("category", "general"),
                            importance=d.get("importance", 0.5),
                            persistent=d.get("persistent", False),
                            ttl_days=d.get("ttl_days"),
                            created_at=created_at,
                            updated_at=updated_at,
                            metadata=d.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return entries

    def _save_all(self, entries: List[KnowledgeEntry]) -> None:
        tmp_path = self._entries_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in entries:
                d = {
                    "id": e.id,
                    "content": e.content,
                    "category": e.category,
                    "importance": e.importance,
                    "persistent": e.persistent,
                    "ttl_days": e.ttl_days,
                    "created_at": e.created_at.isoformat(),
                    "updated_at": e.updated_at.isoformat(),
                    "metadata": e.metadata,
                }
                f.write(json.dumps(d) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._entries_path)

    def save(self, entry: KnowledgeEntry) -> str:
        with self._lock:
            entries = self._load_all()
            # Update existing or append
            for i, e in enumerate(entries):
                if e.id == entry.id:
                    entries[i] = entry
                    self._save_all(entries)
                    return entry.id
            entries.append(entry)
            self._save_all(entries)
        return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        with self._lock:
            for e in self._load_all():
                if e.id == entry_id:
                    return e
        return None

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        with self._lock:
            entries = self._load_all()
        # Normalize naive since to UTC-aware to match stored entry datetimes.
        since_aware: Optional[datetime] = None
        if since is not None:
            since_aware = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        result = []
        for e in entries:
            if e.is_expired:
                continue
            if category and e.category != category:
                continue
            if e.importance < min_importance:
                continue
            if since_aware and e.created_at < since_aware:
                continue
            result.append(e)
        result.sort(key=lambda x: x.importance, reverse=True)
        return result[:limit]

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._load_all()
            before = len(entries)
            entries = [e for e in entries if e.id != entry_id]
            if len(entries) < before:
                self._save_all(entries)
                return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._load_all())

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        with self._lock:
            entries = self._load_all()
            before = len(entries)
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=max_age_days) if max_age_days else None
            )
            kept = []
            for e in entries:
                if e.persistent:
                    kept.append(e)
                    continue
                if e.is_expired:
                    continue
                if cutoff and e.created_at < cutoff:
                    continue
                if e.importance < min_importance:
                    continue
                kept.append(e)
            self._save_all(kept)
            return before - len(kept)


# ======================================================================
# SQLiteKnowledgeStore
# ======================================================================


class SQLiteKnowledgeStore:
    """SQLite-backed knowledge store for production single-process use.

    Args:
        db_path: Path to the SQLite database file.  Created if absent.
    """

    def __init__(self, db_path: str = "knowledge.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    importance REAL DEFAULT 0.5,
                    persistent INTEGER DEFAULT 0,
                    ttl_days INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_importance ON knowledge(importance)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at)"
            )

    def _row_to_entry(self, row: tuple) -> KnowledgeEntry:
        created_at = datetime.fromisoformat(row[6])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        updated_at = datetime.fromisoformat(row[7])
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return KnowledgeEntry(
            id=row[0],
            content=row[1],
            category=row[2],
            importance=row[3],
            persistent=bool(row[4]),
            ttl_days=row[5],
            created_at=created_at,
            updated_at=updated_at,
            metadata=json.loads(row[8]) if row[8] else {},
        )

    def save(self, entry: KnowledgeEntry) -> str:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO knowledge
                   (id, content, category, importance, persistent, ttl_days,
                    created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id,
                    entry.content,
                    entry.category,
                    entry.importance,
                    int(entry.persistent),
                    entry.ttl_days,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    json.dumps(entry.metadata),
                ),
            )
        return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,)).fetchone()
            return self._row_to_entry(row) if row else None

    def query(
        self,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[KnowledgeEntry]:
        clauses = ["importance >= ?"]
        params: List[Any] = [min_importance]
        if category:
            clauses.append("category = ?")
            params.append(category)
        if since:
            # Normalize naive since to UTC-aware so the ISO string comparison
            # against stored '+00:00' timestamps is lexicographically correct.
            since_aware = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
            clauses.append("created_at >= ?")
            params.append(since_aware.isoformat())
        where = " AND ".join(clauses)
        # LIMIT is applied in Python after TTL filtering to avoid returning
        # fewer results than requested when expired entries are present.
        sql = f"SELECT * FROM knowledge WHERE {where} ORDER BY importance DESC"  # nosec B608
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        return [e for e in entries if not e.is_expired][:limit]

    def delete(self, entry_id: str) -> bool:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()
            return row[0] if row else 0

    def prune(
        self,
        max_age_days: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        removed = 0
        with self._lock, sqlite3.connect(self._db_path) as conn:
            # Remove expired TTL entries (non-persistent)
            rows = conn.execute(
                "SELECT id, ttl_days, created_at, persistent FROM knowledge"
            ).fetchall()
            now = datetime.now(timezone.utc)
            for row_id, ttl, created, persistent in rows:
                if persistent:
                    continue
                if ttl is not None:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    if now > created_dt + timedelta(days=ttl):
                        conn.execute("DELETE FROM knowledge WHERE id = ?", (row_id,))
                        removed += 1
                        continue
                if max_age_days is not None:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    if now > created_dt + timedelta(days=max_age_days):
                        conn.execute("DELETE FROM knowledge WHERE id = ?", (row_id,))
                        removed += 1
            # Remove low-importance non-persistent
            if min_importance > 0:
                cursor = conn.execute(
                    "DELETE FROM knowledge WHERE persistent = 0 AND importance < ?",
                    (min_importance,),
                )
                removed += cursor.rowcount
        return removed


# ======================================================================
# KnowledgeMemory — main public class (backward-compatible)
# ======================================================================


class KnowledgeMemory:
    """Maintains cross-session knowledge with daily logs and persistent facts.

    Supports two modes:

    1. **Legacy file mode** (default): Daily log files and ``MEMORY.md``,
       matching the original API.
    2. **Store mode**: Pass a ``KnowledgeStore`` for structured entries with
       importance scoring, TTL, and pluggable backends.

    Args:
        directory: Base directory for knowledge files.  Created if absent.
        store: Optional pluggable backend.  If ``None``, uses ``FileKnowledgeStore``.
        recent_days: Number of recent days to include in context.  Default: 2.
        max_context_chars: Maximum characters to include in context output.
        max_entries: Maximum entries before importance-based eviction.  Default: 50.
    """

    def __init__(
        self,
        directory: str = "./memory",
        store: Optional[KnowledgeStore] = None,
        recent_days: int = 2,
        max_context_chars: int = 5000,
        max_entries: int = 50,
    ) -> None:
        self._directory = directory
        self._store = store or FileKnowledgeStore(directory)
        self._recent_days = recent_days
        self._max_context_chars = max_context_chars
        self._max_entries = max_entries
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    @property
    def directory(self) -> str:
        """Base directory for knowledge files."""
        return self._directory

    @property
    def store(self) -> KnowledgeStore:
        """The underlying knowledge store."""
        return self._store

    def remember(
        self,
        content: str,
        category: str = "general",
        persistent: bool = False,
        importance: float = 0.5,
        ttl_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a piece of knowledge.

        Args:
            content: The text to remember.
            category: Category tag (e.g. ``"fact"``, ``"preference"``).
            persistent: If True, survives importance-based eviction.
            importance: Score from 0.0 to 1.0.
            ttl_days: Days before this entry expires.  ``None`` = no expiry.
            metadata: Arbitrary key-value pairs.

        Returns:
            The entry ID.
        """
        entry = KnowledgeEntry(
            content=content,
            category=category,
            persistent=persistent,
            importance=importance,
            ttl_days=ttl_days,
            metadata=metadata or {},
        )
        entry_id = self._store.save(entry)

        # Also write to legacy daily log for backward compat
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{category}] {content}"
        today = now.strftime("%Y-%m-%d")
        log_path = os.path.join(self._directory, f"{today}.log")
        with self._lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
            if persistent:
                mem_path = os.path.join(self._directory, "MEMORY.md")
                with open(mem_path, "a", encoding="utf-8") as f:
                    f.write(f"- [{category}] {content}\n")

        # Importance-based eviction
        self._enforce_max_entries()

        return entry_id

    def _enforce_max_entries(self) -> None:
        """Evict lowest-importance non-persistent entries if over max_entries.

        Uses the live (non-expired) entry count from ``query()`` rather than
        ``count()`` so that entries already past their TTL do not trigger
        unnecessary eviction of valid entries.
        """
        # Use a large limit to fetch all live (non-expired) entries.
        all_entries = self._store.query(limit=self._max_entries + 1000)
        current = len(all_entries)
        if current <= self._max_entries:
            return
        all_entries.sort(key=lambda e: e.importance)
        to_remove = current - self._max_entries
        removed = 0
        for e in all_entries:
            if removed >= to_remove:
                break
            if e.persistent:
                continue
            self._store.delete(e.id)
            removed += 1

    def get_recent_logs(self, days: Optional[int] = None) -> str:
        """Read recent daily log entries.

        Args:
            days: Number of recent days to read.  Defaults to ``recent_days``.

        Returns:
            Combined text from recent daily log files.
        """
        days = days or self._recent_days
        lines: List[str] = []

        for i in range(days):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            log_path = os.path.join(self._directory, f"{date}.log")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        lines.append(f"=== {date} ===")
                        lines.append(content)

        return "\n".join(lines)

    def get_persistent_facts(self) -> str:
        """Read persistent facts from MEMORY.md.

        Returns:
            Contents of MEMORY.md, or empty string if not found.
        """
        mem_path = os.path.join(self._directory, "MEMORY.md")
        if not os.path.exists(mem_path):
            return ""
        with open(mem_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def build_context(self) -> str:
        """Build a context string for prompt injection.

        Uses the store-based entries (sorted by importance) if available,
        falling back to legacy file format.

        Returns:
            A formatted context block, or empty string if no data.
        """
        parts: List[str] = []

        # Try store-based entries first
        store_entries = self._store.query(limit=self._max_entries)
        if store_entries:
            high = [e for e in store_entries if e.importance >= 0.7]
            rest = [e for e in store_entries if e.importance < 0.7]
            if high:
                parts.append("[Important Knowledge]")
                for e in high:
                    tag = f"[{e.category}]" if e.category != "general" else ""
                    parts.append(f"- {tag} {e.content}".strip())
            if rest:
                if parts:
                    parts.append("")
                parts.append("[General Knowledge]")
                for e in rest:
                    tag = f"[{e.category}]" if e.category != "general" else ""
                    parts.append(f"- {tag} {e.content}".strip())
        else:
            # Fall back to legacy format
            persistent = self.get_persistent_facts()
            if persistent:
                parts.append("[Long-term Memory]")
                parts.append(persistent)

            recent = self.get_recent_logs()
            if recent:
                if parts:
                    parts.append("")
                parts.append("[Recent Memory]")
                parts.append(recent)

        if not parts:
            return ""

        context = "\n".join(parts)
        if len(context) > self._max_context_chars:
            suffix = "\n... (truncated)"
            context = context[: self._max_context_chars - len(suffix)] + suffix
        return context

    def prune_old_logs(self, keep_days: Optional[int] = None) -> int:
        """Remove daily log files older than ``keep_days``.

        Args:
            keep_days: Number of days to keep.  Defaults to ``recent_days``.

        Returns:
            Number of log files removed.
        """
        keep_days = keep_days or self._recent_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        removed = 0

        for filename in os.listdir(self._directory):
            if not filename.endswith(".log"):
                continue
            date_str = filename[:-4]
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    os.remove(os.path.join(self._directory, filename))
                    removed += 1
            except ValueError:
                continue

        return removed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self._directory,
            "recent_days": self._recent_days,
            "max_context_chars": self._max_context_chars,
            "max_entries": self._max_entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeMemory":
        return cls(
            directory=data.get("directory", "./memory"),
            recent_days=data.get("recent_days", 2),
            max_context_chars=data.get("max_context_chars", 5000),
            max_entries=data.get("max_entries", 50),
        )


__all__ = [
    "KnowledgeEntry",
    "KnowledgeStore",
    "FileKnowledgeStore",
    "SQLiteKnowledgeStore",
    "KnowledgeMemory",
]
