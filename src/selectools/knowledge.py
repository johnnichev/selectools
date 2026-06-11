"""
Cross-session knowledge memory — daily logs, persistent facts, and pluggable stores.

Provides durable memory that persists across agent sessions.  The original
file-based backend is preserved for backward compatibility; new code can
use the ``KnowledgeStore`` protocol with ``SQLiteKnowledgeStore`` (or Redis /
Supabase backends in separate modules) for richer querying.

For ephemeral infrastructure (Railway, Lambda, Cloud Run) where the local
filesystem is wiped between deploys, pass a ``KnowledgeBackend`` to
``KnowledgeMemory``.  The directory remains the fast scratch space during
the request; the backend persists a snapshot of it between requests::

    from selectools import KnowledgeMemory
    from selectools.knowledge_backends import SupabaseKnowledgeBackend

    memory = KnowledgeMemory(
        directory="/tmp/agent-memory",
        backend=SupabaseKnowledgeBackend(client, key="user-123"),
    )
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from .knowledge_sanitizers import dedupe_against
from .stability import beta, register_stability, stable

_logger = logging.getLogger(__name__)

PreSaveHook = Callable[[str], Optional[str]]
register_stability("PreSaveHook", "beta")
"""A pre-save sanitizer: returns transformed text, or ``None`` to reject the entry."""

# ======================================================================
# KnowledgeEntry — structured entry for the new store-based API
# ======================================================================


@stable
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
# KnowledgeBackend — protocol for blob persistence of the memory directory
# ======================================================================


@beta
class KnowledgeBackend(Protocol):
    """Protocol for persisting the knowledge directory as a single blob.

    The contract is a single opaque byte payload: ``KnowledgeMemory`` packs
    its scratch directory into one blob after every mutation and restores it
    on construction.  Implementations only need to store and retrieve that
    blob — they never interpret its contents.

    Built-in adapters live in ``selectools.knowledge_backends``
    (``SupabaseKnowledgeBackend``, ``RedisKnowledgeBackend``).
    """

    def load_bytes(self) -> Optional[bytes]:
        """Return the stored blob, or ``None`` if nothing has been saved yet."""
        ...

    def save_bytes(self, data: bytes) -> None:
        """Persist *data*, replacing any previously stored blob."""
        ...


_ARCHIVE_VERSION = 1


def _pack_directory(directory: str) -> bytes:
    """Pack every file under *directory* into a JSON archive blob.

    Layout: ``{"version": 1, "files": {relpath: {"encoding", "content"}}}``.
    Text files use ``utf-8`` encoding; anything else falls back to
    ``base64``.  In-flight ``*.tmp`` files from atomic writes are skipped.
    """
    files: Dict[str, Dict[str, str]] = {}
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if name.endswith(".tmp"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, directory).replace(os.sep, "/")
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            try:
                files[rel] = {"encoding": "utf-8", "content": raw.decode("utf-8")}
            except UnicodeDecodeError:
                files[rel] = {
                    "encoding": "base64",
                    "content": base64.b64encode(raw).decode("ascii"),
                }
    payload = {"version": _ARCHIVE_VERSION, "files": files}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _validate_archive_path(directory: str, rel: str) -> str:
    """Resolve *rel* inside *directory*, rejecting traversal and absolute paths."""
    if not rel or "\x00" in rel:
        raise ValueError(f"Invalid path in knowledge archive: {rel!r}")
    if os.path.isabs(rel) or rel.startswith(("/", "\\")) or (len(rel) > 1 and rel[1] == ":"):
        raise ValueError(f"Absolute path in knowledge archive: {rel!r}")
    base = os.path.abspath(directory)
    target = os.path.abspath(os.path.join(base, rel.replace("/", os.sep)))
    if target != base and not target.startswith(base + os.sep):
        raise ValueError(f"Path escapes knowledge directory: {rel!r}")
    return target


def _unpack_directory(directory: str, blob: bytes) -> None:
    """Restore an archive produced by ``_pack_directory`` into *directory*.

    Corrupt (non-JSON / wrong-shape) blobs are logged and ignored so a
    damaged backend row never bricks agent startup.  Unsafe paths and
    unknown encodings raise ``ValueError`` — those indicate tampering, not
    benign corruption.
    """
    try:
        payload = json.loads(blob.decode("utf-8"))
        files = payload["files"]
        if not isinstance(files, dict):
            raise TypeError("'files' must be a dict")
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        _logger.warning("Ignoring corrupt knowledge archive (%s); starting fresh", exc)
        return
    for rel, entry in files.items():
        target = _validate_archive_path(directory, rel)
        encoding = entry.get("encoding")
        content = entry.get("content", "")
        if encoding == "utf-8":
            raw = content.encode("utf-8")
        elif encoding == "base64":
            try:
                raw = base64.b64decode(content, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(
                    f"Invalid base64 content for {rel!r} in knowledge archive"
                ) from exc
        else:
            raise ValueError(f"Unknown encoding {encoding!r} for {rel!r} in knowledge archive")
        os.makedirs(os.path.dirname(target) or directory, exist_ok=True)
        tmp_path = target + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        os.replace(tmp_path, target)


# ======================================================================
# FileKnowledgeStore — wraps existing file-based logic
# ======================================================================


@stable
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
                datetime.now(timezone.utc) - timedelta(days=max_age_days)
                if max_age_days is not None
                else None
            )
            kept = []
            for e in entries:
                if e.persistent:
                    kept.append(e)
                    continue
                if e.is_expired:
                    continue
                if cutoff is not None and e.created_at < cutoff:
                    continue
                if e.importance < min_importance:
                    continue
                kept.append(e)
            self._save_all(kept)
            return before - len(kept)


# ======================================================================
# SQLiteKnowledgeStore
# ======================================================================


@stable
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


@stable
class KnowledgeMemory:
    """Maintains cross-session knowledge with daily logs and persistent facts.

    Supports two modes:

    1. **Legacy file mode** (default): Daily log files and ``MEMORY.md``,
       matching the original API.
    2. **Store mode**: Pass a ``KnowledgeStore`` for structured entries with
       importance scoring, TTL, and pluggable backends.

    Either mode can additionally be paired with a ``KnowledgeBackend`` for
    ephemeral infrastructure: the directory is restored from the backend on
    construction and re-persisted after every mutation, so ``/tmp`` being
    wiped between deploys loses nothing.  ``backend=None`` (the default)
    keeps the original filesystem-only behavior exactly.

    Note: the backend snapshots the *directory*.  A custom ``store`` whose
    data lives outside the directory (e.g. ``SQLiteKnowledgeStore`` with a
    db path elsewhere) is not covered by the snapshot.

    Args:
        directory: Base directory for knowledge files.  Created if absent.
        store: Optional pluggable backend.  If ``None``, uses ``FileKnowledgeStore``.
        recent_days: Number of recent days to include in context.  Default: 2.
        max_context_chars: Maximum characters to include in context output.
        max_entries: Maximum entries before importance-based eviction.  Default: 50.
        backend: Optional ``KnowledgeBackend`` that persists the directory
            as a single blob between processes.  Restored on init, saved
            after ``remember()`` / ``prune_old_logs()`` / ``flush()``.
        pre_save: Optional sanitization hook (or sequence of hooks) applied
            to entry text before persistence (beta).  Each hook receives the
            text and returns the transformed text, or ``None`` to reject the
            entry entirely (silently skipped, logged at debug level).
            Sequences are applied in order; ``None`` short-circuits.
            Built-ins live in ``selectools.knowledge_sanitizers``.
        dedupe: If ``True``, append a near-duplicate rejection hook (beta)
            that runs after all ``pre_save`` hooks, comparing the sanitized
            text against the ``dedupe_window`` most recent store entries
            via ``difflib.SequenceMatcher``.  Costs one store query plus up
            to one similarity computation per windowed entry on every
            ``remember()``.
        dedupe_threshold: Similarity ratio at or above which ``dedupe``
            rejects the new entry.  Default: 0.9.
        dedupe_window: Number of most-recent entries the ``dedupe`` hook
            compares against.  Default: 200.  Bounds the per-save cost
            (similarity is worst-case O(len(a) * len(b)) per comparison);
            the trade-off is that a near-duplicate older than the window
            can re-enter the store.  For large stores or custom windows by
            category, wire ``knowledge_sanitizers.dedupe_against`` yourself
            with a bounded fetcher via ``pre_save``.
    """

    def __init__(
        self,
        directory: str = "./memory",
        store: Optional[KnowledgeStore] = None,
        recent_days: int = 2,
        max_context_chars: int = 5000,
        max_entries: int = 50,
        backend: Optional[KnowledgeBackend] = None,
        pre_save: Optional[Union[PreSaveHook, Sequence[PreSaveHook]]] = None,
        dedupe: bool = False,
        dedupe_threshold: float = 0.9,
        dedupe_window: int = 200,
    ) -> None:
        self._directory = directory
        self._recent_days = recent_days
        self._max_context_chars = max_context_chars
        self._max_entries = max_entries
        self._dedupe_window = dedupe_window
        self._lock = threading.Lock()
        self._backend = backend
        os.makedirs(directory, exist_ok=True)
        if backend is not None:
            blob = backend.load_bytes()
            if blob is not None:
                _unpack_directory(directory, blob)
        self._store = store or FileKnowledgeStore(directory)
        hooks: List[PreSaveHook] = []
        if pre_save is not None:
            if callable(pre_save):
                hooks.append(pre_save)
            else:
                hooks.extend(pre_save)
        if dedupe:
            hooks.append(dedupe_against(self._existing_contents, threshold=dedupe_threshold))
        self._pre_save_hooks = hooks

    def _existing_contents(self) -> List[str]:
        # Bound the dedupe comparison set to the most recent entries
        # (review finding PR #84): similarity is worst-case
        # O(len(a) * len(b)) per comparison, so an unbounded fetcher
        # degrades remember() latency as the store grows.  The store query
        # orders by importance, so re-sort by recency before windowing.
        entries = self._store.query(limit=self._max_entries + 1000)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return [e.content for e in entries[: self._dedupe_window]]

    def _apply_pre_save(self, content: str) -> Optional[str]:
        """Run *content* through the pre_save hook chain.

        Returns the transformed content, or ``None`` if any hook rejected
        the entry (short-circuits the remaining hooks).
        """
        for hook in self._pre_save_hooks:
            result = hook(content)
            if result is None:
                _logger.debug(
                    "pre_save hook %s rejected knowledge entry; skipping save",
                    getattr(hook, "__name__", repr(hook)),
                )
                return None
            content = result
        return content

    @property
    def directory(self) -> str:
        """Base directory for knowledge files."""
        return self._directory

    @property
    def store(self) -> KnowledgeStore:
        """The underlying knowledge store."""
        return self._store

    @property
    def backend(self) -> Optional[KnowledgeBackend]:
        """The blob-persistence backend, or ``None`` for filesystem-only mode."""
        return self._backend

    def flush(self) -> None:
        """Persist the current directory state to the backend.

        No-op when no backend is configured.  ``remember()`` and
        ``prune_old_logs()`` call this automatically; call it manually after
        mutating ``store`` directly or writing files into the directory.
        """
        if self._backend is None:
            return
        with self._lock:
            blob = _pack_directory(self._directory)
        self._backend.save_bytes(blob)

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
            The entry ID, or an empty string if a ``pre_save`` hook rejected
            the entry (nothing is persisted in that case).
        """
        processed = self._apply_pre_save(content)
        if processed is None:
            return ""
        content = processed
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

        self.flush()

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
        days = self._recent_days if days is None else days
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
        keep_days = self._recent_days if keep_days is None else keep_days
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

        if removed:
            self.flush()

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


# KnowledgeStore is @runtime_checkable: a ``__stability__`` class attribute would become
# a structural protocol member on Python 3.9-3.11 and break ``isinstance()``
# for implementations that do not define it, so it is registered instead.
register_stability("KnowledgeStore", "stable")

__stability__ = "beta"

__all__ = [
    "KnowledgeEntry",
    "KnowledgeStore",
    "KnowledgeBackend",
    "FileKnowledgeStore",
    "SQLiteKnowledgeStore",
    "KnowledgeMemory",
    "PreSaveHook",
]
