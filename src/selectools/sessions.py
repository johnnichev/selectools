"""
Persistent session storage for conversation memory.

Provides a protocol and three backends for saving/loading
``ConversationMemory`` across agent restarts.
"""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .memory import ConversationMemory
from .stability import beta, stable


def _make_key(session_id: str, namespace: Optional[str]) -> str:
    """Derive storage key from session_id and optional namespace.

    When namespace is None/empty, returns the bare session_id for
    backward compatibility. When namespace is set, returns
    ``"{namespace}:{session_id}"`` so distinct agents (or agent + team)
    sharing the same session_id do not overwrite each other.
    """
    if namespace:
        return f"{namespace}:{session_id}"
    return session_id


@stable
@dataclass
class SessionMetadata:
    """Lightweight summary of a stored session.

    Attributes:
        session_id: Unique identifier for the session.
        message_count: Number of messages in the session.
        created_at: Unix timestamp when the session was first saved.
        updated_at: Unix timestamp of the most recent save.
    """

    session_id: str
    message_count: int
    created_at: float
    updated_at: float


@beta
@dataclass(frozen=True)
class SessionSearchResult:
    """One session matched by :meth:`SessionStore.search`.

    Attributes:
        session_id: Identifier of the matched session.  When ``search`` was
            called with a ``namespace``, this is the bare session_id (so it
            can be passed straight back to ``load(session_id, namespace=...)``).
            When searching across all namespaces, it follows the same
            convention as ``list()`` for the backend in use.
        score: Relevance score.  Higher is more relevant.  Scores are only
            comparable within a single result set — backends use different
            scoring functions (FTS5 bm25 for SQLite, term frequency elsewhere).
        matched_messages: Snippets (length-capped) of the user/assistant
            messages that matched the query, most relevant first.
    """

    session_id: str
    score: float
    matched_messages: List[str] = field(default_factory=list)


# -- shared search helpers -------------------------------------------------

_SEARCHED_ROLES = ("user", "assistant")
_SNIPPET_MAX_LEN = 160
_MAX_SNIPPETS_PER_SESSION = 5
# Explicit cap on each per-term candidate select in SupabaseSessionStore
# .search().  PostgREST deployments may enforce a server-side ``max-rows``
# that silently truncates unbounded selects; an explicit limit keeps the
# behavior deterministic rather than server-config-dependent.
_SUPABASE_SEARCH_CANDIDATE_LIMIT = 1000


def _search_terms(query: str) -> List[str]:
    """Split *query* into lowercase whitespace-delimited terms."""
    return [t for t in query.lower().split() if t]


def _make_snippet(content: str, terms: List[str], max_len: int = _SNIPPET_MAX_LEN) -> str:
    """Return *content* trimmed to ``max_len`` chars, centered on the first match."""
    if len(content) <= max_len:
        return content
    low = content.lower()
    positions = [p for p in (low.find(t) for t in terms) if p != -1]
    first = min(positions) if positions else 0
    start = max(0, first - max_len // 4)
    end = start + max_len - 6  # leave room for ellipses
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"


def _score_memory_dict(memory_data: Dict[str, Any], terms: List[str]) -> Tuple[float, List[str]]:
    """Term-frequency score a serialized ``ConversationMemory`` dict.

    Counts case-insensitive occurrences of each term in the ``content`` of
    user and assistant messages.  Returns ``(score, snippets)`` where
    *snippets* are the matched message contents, length-capped.
    """
    score = 0
    snippets: List[str] = []
    for msg in memory_data.get("messages", []):
        if msg.get("role") not in _SEARCHED_ROLES:
            continue
        content = msg.get("content") or ""
        low = content.lower()
        hits = sum(low.count(t) for t in terms)
        if hits:
            score += hits
            if len(snippets) < _MAX_SNIPPETS_PER_SESSION:
                snippets.append(_make_snippet(content, terms))
    return float(score), snippets


@stable
class SessionStore(Protocol):
    """Protocol for persistent session backends."""

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        """Persist a conversation memory snapshot.

        Args:
            session_id: Unique identifier for the session.
            memory: Conversation memory snapshot to persist.
            namespace: Optional qualifier (e.g. an agent or team name) that
                isolates sessions that would otherwise collide under the
                same ``session_id``. When ``None`` the bare session_id is
                used (backward-compatible default).
        """
        ...

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        """Load a session, or return ``None`` if it does not exist."""
        ...

    def list(self) -> List[SessionMetadata]:
        """Return metadata for every stored session."""
        ...

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        """Delete a session.  Returns ``True`` if it existed."""
        ...

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        """Check whether a session exists."""
        ...

    def branch(self, source_id: str, new_id: str) -> None:
        """Copy session *source_id* to *new_id*.

        The two sessions are completely independent after branching — modifying
        one does not affect the other.

        Raises:
            ValueError: If *source_id* does not exist.
        """
        ...

    @beta
    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Search message content across stored sessions.

        Matches *query* terms (case-insensitive) against the ``content`` of
        user and assistant messages in every stored session.

        Args:
            query: Free-text query.  Split on whitespace; a session matches
                if any term occurs in any user/assistant message.
            namespace: Restrict the search to sessions saved under this
                namespace.  ``None`` searches all sessions.
            limit: Maximum number of sessions to return.

        Returns:
            Up to *limit* :class:`SessionSearchResult` ordered by descending
            relevance score.

        Note:
            ``search`` is a ``@beta`` addition to the protocol.  Third-party
            stores written before it existed do not implement it: explicit
            subclasses of ``SessionStore`` inherit this default, which raises
            ``NotImplementedError``; purely structural implementations raise
            ``AttributeError``.  Feature-detect with
            ``callable(getattr(store, "search", None))``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement search(). "
            "Cross-session search is a @beta protocol addition; "
            "implement search() or upgrade the store."
        )


# ======================================================================
# JSON file backend
# ======================================================================


@stable
class JsonFileSessionStore:
    """File-based session store using one JSON file per session.

    Follows the ``AuditLogger`` file-based pattern: thread-safe writes,
    ``os.makedirs`` on init, and a simple directory layout.

    Args:
        directory: Directory to store session files.  Created if missing.
        default_ttl: Optional TTL in seconds.  Sessions older than this
            are automatically purged on ``load``/``list``/``exists``.
            ``None`` means sessions never expire.
    """

    def __init__(
        self,
        directory: str = "./sessions",
        default_ttl: Optional[int] = None,
    ) -> None:
        self._directory = directory
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_id: str, namespace: Optional[str] = None) -> str:
        if not session_id:
            raise ValueError("session_id must not be empty")
        key = _make_key(session_id, namespace)
        safe_id = os.path.basename(key)
        if safe_id != key or ".." in key or "\x00" in key or "/" in key:
            raise ValueError(
                f"Invalid session_id/namespace: session_id={session_id!r}, namespace={namespace!r}"
            )
        return os.path.join(self._directory, f"{safe_id}.json")

    def _is_expired(self, data: Dict[str, Any]) -> bool:
        if self._default_ttl is None:
            return False
        updated_at: float = data.get("updated_at", data.get("created_at", 0))
        return (time.time() - updated_at) > self._default_ttl

    # -- public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        path = self._path(session_id, namespace)
        now = time.time()
        existing_created: Optional[float] = None
        with self._lock:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing_created = existing.get("created_at")
                except (json.JSONDecodeError, OSError):
                    pass
            payload = {
                "session_id": session_id,
                "namespace": namespace,
                "created_at": existing_created if existing_created is not None else now,
                "updated_at": now,
                "memory": memory.to_dict(),
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        path = self._path(session_id, namespace)
        with self._lock:
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
            if self._is_expired(data):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None
        return ConversationMemory.from_dict(data["memory"])

    def list(self) -> List[SessionMetadata]:
        results: List[SessionMetadata] = []
        with self._lock:
            for fname in os.listdir(self._directory):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self._directory, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                if self._is_expired(data):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    continue
                results.append(
                    SessionMetadata(
                        session_id=data["session_id"],
                        message_count=data["memory"].get("message_count", 0),
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                    )
                )
        return results

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        path = self._path(session_id, namespace)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
        return False

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        path = self._path(session_id, namespace)
        with self._lock:
            if not os.path.exists(path):
                return False
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return False
        return not self._is_expired(data)

    def branch(self, source_id: str, new_id: str) -> None:
        """Copy session *source_id* to a new session *new_id*."""
        memory = self.load(source_id)
        if memory is None:
            raise ValueError(f"Session {source_id!r} not found")
        self.save(new_id, memory)

    @beta
    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Linear scan over session files with term-frequency scoring.

        Cost is O(total sessions): every ``.json`` file in the directory is
        read and scored in-process.  Fine at this backend's intended scale.
        Expired sessions are skipped (but not purged — search is read-only).
        """
        terms = _search_terms(query)
        if not terms or limit <= 0:
            return []
        results: List[SessionSearchResult] = []
        with self._lock:
            for fname in os.listdir(self._directory):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self._directory, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                if self._is_expired(data):
                    continue
                if namespace is not None and data.get("namespace") != namespace:
                    continue
                score, snippets = _score_memory_dict(data.get("memory", {}), terms)
                if score > 0:
                    results.append(
                        SessionSearchResult(
                            session_id=data["session_id"],
                            score=score,
                            matched_messages=snippets,
                        )
                    )
        results.sort(key=lambda r: (-r.score, r.session_id))
        return results[:limit]


# ======================================================================
# SQLite backend
# ======================================================================


@stable
class SQLiteSessionStore:
    """SQLite-based session store.

    Follows the ``SQLiteVectorStore`` pattern: one ``sqlite3.connect()``
    per operation, ``CREATE TABLE IF NOT EXISTS`` on init.

    Args:
        db_path: Path to the SQLite database file.
        default_ttl: Optional TTL in seconds.  ``None`` means no expiry.
    """

    def __init__(
        self,
        db_path: str = "sessions.db",
        default_ttl: Optional[int] = None,
    ) -> None:
        self._db_path = db_path
        self._default_ttl = default_ttl
        self._fts_enabled = True
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            # Search index tables are purely additive: databases created
            # before this feature gain them here without touching existing
            # rows.  Pre-existing sessions are indexed lazily on first
            # search (see _backfill_search_index).
            try:
                # Probe FTS5 availability directly before touching the real
                # table: ``CREATE VIRTUAL TABLE IF NOT EXISTS`` short-circuits
                # on table-name existence BEFORE resolving the module, so a
                # database created on an FTS5 build and reopened on a build
                # without FTS5 would otherwise "succeed" here and only fail
                # later, inside save()/delete()/search().  The temp-schema
                # probe has no persistent side effects.
                conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe__ USING fts5(x)")
                conn.execute("DROP TABLE temp.__fts5_probe__")
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts
                    USING fts5(content, session_key UNINDEXED)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_search_index (
                        session_key TEXT PRIMARY KEY,
                        indexed_at REAL NOT NULL
                    )
                    """
                )
            except sqlite3.OperationalError:
                # SQLite build without FTS5 (rare): search degrades to a
                # LIKE scan over the sessions table.
                self._fts_enabled = False
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> Any:
        import sqlite3

        return sqlite3.connect(self._db_path)

    def _is_expired_ts(self, updated_at: float) -> bool:
        if self._default_ttl is None:
            return False
        return (time.time() - updated_at) > self._default_ttl

    def _index_session(self, conn: Any, key: str, memory_data: Dict[str, Any]) -> None:
        """(Re)build FTS rows for one session inside an open transaction."""
        conn.execute("DELETE FROM session_messages_fts WHERE session_key = ?", (key,))
        rows = [
            (msg.get("content"), key)
            for msg in memory_data.get("messages", [])
            if msg.get("role") in _SEARCHED_ROLES and msg.get("content")
        ]
        if rows:
            conn.executemany(
                "INSERT INTO session_messages_fts (content, session_key) VALUES (?, ?)",
                rows,
            )
        conn.execute(
            """
            INSERT INTO session_search_index (session_key, indexed_at)
            VALUES (?, ?)
            ON CONFLICT(session_key) DO UPDATE SET indexed_at = excluded.indexed_at
            """,
            (key, time.time()),
        )

    # -- public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        now = time.time()
        memory_dict = memory.to_dict()
        memory_json = json.dumps(memory_dict, ensure_ascii=False)
        msg_count = len(memory)
        key = _make_key(session_id, namespace)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT created_at FROM sessions WHERE session_id = ?",
                (key,),
            ).fetchone()
            created_at = row[0] if row else now
            conn.execute(
                """
                INSERT INTO sessions (session_id, memory_json, message_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    memory_json = excluded.memory_json,
                    message_count = excluded.message_count,
                    updated_at = excluded.updated_at
                """,
                (key, memory_json, msg_count, created_at, now),
            )
            if self._fts_enabled:
                self._index_session(conn, key, memory_dict)
            conn.commit()
        finally:
            conn.close()

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        key = _make_key(session_id, namespace)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT memory_json, updated_at FROM sessions WHERE session_id = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        if self._is_expired_ts(row[1]):
            self.delete(session_id, namespace=namespace)
            return None
        return ConversationMemory.from_dict(json.loads(row[0]))

    def list(self) -> List[SessionMetadata]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT session_id, message_count, created_at, updated_at FROM sessions"
            ).fetchall()
        finally:
            conn.close()
        expired_ids: List[str] = []
        results: List[SessionMetadata] = []
        for sid, mc, ca, ua in rows:
            if self._is_expired_ts(ua):
                expired_ids.append(sid)
                continue
            results.append(
                SessionMetadata(session_id=sid, message_count=mc, created_at=ca, updated_at=ua)
            )
        for sid in expired_ids:
            self.delete(sid)
        return results

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        key = _make_key(session_id, namespace)
        conn = self._conn()
        try:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (key,))
            if self._fts_enabled:
                conn.execute("DELETE FROM session_messages_fts WHERE session_key = ?", (key,))
                conn.execute("DELETE FROM session_search_index WHERE session_key = ?", (key,))
            conn.commit()
            return int(cursor.rowcount) > 0
        finally:
            conn.close()

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        key = _make_key(session_id, namespace)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT updated_at FROM sessions WHERE session_id = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        return not self._is_expired_ts(row[0])

    def branch(self, source_id: str, new_id: str) -> None:
        """Copy session *source_id* to a new session *new_id*."""
        memory = self.load(source_id)
        if memory is None:
            raise ValueError(f"Session {source_id!r} not found")
        self.save(new_id, memory)

    @beta
    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Full-text search over message content via an FTS5 index.

        Each matching message contributes a bounded score: 1.0 for the hit
        plus a bm25-derived tiebreak in [0, 1), so sessions with more
        matching messages always rank higher and bm25 relevance only breaks
        ties.  Sessions saved before this feature existed (or updated by
        older library versions that do not maintain the index) are indexed
        lazily on search — existing rows are never modified.

        If the SQLite build lacks FTS5, degrades to an in-process scan with
        term-frequency scoring (a ``RuntimeWarning`` is emitted).
        """
        terms = _search_terms(query)
        if not terms or limit <= 0:
            return []
        if not self._fts_enabled:
            warnings.warn(
                "This SQLite build lacks FTS5; session search is degrading to a "
                "LIKE scan over the sessions table.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._search_scan(terms, namespace, limit)
        conn = self._conn()
        try:
            self._backfill_search_index(conn)
            return self._search_fts(conn, terms, namespace, limit)
        finally:
            conn.close()

    def _backfill_search_index(self, conn: Any) -> None:
        """Lazily (re)index sessions the FTS index does not cover.

        Catches both rows that predate the index (no ``session_search_index``
        entry) and rows rewritten by a search-unaware writer — e.g. an older
        library version sharing the database — whose ``indexed_at`` lags the
        session's ``updated_at``.
        """
        rows = conn.execute(
            """
            SELECT s.session_id, s.memory_json
            FROM sessions s
            LEFT JOIN session_search_index i ON i.session_key = s.session_id
            WHERE i.session_key IS NULL OR i.indexed_at < s.updated_at
            """
        ).fetchall()
        for key, memory_json in rows:
            try:
                memory_data = json.loads(memory_json)
            except json.JSONDecodeError:
                memory_data = {}
            self._index_session(conn, key, memory_data)
        if rows:
            conn.commit()

    def _valid_keys(self, conn: Any) -> Optional[set]:
        """Return non-expired session keys, or ``None`` when TTL is off."""
        if self._default_ttl is None:
            return None
        rows = conn.execute("SELECT session_id, updated_at FROM sessions").fetchall()
        return {key for key, updated_at in rows if not self._is_expired_ts(updated_at)}

    @staticmethod
    def _strip_namespace(key: str, namespace: Optional[str]) -> Optional[str]:
        """Return the result session_id for *key*, or ``None`` if filtered out.

        With a namespace, only ``"{namespace}:..."`` keys match and the
        prefix is stripped so the id can be passed back to ``load``.
        """
        if namespace is None:
            return key
        prefix = f"{namespace}:"
        if not key.startswith(prefix):
            return None
        return key[len(prefix) :]

    def _search_fts(
        self, conn: Any, terms: List[str], namespace: Optional[str], limit: int
    ) -> List[SessionSearchResult]:
        match_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = conn.execute(
            """
            SELECT session_key, content, bm25(session_messages_fts) AS rank
            FROM session_messages_fts
            WHERE session_messages_fts MATCH ?
            ORDER BY rank
            """,
            (match_expr,),
        ).fetchall()
        valid_keys = self._valid_keys(conn)
        scores: Dict[str, float] = {}
        snippets: Dict[str, List[str]] = {}
        sids: Dict[str, str] = {}
        for key, content, rank in rows:
            if valid_keys is not None and key not in valid_keys:
                continue
            sid = self._strip_namespace(key, namespace)
            if sid is None:
                continue
            sids[key] = sid
            # bm25() is lower-is-better (negative for matches).  Each
            # matching message contributes 1.0 plus a bounded relevance
            # tiebreak r/(1+r) where r = max(0, -rank), keeping the
            # per-message contribution in [1, 2).  Hit count therefore
            # strictly dominates and bm25 only breaks ties: an unbounded
            # rank let a single rare-term hit (IDF-driven bm25 of 9+)
            # outscore several common-term hits, inverting the documented
            # "more matches ranks higher" invariant.
            r = max(0.0, -rank)
            scores[key] = scores.get(key, 0.0) + 1.0 + r / (1.0 + r)
            bucket = snippets.setdefault(key, [])
            if len(bucket) < _MAX_SNIPPETS_PER_SESSION:
                bucket.append(_make_snippet(content, terms))
        results = [
            SessionSearchResult(session_id=sids[key], score=score, matched_messages=snippets[key])
            for key, score in scores.items()
        ]
        results.sort(key=lambda r: (-r.score, r.session_id))
        return results[:limit]

    def _search_scan(
        self, terms: List[str], namespace: Optional[str], limit: int
    ) -> List[SessionSearchResult]:
        """LIKE-prefiltered scan fallback for SQLite builds without FTS5."""
        clauses = " OR ".join(["lower(memory_json) LIKE ? ESCAPE '\\'"] * len(terms))
        params = [
            "%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            for t in terms
        ]
        conn = self._conn()
        try:
            rows = conn.execute(
                # clauses is built from a repeated constant; terms are bound params
                f"SELECT session_id, memory_json, updated_at FROM sessions WHERE {clauses}",  # nosec B608
                params,
            ).fetchall()
        finally:
            conn.close()
        results: List[SessionSearchResult] = []
        for key, memory_json, updated_at in rows:
            if self._is_expired_ts(updated_at):
                continue
            sid = self._strip_namespace(key, namespace)
            if sid is None:
                continue
            try:
                memory_data = json.loads(memory_json)
            except json.JSONDecodeError:
                continue
            score, matched = _score_memory_dict(memory_data, terms)
            if score > 0:
                results.append(
                    SessionSearchResult(session_id=sid, score=score, matched_messages=matched)
                )
        results.sort(key=lambda r: (-r.score, r.session_id))
        return results[:limit]


# ======================================================================
# Redis backend
# ======================================================================


@stable
class RedisSessionStore:
    """Redis-backed session store.

    Follows the ``RedisCache`` pattern: lazy ``import redis``, prefix
    namespace, server-side TTL.

    Args:
        url: Redis connection URL.
        prefix: Key prefix for namespacing.
        default_ttl: Optional TTL in seconds.  ``None`` means no expiry.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "selectools:session:",
        default_ttl: Optional[int] = None,
    ) -> None:
        try:
            import redis as redis_lib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RedisSessionStore requires the 'redis' package. "
                "Install it with: pip install selectools[cache]"
            ) from exc

        self._client: Any = redis_lib.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._default_ttl = default_ttl

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject session IDs that could cause key collisions or other problems."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        if "\x00" in session_id:
            raise ValueError(f"session_id must not contain null bytes: {session_id!r}")
        if len(session_id) > 512:
            raise ValueError(
                f"session_id too long ({len(session_id)} chars, max 512): {session_id!r}"
            )

    @staticmethod
    def _validate_namespace(namespace: Optional[str]) -> None:
        if namespace is None:
            return
        if not namespace:
            raise ValueError("namespace must not be empty when provided")
        if "\x00" in namespace:
            raise ValueError(f"namespace must not contain null bytes: {namespace!r}")
        if len(namespace) > 512:
            raise ValueError(f"namespace too long ({len(namespace)} chars, max 512): {namespace!r}")

    def _key(self, session_id: str, namespace: Optional[str] = None) -> str:
        self._validate_session_id(session_id)
        self._validate_namespace(namespace)
        return f"{self._prefix}{_make_key(session_id, namespace)}"

    def _meta_key(self, session_id: str, namespace: Optional[str] = None) -> str:
        self._validate_session_id(session_id)
        self._validate_namespace(namespace)
        return f"{self._prefix}__meta__{_make_key(session_id, namespace)}"

    # -- public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        now = time.time()
        key = self._key(session_id, namespace)
        meta_key = self._meta_key(session_id, namespace)

        existing_meta = self._client.get(meta_key)
        created_at = now
        if existing_meta:
            try:
                created_at = json.loads(existing_meta).get("created_at", now)
            except (json.JSONDecodeError, TypeError):
                pass

        memory_json = json.dumps(memory.to_dict(), ensure_ascii=False)
        meta_json = json.dumps(
            {
                "session_id": session_id,
                "namespace": namespace,
                "message_count": len(memory),
                "created_at": created_at,
                "updated_at": now,
            }
        )

        pipe = self._client.pipeline()
        if self._default_ttl is not None:
            pipe.setex(key, self._default_ttl, memory_json)
            pipe.setex(meta_key, self._default_ttl, meta_json)
        else:
            pipe.set(key, memory_json)
            pipe.set(meta_key, meta_json)
        pipe.execute()

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        raw = self._client.get(self._key(session_id, namespace))
        if raw is None:
            return None
        return ConversationMemory.from_dict(json.loads(raw))

    def list(self) -> List[SessionMetadata]:
        results: List[SessionMetadata] = []
        seen_ids: set = set()
        cursor: int = 0
        pattern = f"{self._prefix}__meta__*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
            for meta_key in keys:
                raw = self._client.get(meta_key)
                if raw is None:
                    continue
                try:
                    meta = json.loads(raw)
                    # Guard against non-metadata JSON (e.g. data key accidentally
                    # matched when session_id ends with ":meta") and SCAN duplicates.
                    session_id = meta["session_id"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
                if session_id in seen_ids:
                    continue
                seen_ids.add(session_id)
                results.append(
                    SessionMetadata(
                        session_id=session_id,
                        message_count=meta.get("message_count", 0),
                        created_at=meta.get("created_at", 0),
                        updated_at=meta.get("updated_at", 0),
                    )
                )
            if cursor == 0:
                break
        return results

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        key = self._key(session_id, namespace)
        meta_key = self._meta_key(session_id, namespace)
        removed = self._client.delete(key, meta_key)
        return int(removed) > 0

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        return bool(self._client.exists(self._key(session_id, namespace)))

    def branch(self, source_id: str, new_id: str) -> None:
        """Copy session *source_id* to a new session *new_id*."""
        memory = self.load(source_id)
        if memory is None:
            raise ValueError(f"Session {source_id!r} not found")
        self.save(new_id, memory)

    @beta
    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Linear SCAN over the key prefix with in-process matching.

        RediSearch is NOT assumed: every session under the prefix is
        fetched (one ``GET`` per data key plus one for its metadata) and
        scored in-process with term frequency.  Cost is O(total sessions)
        in round-trips and transfers every payload — fine for hundreds of
        sessions, not for very large fleets.
        """
        terms = _search_terms(query)
        if not terms or limit <= 0:
            return []
        meta_prefix = f"{self._prefix}__meta__"
        results: List[SessionSearchResult] = []
        seen: set = set()
        cursor: int = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            for key in keys:
                if key.startswith(meta_prefix) or key in seen:
                    continue
                seen.add(key)
                composite = key[len(self._prefix) :]
                sid, ns = self._resolve_identity(composite)
                if namespace is not None:
                    if ns != namespace:
                        continue
                raw = self._client.get(key)
                if raw is None:
                    continue
                try:
                    memory_data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                score, snippets = _score_memory_dict(memory_data, terms)
                if score > 0:
                    results.append(
                        SessionSearchResult(session_id=sid, score=score, matched_messages=snippets)
                    )
            if cursor == 0:
                break
        results.sort(key=lambda r: (-r.score, r.session_id))
        return results[:limit]

    def _resolve_identity(self, composite: str) -> "Tuple[str, Optional[str]]":
        """Map a composite storage key to ``(session_id, namespace)`` via metadata."""
        meta_raw = self._client.get(f"{self._prefix}__meta__{composite}")
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                return meta.get("session_id", composite), meta.get("namespace")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        return composite, None


# ======================================================================
# Supabase backend
# ======================================================================


def _iso_to_ts(value: Any) -> float:
    """Convert an ISO-8601 string (or numeric timestamp) to a Unix float.

    Handles the ``Z`` suffix and naive datetimes (treated as UTC).
    Returns ``0.0`` on any parse failure rather than raising.
    """
    from datetime import datetime, timezone

    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


@stable
class SupabaseSessionStore:
    """Postgres-backed session store via Supabase PostgREST.

    Each session is stored as a single row keyed by ``session_id`` (or
    ``namespace:session_id`` when a namespace is supplied).  The
    ``memory_json`` column holds the full ``ConversationMemory`` dict.
    Saves use ``upsert(on_conflict='session_id')`` so repeated saves are
    idempotent.

    The ``supabase`` package is an optional dependency.  Install it with::

        pip install selectools[supabase]

    You must create and manage the Supabase client yourself and pass it
    in.  This class never constructs a client internally.

    Args:
        client: An initialised ``supabase.Client`` instance.  Typed as
            ``Any`` to avoid a hard import at module load time.
        table_name: Name of the Postgres table to use.  Defaults to
            ``"selectools_sessions"``.

    Required table DDL (run once in your Supabase project)::

        create table selectools_sessions (
            session_id   text        primary key,
            memory_json  jsonb       not null,
            message_count integer    not null default 0,
            created_at   timestamptz not null default now(),
            updated_at   timestamptz not null default now()
        );

    Lifted from Sheriff (johnnichev/sheriff,
    ``api/src/cashcop/core/session_store.py``) and aligned with the
    upstream backends: validation guards (null bytes, length cap),
    namespace prefix support, ``@stable`` decorator.
    """

    def __init__(
        self,
        client: Any,
        table_name: str = "selectools_sessions",
    ) -> None:
        try:
            import supabase as supabase_lib  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SupabaseSessionStore requires the 'supabase' package. "
                "Install it with: pip install selectools[supabase]"
            ) from exc

        self._client = client
        self._table = table_name

    # -- validation helpers ------------------------------------------------

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject session IDs that could cause key collisions or other problems."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        if "\x00" in session_id:
            raise ValueError(f"session_id must not contain null bytes: {session_id!r}")
        if len(session_id) > 512:
            raise ValueError(
                f"session_id too long ({len(session_id)} chars, max 512): {session_id!r}"
            )

    @staticmethod
    def _validate_namespace(namespace: Optional[str]) -> None:
        if namespace is None:
            return
        if not namespace:
            raise ValueError("namespace must not be empty when provided")
        if "\x00" in namespace:
            raise ValueError(f"namespace must not contain null bytes: {namespace!r}")
        if len(namespace) > 512:
            raise ValueError(f"namespace too long ({len(namespace)} chars, max 512): {namespace!r}")

    def _key(self, session_id: str, namespace: Optional[str] = None) -> str:
        self._validate_session_id(session_id)
        self._validate_namespace(namespace)
        return _make_key(session_id, namespace)

    # -- public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        """Persist *memory* under *session_id* (upsert on conflict)."""
        from datetime import datetime, timezone

        key = self._key(session_id, namespace)
        payload = {
            "session_id": key,
            "memory_json": memory.to_dict(),
            "message_count": len(memory),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._client.table(self._table).upsert(payload, on_conflict="session_id").execute()

    def load(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Optional[ConversationMemory]:
        """Return the stored ``ConversationMemory``, or ``None`` if absent."""
        key = self._key(session_id, namespace)
        response = (
            self._client.table(self._table)
            .select("memory_json")
            .eq("session_id", key)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return ConversationMemory.from_dict(response.data[0]["memory_json"])

    def list(self) -> List[SessionMetadata]:
        """Return metadata for every row in the table."""
        response = (
            self._client.table(self._table)
            .select("session_id,message_count,created_at,updated_at")
            .execute()
        )
        rows = response.data or []
        return [
            SessionMetadata(
                session_id=row["session_id"],
                message_count=int(row.get("message_count") or 0),
                created_at=_iso_to_ts(row.get("created_at")),
                updated_at=_iso_to_ts(row.get("updated_at")),
            )
            for row in rows
        ]

    def delete(self, session_id: str, namespace: Optional[str] = None) -> bool:
        """Delete *session_id*.  Returns ``True`` if the row existed."""
        key = self._key(session_id, namespace)
        response = self._client.table(self._table).delete().eq("session_id", key).execute()
        return bool(response.data)

    def exists(self, session_id: str, namespace: Optional[str] = None) -> bool:
        """Return ``True`` if *session_id* has a stored row."""
        key = self._key(session_id, namespace)
        response = (
            self._client.table(self._table)
            .select("session_id")
            .eq("session_id", key)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def branch(self, source_id: str, new_id: str) -> None:
        """Copy session *source_id* to a new session *new_id*."""
        memory = self.load(source_id)
        if memory is None:
            raise ValueError(f"Session {source_id!r} not found")
        self.save(new_id, memory)

    @beta
    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[SessionSearchResult]:
        """Server-prefiltered search via PostgREST ``ilike``.

        The ``memory_json`` column is ``jsonb``, so the filter targets the
        text projection of its ``messages`` array
        (``memory_json->>messages``) with one ``ilike '%term%'`` request
        per query term.  Because string contents appear JSON-escaped in
        that projection (quotes as ``\\"``, backslashes as ``\\\\``), each
        term is JSON-escaped the same way before building the pattern, so
        terms like ``C:\\Users`` still match.  Matched rows are then scored
        in-process with term frequency, which also discards false positives
        from the coarse server-side filter (e.g. matches inside tool
        messages or metadata).

        Limits: one round-trip per term; full ``memory_json`` payloads of
        every server-side match are transferred; no server-side ranking.
        Each per-term candidate set is explicitly capped at
        ``_SUPABASE_SEARCH_CANDIDATE_LIMIT`` (1000) rows — note that
        PostgREST's server-side ``max-rows`` setting may otherwise truncate
        result sets on large tables at a server-configured (and therefore
        nondeterministic from the client's perspective) point.  For tables
        where matches may exceed the cap, add a proper ``tsvector`` index
        and a dedicated RPC instead.
        """
        terms = _search_terms(query)
        if not terms or limit <= 0:
            return []
        rows_by_key: Dict[str, Dict[str, Any]] = {}
        for term in terms:
            # Match the JSON text projection: escape the term exactly as
            # jsonb renders string contents (json.dumps minus the
            # surrounding quotes).
            escaped_term = json.dumps(term, ensure_ascii=False)[1:-1]
            response = (
                self._client.table(self._table)
                .select("session_id,memory_json")
                .ilike("memory_json->>messages", f"%{escaped_term}%")
                .limit(_SUPABASE_SEARCH_CANDIDATE_LIMIT)
                .execute()
            )
            for row in response.data or []:
                rows_by_key[row["session_id"]] = row
        results: List[SessionSearchResult] = []
        for key, row in rows_by_key.items():
            sid = key
            if namespace is not None:
                prefix = f"{namespace}:"
                if not key.startswith(prefix):
                    continue
                sid = key[len(prefix) :]
            memory_data = row.get("memory_json") or {}
            score, snippets = _score_memory_dict(memory_data, terms)
            if score > 0:
                results.append(
                    SessionSearchResult(session_id=sid, score=score, matched_messages=snippets)
                )
        results.sort(key=lambda r: (-r.score, r.session_id))
        return results[:limit]


__stability__ = "stable"

__all__ = [
    "SessionStore",
    "SessionMetadata",
    "SessionSearchResult",
    "JsonFileSessionStore",
    "SQLiteSessionStore",
    "RedisSessionStore",
    "SupabaseSessionStore",
]
