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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

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
                f"Invalid session_id/namespace: session_id={session_id!r}, "
                f"namespace={namespace!r}"
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

    # -- public API --------------------------------------------------------

    def save(
        self,
        session_id: str,
        memory: ConversationMemory,
        namespace: Optional[str] = None,
    ) -> None:
        now = time.time()
        memory_json = json.dumps(memory.to_dict(), ensure_ascii=False)
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


# ======================================================================
# Redis backend
# ======================================================================


@beta
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


__all__ = [
    "SessionStore",
    "SessionMetadata",
    "JsonFileSessionStore",
    "SQLiteSessionStore",
    "RedisSessionStore",
]
