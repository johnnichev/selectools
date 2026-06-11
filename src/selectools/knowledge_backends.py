"""
Blob-persistence adapters for ``KnowledgeMemory`` on ephemeral infrastructure.

Implements the ``selectools.knowledge.KnowledgeBackend`` protocol: a single
opaque byte payload per key, loaded on ``KnowledgeMemory`` construction and
saved after every mutation.  The pattern is "scratch on disk during the
request, persist to blob/DB between requests" — Railway, Lambda, Cloud Run
and friends wipe ``/tmp`` between deploys, the backend doesn't.

Usage::

    from selectools import KnowledgeMemory
    from selectools.knowledge_backends import SupabaseKnowledgeBackend

    memory = KnowledgeMemory(
        directory="/tmp/agent-memory",
        backend=SupabaseKnowledgeBackend(client, key="user-123"),
    )

Both adapters use lazy imports — ``supabase`` and ``redis`` stay optional
dependencies (``pip install selectools[supabase]`` / ``selectools[cache]``).
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from .stability import beta

_B64_MARKER = "b64:"


def _validate_key(key: str) -> None:
    """Reject keys that could cause collisions or storage problems."""
    if not key:
        raise ValueError("key must not be empty")
    if "\x00" in key:
        raise ValueError(f"key must not contain null bytes: {key!r}")
    if len(key) > 512:
        raise ValueError(f"key too long ({len(key)} chars, max 512): {key!r}")


def _encode_text(data: bytes) -> str:
    """Encode arbitrary bytes into a text-column-safe string.

    UTF-8 payloads (the common case — the archive is UTF-8 JSON) are stored
    verbatim so the column stays human-readable in the dashboard.  Anything
    else, or text that collides with the marker, is base64-encoded behind a
    ``b64:`` prefix.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _B64_MARKER + base64.b64encode(data).decode("ascii")
    if text.startswith(_B64_MARKER):
        return _B64_MARKER + base64.b64encode(data).decode("ascii")
    return text


def _decode_text(value: str) -> bytes:
    """Inverse of ``_encode_text``."""
    if value.startswith(_B64_MARKER):
        return base64.b64decode(value[len(_B64_MARKER) :])
    return value.encode("utf-8")


@beta
class SupabaseKnowledgeBackend:
    """Postgres-backed knowledge blob storage via Supabase PostgREST.

    Stores one row per *key* in a plain table — same client handling as
    ``SupabaseSessionStore``: you create and manage the ``supabase.Client``
    yourself and pass it in.  The blob is stored in a ``text`` column
    (UTF-8 verbatim when possible, ``b64:``-prefixed base64 otherwise) so
    memory contents stay inspectable in the Supabase dashboard.

    The ``supabase`` package is an optional dependency.  Install it with::

        pip install selectools[supabase]

    Args:
        client: An initialised ``supabase.Client`` instance.  Typed as
            ``Any`` to avoid a hard import at module load time.
        key: Row key identifying this memory (e.g. a user id).
        table_name: Postgres table to use.  Default: ``"selectools_knowledge"``.
        key_column: Primary-key column name.  Default: ``"key"``.
        data_column: Text column holding the blob.  Default: ``"data"``.

    Required table DDL (run once in your Supabase project)::

        create table selectools_knowledge (
            key        text        primary key,
            data       text        not null,
            updated_at timestamptz not null default now()
        );

    To reuse an existing table (e.g. Sheriff's ``users.memory_text``)::

        SupabaseKnowledgeBackend(
            client,
            key=user_id,
            table_name="users",
            key_column="user_id",
            data_column="memory_text",
        )
    """

    def __init__(
        self,
        client: Any,
        key: str,
        table_name: str = "selectools_knowledge",
        key_column: str = "key",
        data_column: str = "data",
    ) -> None:
        try:
            import supabase as supabase_lib  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SupabaseKnowledgeBackend requires the 'supabase' package. "
                "Install it with: pip install selectools[supabase]"
            ) from exc

        _validate_key(key)
        self._client = client
        self._key = key
        self._table = table_name
        self._key_column = key_column
        self._data_column = data_column

    def load_bytes(self) -> Optional[bytes]:
        """Return the stored blob for this key, or ``None`` if absent."""
        response = (
            self._client.table(self._table)
            .select(self._data_column)
            .eq(self._key_column, self._key)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        value = response.data[0].get(self._data_column)
        if value is None:
            return None
        return _decode_text(str(value))

    def save_bytes(self, data: bytes) -> None:
        """Upsert *data* under this key (idempotent on repeated saves)."""
        from datetime import datetime, timezone

        payload = {
            self._key_column: self._key,
            self._data_column: _encode_text(data),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._client.table(self._table).upsert(payload, on_conflict=self._key_column).execute()


@beta
class RedisKnowledgeBackend:
    """Redis-backed knowledge blob storage.

    Follows the ``RedisSessionStore`` pattern: lazy ``import redis``, prefix
    namespace, optional server-side TTL.  Values are raw bytes
    (``decode_responses`` is left off), so the archive round-trips exactly.

    The ``redis`` package is an optional dependency.  Install it with::

        pip install selectools[cache]

    Args:
        key: Key identifying this memory (e.g. a user id).
        url: Redis connection URL.
        prefix: Key prefix for namespacing.  Default: ``"selectools:knowledge:"``.
        ttl: Optional TTL in seconds.  ``None`` means the blob never expires.
    """

    def __init__(
        self,
        key: str,
        url: str = "redis://localhost:6379/0",
        prefix: str = "selectools:knowledge:",
        ttl: Optional[int] = None,
    ) -> None:
        try:
            import redis as redis_lib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RedisKnowledgeBackend requires the 'redis' package. "
                "Install it with: pip install selectools[cache]"
            ) from exc

        _validate_key(key)
        self._client: Any = redis_lib.from_url(url)
        self._key = f"{prefix}{key}"
        self._ttl = ttl

    def load_bytes(self) -> Optional[bytes]:
        """Return the stored blob for this key, or ``None`` if absent."""
        raw = self._client.get(self._key)
        if raw is None:
            return None
        if isinstance(raw, str):
            # Tolerate clients constructed with decode_responses=True.
            return raw.encode("utf-8")
        return bytes(raw)

    def save_bytes(self, data: bytes) -> None:
        """Store *data* under this key, applying the TTL when configured."""
        if self._ttl is not None:
            self._client.setex(self._key, self._ttl, data)
        else:
            self._client.set(self._key, data)


__stability__ = "beta"

__all__ = [
    "SupabaseKnowledgeBackend",
    "RedisKnowledgeBackend",
]
