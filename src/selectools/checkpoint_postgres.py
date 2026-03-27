"""
PostgresCheckpointStore — durable checkpoint backend for production.

Requires ``asyncpg`` (optional dependency)::

    pip install selectools[postgres]
    # or: pip install asyncpg

Usage::

    from selectools.checkpoint_postgres import PostgresCheckpointStore

    store = PostgresCheckpointStore(dsn="postgresql://user:pass@host/db")
    result = graph.run("...", checkpoint_store=store)
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .orchestration.checkpoint import (
    CheckpointMetadata,
    _deserialize_checkpoint,
    _serialize_checkpoint,
)
from .orchestration.state import GraphState


class PostgresCheckpointStore:
    """Postgres-backed checkpoint store for production orchestration.

    Uses synchronous ``psycopg2`` for compatibility with the sync
    ``CheckpointStore`` protocol. For async usage, wrap in
    ``asyncio.to_thread``.

    Args:
        dsn: PostgreSQL connection string.
        table: Table name for checkpoints. Default: ``checkpoints``.
    """

    def __init__(self, dsn: str, table: str = "checkpoints") -> None:
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError:
            try:
                import psycopg2cffi as psycopg2  # type: ignore[import-untyped,no-redef]
            except ImportError:
                raise ImportError(
                    "psycopg2 or psycopg2-binary is required for PostgresCheckpointStore. "
                    "Install with: pip install psycopg2-binary"
                ) from None

        self._dsn = dsn
        self._table = table
        self._lock = threading.Lock()
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._init_table()

    def _init_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    checkpoint_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL DEFAULT '',
                    interrupted BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    state_json JSONB NOT NULL
                )
                """  # nosec B608
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_graph_id
                ON {self._table}(graph_id)
                """  # nosec B608
            )

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        """Persist a checkpoint. Returns checkpoint_id."""
        checkpoint_id = uuid.uuid4().hex
        data = _serialize_checkpoint(state, step)
        interrupted = bool(state.metadata.get("__pending_interrupt_key__"))
        state_json = json.dumps(data, default=str)

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._table} "  # nosec B608
                    "(checkpoint_id, graph_id, step, node_name, interrupted, state_json) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (checkpoint_id, graph_id, step, state.current_node, interrupted, state_json),
                )
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        """Load a checkpoint by ID. Raises ValueError if not found."""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT state_json FROM {self._table} WHERE checkpoint_id = %s",  # nosec B608
                    (checkpoint_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"Checkpoint {checkpoint_id!r} not found")
        data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        return _deserialize_checkpoint(dict(data))

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a graph_id, sorted by created_at."""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT checkpoint_id, graph_id, step, node_name, interrupted, created_at "  # nosec B608
                    f"FROM {self._table} WHERE graph_id = %s ORDER BY created_at ASC",
                    (graph_id,),
                )
                rows = cur.fetchall()
        return [
            CheckpointMetadata(
                checkpoint_id=r[0],
                graph_id=r[1],
                step=r[2],
                node_name=r[3],
                interrupted=r[4],
                created_at=(
                    r[5] if isinstance(r[5], datetime) else datetime.fromisoformat(str(r[5]))
                ),
            )
            for r in rows
        ]

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if deleted."""
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE checkpoint_id = %s",  # nosec B608
                    (checkpoint_id,),
                )
                return cur.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


__all__ = ["PostgresCheckpointStore"]
