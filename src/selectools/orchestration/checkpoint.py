"""
Checkpoint storage for AgentGraph execution.

Provides durable mid-graph state persistence for:
- Human-in-the-loop (HITL) pause/resume
- Recovery from failures at any node
- Distributed execution across processes

Three backends mirror the SessionStore pattern:
- InMemoryCheckpointStore: dict-backed, thread-safe (dev/test)
- FileCheckpointStore: one JSON file per checkpoint (single-machine)
- SQLiteCheckpointStore: WAL-mode SQLite (concurrent access, production)

Protocol::

    store = SQLiteCheckpointStore("checkpoints.db")
    checkpoint_id = store.save(graph_id, state, step)
    state, step = store.load(checkpoint_id)
    metas = store.list(graph_id)
    store.delete(checkpoint_id)
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .state import GraphState


@dataclass
class CheckpointMetadata:
    """Metadata record for a stored checkpoint.

    Attributes:
        checkpoint_id: Unique identifier returned by save().
        graph_id: The run_id of the graph execution this checkpoint belongs to.
        step: Graph step number at the time of checkpointing.
        node_name: The node that was executing when the checkpoint was created.
        interrupted: True if this checkpoint was created by an InterruptRequest.
        created_at: UTC timestamp of creation.
    """

    checkpoint_id: str
    graph_id: str
    step: int
    node_name: str
    interrupted: bool
    created_at: datetime


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint backends.

    All implementations must be thread-safe.
    """

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        """Persist a checkpoint and return its unique checkpoint_id.

        The checkpoint_id is used by graph.resume() to reload this exact state.
        ``_interrupt_responses`` from state is serialized into a separate
        ``__interrupt__`` key so it survives the checkpoint/resume cycle.
        """
        ...

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        """Load a checkpoint by ID. Returns (state, step).

        Raises:
            ValueError: If checkpoint_id is not found.
        """
        ...

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a graph_id, sorted by created_at ascending."""
        ...

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID. Returns True if deleted, False if not found."""
        ...


def _serialize_checkpoint(state: GraphState, step: int) -> Dict:
    """Serialize state + interrupt responses into a JSON-safe dict."""
    d = state.to_dict()
    # Separately serialize _interrupt_responses (excluded from to_dict)
    d["__interrupt__"] = copy.deepcopy(state._interrupt_responses)
    d["__step__"] = step
    d["__node_name__"] = state.current_node
    d["__interrupted__"] = bool(state.metadata.get("__pending_interrupt_key__"))
    return d


def _deserialize_checkpoint(data: Dict) -> Tuple[GraphState, int]:
    """Deserialize a checkpoint dict back to (GraphState, step)."""
    step = data.pop("__step__", 0)
    interrupt_responses = data.pop("__interrupt__", {})
    data.pop("__node_name__", None)
    data.pop("__interrupted__", None)

    state = GraphState.from_dict(data)
    state._interrupt_responses = interrupt_responses
    return state, step


# ------------------------------------------------------------------
# InMemoryCheckpointStore
# ------------------------------------------------------------------


class InMemoryCheckpointStore:
    """Thread-safe in-memory checkpoint store.

    Suitable for development and testing. All data is lost when the
    process exits.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, Dict] = {}  # checkpoint_id → serialized dict
        self._meta: Dict[str, CheckpointMetadata] = {}

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        checkpoint_id = uuid.uuid4().hex
        data = _serialize_checkpoint(state, step)
        interrupted = bool(state.metadata.get("__pending_interrupt_key__"))
        meta = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            graph_id=graph_id,
            step=step,
            node_name=state.current_node,
            interrupted=interrupted,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._store[checkpoint_id] = data
            self._meta[checkpoint_id] = meta
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        with self._lock:
            data = self._store.get(checkpoint_id)
        if data is None:
            raise ValueError(f"Checkpoint {checkpoint_id!r} not found")
        return _deserialize_checkpoint(dict(data))

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        with self._lock:
            metas = [m for m in self._meta.values() if m.graph_id == graph_id]
        return sorted(metas, key=lambda m: m.created_at)

    def delete(self, checkpoint_id: str) -> bool:
        with self._lock:
            found = checkpoint_id in self._store
            if found:
                del self._store[checkpoint_id]
                self._meta.pop(checkpoint_id, None)
        return found


# ------------------------------------------------------------------
# FileCheckpointStore
# ------------------------------------------------------------------


class FileCheckpointStore:
    """File-based checkpoint store.

    Each checkpoint is stored as ``{directory}/{graph_id}/{checkpoint_id}.json``.
    Suitable for single-machine production use.

    Args:
        directory: Root directory for checkpoint files. Created if not exists.
    """

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        os.makedirs(directory, exist_ok=True)

    def _graph_dir(self, graph_id: str) -> str:
        safe_id = os.path.basename(graph_id)
        if not safe_id or safe_id != graph_id:
            raise ValueError(f"Invalid graph_id: {graph_id!r}")
        d = os.path.join(self._directory, safe_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _checkpoint_path(self, graph_id: str, checkpoint_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), f"{checkpoint_id}.json")

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        checkpoint_id = uuid.uuid4().hex
        data = _serialize_checkpoint(state, step)
        interrupted = bool(state.metadata.get("__pending_interrupt_key__"))
        meta = {
            "checkpoint_id": checkpoint_id,
            "graph_id": graph_id,
            "step": step,
            "node_name": state.current_node,
            "interrupted": interrupted,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"meta": meta, "state": data}

        path = self._checkpoint_path(graph_id, checkpoint_id)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        # Search for the file across all graph subdirectories
        with self._lock:
            for graph_id in os.listdir(self._directory):
                path = os.path.join(self._directory, graph_id, f"{checkpoint_id}.json")
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as f:
                        payload = json.load(f)
                    return _deserialize_checkpoint(payload["state"])
        raise ValueError(f"Checkpoint {checkpoint_id!r} not found in {self._directory!r}")

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        graph_dir = os.path.join(self._directory, graph_id)
        if not os.path.isdir(graph_dir):
            return []

        metas = []
        with self._lock:
            for fname in os.listdir(graph_dir):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(graph_dir, fname)
                try:
                    with open(path, encoding="utf-8") as f:
                        payload = json.load(f)
                    m = payload["meta"]
                    metas.append(
                        CheckpointMetadata(
                            checkpoint_id=m["checkpoint_id"],
                            graph_id=m["graph_id"],
                            step=m["step"],
                            node_name=m.get("node_name", ""),
                            interrupted=m.get("interrupted", False),
                            created_at=datetime.fromisoformat(m["created_at"]),
                        )
                    )
                except Exception:  # nosec B110
                    pass

        return sorted(metas, key=lambda m: m.created_at)

    def delete(self, checkpoint_id: str) -> bool:
        with self._lock:
            for graph_id in os.listdir(self._directory):
                path = os.path.join(self._directory, graph_id, f"{checkpoint_id}.json")
                if os.path.isfile(path):
                    os.remove(path)
                    return True
        return False


# ------------------------------------------------------------------
# SQLiteCheckpointStore
# ------------------------------------------------------------------


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    graph_id      TEXT NOT NULL,
    step          INTEGER NOT NULL,
    node_name     TEXT NOT NULL DEFAULT '',
    interrupted   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    state_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_graph_id ON checkpoints (graph_id);
"""


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint store with WAL mode for concurrent access.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_CREATE_TABLE)
        conn.commit()

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        checkpoint_id = uuid.uuid4().hex
        data = _serialize_checkpoint(state, step)
        interrupted = bool(state.metadata.get("__pending_interrupt_key__"))
        created_at = datetime.now(timezone.utc).isoformat()

        conn = self._conn()
        conn.execute(
            """
            INSERT INTO checkpoints
                (checkpoint_id, graph_id, step, node_name, interrupted, created_at, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                graph_id,
                step,
                state.current_node,
                1 if interrupted else 0,
                created_at,
                json.dumps(data, default=str),
            ),
        )
        conn.commit()
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        conn = self._conn()
        row = conn.execute(
            "SELECT state_json FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Checkpoint {checkpoint_id!r} not found")
        data = json.loads(row["state_json"])
        return _deserialize_checkpoint(data)

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT checkpoint_id, graph_id, step, node_name, interrupted, created_at
            FROM checkpoints WHERE graph_id = ?
            ORDER BY created_at ASC
            """,
            (graph_id,),
        ).fetchall()
        return [
            CheckpointMetadata(
                checkpoint_id=row["checkpoint_id"],
                graph_id=row["graph_id"],
                step=row["step"],
                node_name=row["node_name"],
                interrupted=bool(row["interrupted"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def delete(self, checkpoint_id: str) -> bool:
        conn = self._conn()
        cursor = conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        conn.commit()
        return cursor.rowcount > 0


__all__ = [
    "CheckpointMetadata",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "SQLiteCheckpointStore",
]
