"""
TraceStore protocol and implementations for persisting agent traces.

Three backends: InMemory (dev), SQLite (production), JSONL (export/archive).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from ..trace import AgentTrace


@dataclass
class TraceSummary:
    """Lightweight summary of a stored trace."""

    run_id: str
    steps: int
    total_ms: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceFilter:
    """Filter criteria for querying traces."""

    metadata_match: Optional[Dict[str, Any]] = None
    min_steps: Optional[int] = None
    max_steps: Optional[int] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for trace persistence backends."""

    def save(self, trace: AgentTrace) -> str:
        """Persist a trace. Returns the run_id."""
        ...

    def load(self, run_id: str) -> AgentTrace:
        """Load a trace by run_id. Raises ValueError if not found."""
        ...

    def list(self, limit: int = 50, offset: int = 0) -> List[TraceSummary]:
        """List trace summaries, newest first."""
        ...

    def query(self, filters: TraceFilter) -> List[TraceSummary]:
        """Query traces matching filter criteria."""
        ...

    def delete(self, run_id: str) -> bool:
        """Delete a trace. Returns True if deleted."""
        ...


# ---------------------------------------------------------------------------
# InMemoryTraceStore
# ---------------------------------------------------------------------------


class InMemoryTraceStore:
    """In-memory trace store for development and testing."""

    def __init__(self, max_size: int = 1000) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def save(self, trace: AgentTrace) -> str:
        run_id = trace.run_id
        data = {
            "trace": trace,
            "created_at": datetime.now(timezone.utc),
        }
        with self._lock:
            self._store[run_id] = data
            self._order.append(run_id)
            while len(self._order) > self._max_size:
                old = self._order.pop(0)
                self._store.pop(old, None)
        return run_id

    def load(self, run_id: str) -> AgentTrace:
        with self._lock:
            entry = self._store.get(run_id)
        if entry is None:
            raise ValueError(f"Trace {run_id!r} not found")
        return entry["trace"]  # type: ignore[no-any-return]

    def list(self, limit: int = 50, offset: int = 0) -> List[TraceSummary]:
        with self._lock:
            ids = list(reversed(self._order))[offset : offset + limit]
            return [self._to_summary(self._store[rid]) for rid in ids if rid in self._store]

    def query(self, filters: TraceFilter) -> List[TraceSummary]:
        results = []
        with self._lock:
            for rid in reversed(self._order):
                entry = self._store.get(rid)
                if entry and self._matches(entry, filters):
                    results.append(self._to_summary(entry))
        return results

    def delete(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._store:
                del self._store[run_id]
                if run_id in self._order:
                    self._order.remove(run_id)
                return True
        return False

    def _to_summary(self, entry: Dict[str, Any]) -> TraceSummary:
        trace: AgentTrace = entry["trace"]
        return TraceSummary(
            run_id=trace.run_id,
            steps=len(trace.steps),
            total_ms=trace.total_duration_ms,
            created_at=entry["created_at"],
            metadata=trace.metadata,
        )

    def _matches(self, entry: Dict[str, Any], f: TraceFilter) -> bool:
        trace: AgentTrace = entry["trace"]
        created = entry["created_at"]
        if f.min_steps is not None and len(trace.steps) < f.min_steps:
            return False
        if f.max_steps is not None and len(trace.steps) > f.max_steps:
            return False
        if f.since is not None and created < f.since:
            return False
        if f.until is not None and created > f.until:
            return False
        if f.metadata_match:
            for k, v in f.metadata_match.items():
                if trace.metadata.get(k) != v:
                    return False
        return True


# ---------------------------------------------------------------------------
# SQLiteTraceStore
# ---------------------------------------------------------------------------

_TRACE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    run_id TEXT PRIMARY KEY,
    steps INTEGER NOT NULL,
    total_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
"""


class SQLiteTraceStore:
    """SQLite-backed trace store for production use."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn  # type: ignore[no-any-return]

    def _init_db(self) -> None:
        self._conn().executescript(_TRACE_TABLE)

    def save(self, trace: AgentTrace) -> str:
        run_id = trace.run_id
        now = datetime.now(timezone.utc).isoformat()
        trace_json = json.dumps(trace.to_dict(), default=str)
        meta_json = json.dumps(trace.metadata, default=str)

        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO traces (run_id, steps, total_ms, created_at, metadata, trace_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, len(trace.steps), trace.total_duration_ms, now, meta_json, trace_json),
        )
        conn.commit()
        return run_id

    def load(self, run_id: str) -> AgentTrace:
        row = (
            self._conn()
            .execute("SELECT trace_json FROM traces WHERE run_id = ?", (run_id,))
            .fetchone()
        )
        if row is None:
            raise ValueError(f"Trace {run_id!r} not found")
        return AgentTrace.from_dict(json.loads(row[0]))

    def list(self, limit: int = 50, offset: int = 0) -> List[TraceSummary]:
        rows = (
            self._conn()
            .execute(
                "SELECT run_id, steps, total_ms, created_at, metadata FROM traces "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            .fetchall()
        )
        return [self._row_to_summary(r) for r in rows]

    def query(self, filters: TraceFilter) -> List[TraceSummary]:
        conditions = []
        params: List[Any] = []

        if filters.min_steps is not None:
            conditions.append("steps >= ?")
            params.append(filters.min_steps)
        if filters.max_steps is not None:
            conditions.append("steps <= ?")
            params.append(filters.max_steps)
        if filters.since is not None:
            conditions.append("created_at >= ?")
            params.append(filters.since.isoformat())
        if filters.until is not None:
            conditions.append("created_at <= ?")
            params.append(filters.until.isoformat())

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = (
            self._conn()
            .execute(
                f"SELECT run_id, steps, total_ms, created_at, metadata FROM traces "  # noqa: S608 # nosec B608
                f"WHERE {where} ORDER BY created_at DESC",
                params,
            )
            .fetchall()
        )

        results = [self._row_to_summary(r) for r in rows]

        # Metadata filtering in Python (not SQL)
        if filters.metadata_match:
            results = [
                s
                for s in results
                if all(s.metadata.get(k) == v for k, v in filters.metadata_match.items())
            ]
        return results

    def delete(self, run_id: str) -> bool:
        conn = self._conn()
        cursor = conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0

    def _row_to_summary(self, row: tuple) -> TraceSummary:
        run_id, steps, total_ms, created_at, meta_json = row
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except json.JSONDecodeError:
            meta = {}
        return TraceSummary(
            run_id=run_id,
            steps=steps,
            total_ms=total_ms,
            created_at=datetime.fromisoformat(created_at),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# JSONLTraceStore
# ---------------------------------------------------------------------------


class JSONLTraceStore:
    """Append-only JSONL trace store for export and archival."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trace: AgentTrace) -> str:
        run_id = trace.run_id
        entry = {
            "run_id": run_id,
            "steps": len(trace.steps),
            "total_ms": trace.total_duration_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": trace.metadata,
            "trace": trace.to_dict(),
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return run_id

    def load(self, run_id: str) -> AgentTrace:
        for entry in self._iter_entries():
            if entry.get("run_id") == run_id:
                return AgentTrace.from_dict(entry["trace"])
        raise ValueError(f"Trace {run_id!r} not found in {self._path}")

    def list(self, limit: int = 50, offset: int = 0) -> List[TraceSummary]:
        entries = list(self._iter_entries())
        entries.reverse()  # newest first
        return [self._to_summary(e) for e in entries[offset : offset + limit]]

    def query(self, filters: TraceFilter) -> List[TraceSummary]:
        results = []
        for entry in self._iter_entries():
            summary = self._to_summary(entry)
            if self._matches(summary, entry, filters):
                results.append(summary)
        results.reverse()
        return results

    def delete(self, run_id: str) -> bool:
        # JSONL is append-only; delete rewrites the file
        with self._lock:
            entries = list(self._iter_entries())
            filtered = [e for e in entries if e.get("run_id") != run_id]
            if len(filtered) == len(entries):
                return False
            with open(self._path, "w", encoding="utf-8") as f:
                for entry in filtered:
                    f.write(json.dumps(entry, default=str) + "\n")
            return True

    def _iter_entries(self) -> Iterator[Dict[str, Any]]:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _to_summary(self, entry: Dict[str, Any]) -> TraceSummary:
        return TraceSummary(
            run_id=entry.get("run_id", ""),
            steps=entry.get("steps", 0),
            total_ms=entry.get("total_ms", 0.0),
            created_at=datetime.fromisoformat(entry.get("created_at", "2000-01-01T00:00:00+00:00")),
            metadata=entry.get("metadata", {}),
        )

    def _matches(self, summary: TraceSummary, entry: Dict, f: TraceFilter) -> bool:
        if f.min_steps is not None and summary.steps < f.min_steps:
            return False
        if f.max_steps is not None and summary.steps > f.max_steps:
            return False
        if f.since is not None and summary.created_at < f.since:
            return False
        if f.until is not None and summary.created_at > f.until:
            return False
        if f.metadata_match:
            for k, v in f.metadata_match.items():
                if summary.metadata.get(k) != v:
                    return False
        return True


__all__ = [
    "TraceStore",
    "TraceSummary",
    "TraceFilter",
    "InMemoryTraceStore",
    "SQLiteTraceStore",
    "JSONLTraceStore",
]
