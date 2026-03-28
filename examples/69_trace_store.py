"""
Example 69: Trace Storage and Querying

Demonstrates saving and querying agent traces (v0.19.0):
- InMemoryTraceStore for development
- SQLiteTraceStore for persistent storage
- Saving traces from agent runs
- Querying by run_id, time range, and metadata filters

Uses LocalProvider so no API keys are needed.

Run:
    python examples/69_trace_store.py
"""

import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider
from selectools.trace import AgentTrace, StepType, TraceStep

# --- TraceStore protocol and implementations ---
# In v0.19.0 these ship as selectools.trace_store; here we define them
# inline to make the example self-contained and runnable today.


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for trace storage backends."""

    def save(self, trace: AgentTrace) -> str: ...

    def load(self, run_id: str) -> AgentTrace: ...

    def list_runs(
        self,
        *,
        limit: int = 50,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]: ...
    def delete(self, run_id: str) -> bool: ...


class InMemoryTraceStore:
    """In-memory trace store for development and testing."""

    def __init__(self) -> None:
        self._traces: Dict[str, AgentTrace] = {}

    def save(self, trace: AgentTrace) -> str:
        self._traces[trace.run_id] = trace
        return trace.run_id

    def load(self, run_id: str) -> AgentTrace:
        if run_id not in self._traces:
            raise ValueError(f"Trace {run_id!r} not found")
        return self._traces[run_id]

    def list_runs(
        self,
        *,
        limit: int = 50,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for trace in self._traces.values():
            if metadata_filter:
                if not all(trace.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            results.append(
                {
                    "run_id": trace.run_id,
                    "steps": len(trace.steps),
                    "total_ms": trace.total_duration_ms,
                    "metadata": trace.metadata,
                }
            )
        results.sort(key=lambda r: r.get("run_id", ""))
        return results[:limit]

    def delete(self, run_id: str) -> bool:
        return self._traces.pop(run_id, None) is not None

    def __len__(self) -> int:
        return len(self._traces)


class SQLiteTraceStore:
    """SQLite-backed trace store for persistent storage."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                run_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                trace_json TEXT NOT NULL
            )
        """
        )
        self._conn.commit()

    def save(self, trace: AgentTrace) -> str:
        import json

        self._conn.execute(
            "INSERT OR REPLACE INTO traces (run_id, start_time, metadata_json, trace_json) "
            "VALUES (?, ?, ?, ?)",
            (
                trace.run_id,
                trace.start_time,
                json.dumps(trace.metadata),
                json.dumps(trace.to_dict()),
            ),
        )
        self._conn.commit()
        return trace.run_id

    def load(self, run_id: str) -> AgentTrace:
        import json

        row = self._conn.execute(
            "SELECT trace_json FROM traces WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Trace {run_id!r} not found")
        return AgentTrace.from_dict(json.loads(row[0]))

    def list_runs(
        self,
        *,
        limit: int = 50,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        import json

        rows = self._conn.execute(
            "SELECT run_id, start_time, metadata_json, trace_json "
            "FROM traces ORDER BY start_time DESC LIMIT ?",
            (limit * 5,),  # over-fetch for filtering
        ).fetchall()

        results = []
        for run_id, _start_time, meta_json, trace_json in rows:
            meta = json.loads(meta_json)
            if metadata_filter:
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            trace_data = json.loads(trace_json)
            results.append(
                {
                    "run_id": run_id,
                    "steps": trace_data.get("step_count", 0),
                    "total_ms": trace_data.get("total_duration_ms", 0),
                    "metadata": meta,
                }
            )
            if len(results) >= limit:
                break
        return results

    def delete(self, run_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()


# --- Tools ---


@tool(description="Translate text to Spanish")
def translate_es(text: str) -> str:
    """Simulate translation to Spanish."""
    return f"[ES] {text}"


@tool(description="Translate text to French")
def translate_fr(text: str) -> str:
    """Simulate translation to French."""
    return f"[FR] {text}"


def make_sample_trace(
    run_id: str,
    metadata: Optional[Dict[str, str]] = None,
) -> AgentTrace:
    """Create a sample trace with realistic steps."""
    trace = AgentTrace(metadata=metadata or {})
    trace.run_id = run_id

    trace.add(
        TraceStep(
            type=StepType.LLM_CALL,
            model="gpt-5-mini",
            prompt_tokens=150,
            completion_tokens=50,
            duration_ms=320.5,
        )
    )
    trace.add(
        TraceStep(
            type=StepType.TOOL_EXECUTION,
            tool_name="translate_es",
            tool_result="[ES] Hello world",
            duration_ms=5.2,
        )
    )
    trace.add(
        TraceStep(
            type=StepType.LLM_CALL,
            model="gpt-5-mini",
            prompt_tokens=200,
            completion_tokens=80,
            duration_ms=410.1,
        )
    )
    return trace


def demo_inmemory():
    """Demonstrate InMemoryTraceStore."""
    print("=== InMemoryTraceStore ===")

    store = InMemoryTraceStore()

    # Save several traces with different metadata
    t1 = make_sample_trace("run-001", {"user_id": "u100", "env": "prod"})
    t2 = make_sample_trace("run-002", {"user_id": "u200", "env": "prod"})
    t3 = make_sample_trace("run-003", {"user_id": "u100", "env": "staging"})

    store.save(t1)
    store.save(t2)
    store.save(t3)
    print(f"Saved {len(store)} traces")

    # Load a specific trace
    loaded = store.load("run-001")
    print(f"\nLoaded run-001: {len(loaded)} steps, {loaded.total_duration_ms:.1f}ms")
    print(f"  Timeline:\n{loaded.timeline()}")

    # List all runs
    all_runs = store.list_runs()
    print(f"\nAll runs ({len(all_runs)}):")
    for r in all_runs:
        print(f"  {r['run_id']}: {r['steps']} steps, {r['total_ms']:.1f}ms, meta={r['metadata']}")

    # Filter by metadata
    prod_runs = store.list_runs(metadata_filter={"env": "prod"})
    print(f"\nProd runs: {len(prod_runs)}")
    for r in prod_runs:
        print(f"  {r['run_id']}: user={r['metadata'].get('user_id')}")

    user_runs = store.list_runs(metadata_filter={"user_id": "u100"})
    print(f"\nUser u100 runs: {len(user_runs)}")

    # Delete
    deleted = store.delete("run-002")
    print(f"\nDeleted run-002: {deleted}")
    print(f"Remaining: {len(store)} traces")


def demo_sqlite():
    """Demonstrate SQLiteTraceStore."""
    print("\n=== SQLiteTraceStore ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "traces.db")
        store = SQLiteTraceStore(db_path)
        print(f"Created SQLite store at: {db_path}")

        # Save traces
        for i in range(5):
            trace = make_sample_trace(
                f"sqlite-run-{i:03d}",
                {"batch": "demo", "index": str(i)},
            )
            store.save(trace)

        print(f"Saved 5 traces")

        # List
        runs = store.list_runs(limit=3)
        print(f"\nLast 3 runs:")
        for r in runs:
            print(f"  {r['run_id']}: {r['steps']} steps")

        # Filter
        filtered = store.list_runs(metadata_filter={"batch": "demo"})
        print(f"\nBatch 'demo' runs: {len(filtered)}")

        # Load and inspect
        loaded = store.load("sqlite-run-002")
        print(f"\nLoaded sqlite-run-002:")
        print(f"  Steps: {len(loaded)}")
        print(f"  LLM calls: {len(loaded.filter(type=StepType.LLM_CALL))}")
        print(f"  Tool calls: {len(loaded.filter(type=StepType.TOOL_EXECUTION))}")

        # Verify persistence
        store.close()
        store2 = SQLiteTraceStore(db_path)
        reloaded = store2.load("sqlite-run-002")
        print(f"\n  Reloaded after close: {len(reloaded)} steps (persistence works!)")
        store2.close()


def demo_agent_integration():
    """Show how to capture traces from real agent runs."""
    print("\n=== Agent Integration ===")

    store = InMemoryTraceStore()

    agent = Agent(
        tools=[translate_es, translate_fr],
        provider=LocalProvider(),
        config=AgentConfig(
            model="gpt-5-mini",
            max_iterations=3,
            trace_metadata={"user_id": "u100", "session": "demo"},
        ),
    )

    # Run the agent and capture traces
    result = agent.run("Translate 'hello' to Spanish")
    if result.trace:
        store.save(result.trace)
        print(f"Saved trace: {result.trace.run_id[:12]}...")
        print(f"  Steps: {len(result.trace)}")
        print(f"  Metadata: {result.trace.metadata}")

    result2 = agent.run("Now translate it to French")
    if result2.trace:
        store.save(result2.trace)
        print(f"Saved trace: {result2.trace.run_id[:12]}...")

    # Query saved traces
    all_traces = store.list_runs()
    print(f"\nTotal traces stored: {len(all_traces)}")
    for r in all_traces:
        print(f"  {r['run_id'][:12]}...: {r['steps']} steps, {r['total_ms']:.1f}ms")


def main() -> None:
    print("=" * 60)
    print("Trace Store Demo")
    print("=" * 60)

    demo_inmemory()
    demo_sqlite()
    demo_agent_integration()

    print("\nDone! TraceStore makes agent traces searchable and persistent.")


if __name__ == "__main__":
    main()
