# Trace Store Module

**Added in:** v0.19.0
**Package:** `src/selectools/observe/`
**Protocol:** `TraceStore`
**Classes:** `InMemoryTraceStore`, `SQLiteTraceStore`, `JSONLTraceStore`, `TraceSummary`, `TraceFilter`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [TraceStore Protocol](#tracestore-protocol)
4. [Backends](#backends)
5. [TraceFilter](#tracefilter)
6. [TraceSummary](#tracesummary)
7. [Integration with Serve](#integration-with-serve)
8. [Patterns](#patterns)
9. [API Reference](#api-reference)
10. [Examples](#examples)

---

## Overview

The **trace store** module persists and queries `AgentTrace` objects -- the detailed execution logs produced by every agent run. Instead of traces disappearing when the process exits, you can save them to SQLite, append them to a JSONL file, or hold them in memory for the duration of a session.

### Why Trace Store?

| Without Trace Store | With Trace Store |
|---|---|
| Traces lost on process exit | Persisted to disk or database |
| No way to compare runs | Query by date, step count, metadata |
| Manual logging of costs and durations | Automatic structured storage |
| No audit trail | Full execution history for compliance |

### Design Philosophy

- **Protocol-based.** `TraceStore` is a `Protocol` class. Implement 5 methods and any backend works.
- **Three built-in backends.** InMemory for dev, SQLite for production, JSONL for export/archive.
- **Thread-safe.** All backends use locks or WAL mode for concurrent access.
- **Zero required dependencies.** All three backends use Python stdlib only.

---

## Quick Start

```python
from selectools import Agent, AgentConfig
from selectools.observe import SQLiteTraceStore

# Create a trace store
store = SQLiteTraceStore("traces.db")

# Run an agent
agent = Agent(provider=provider, config=AgentConfig(model="gpt-4o"))
result = agent.run("What is the capital of France?")

# Save the trace
run_id = store.save(result.trace)
print(f"Saved trace: {run_id}")

# Load it back
trace = store.load(run_id)
print(f"Steps: {len(trace.steps)}, Duration: {trace.total_duration_ms:.0f}ms")

# List recent traces
for summary in store.list(limit=10):
    print(f"  {summary.run_id}: {summary.steps} steps, {summary.total_ms:.0f}ms")
```

---

## TraceStore Protocol

**File:** `src/selectools/observe/trace_store.py`

Any class that implements these 5 methods satisfies the `TraceStore` protocol:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TraceStore(Protocol):
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
```

The protocol is `@runtime_checkable`, so you can use `isinstance(obj, TraceStore)` to verify compliance.

---

## Backends

### InMemoryTraceStore

In-memory storage for development and testing. Traces are lost when the process exits.

```python
from selectools.observe import InMemoryTraceStore

store = InMemoryTraceStore(max_size=1000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_size` | `int` | `1000` | Maximum traces to keep. Oldest evicted when full. |

**Characteristics:**
- Thread-safe (uses `threading.Lock`).
- LRU eviction when `max_size` is exceeded.
- Fastest backend -- no I/O overhead.

### SQLiteTraceStore

SQLite-backed storage for production use. Uses WAL mode for concurrent read/write access.

```python
from selectools.observe import SQLiteTraceStore

store = SQLiteTraceStore("traces.db")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | (required) | Path to the SQLite database file. Created if it does not exist. |

**Characteristics:**
- Thread-safe (per-thread connections via `threading.local`).
- WAL journal mode for concurrent readers.
- Indexed on `created_at DESC` for fast listing.
- Traces serialized as JSON in the `trace_json` column.
- Supports SQL-level filtering for `min_steps`, `max_steps`, `since`, `until`.
- Metadata filtering done in Python (not SQL) for flexibility.

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS traces (
    run_id TEXT PRIMARY KEY,
    steps INTEGER NOT NULL,
    total_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
```

### JSONLTraceStore

Append-only JSONL (JSON Lines) storage for export and archival. Each trace is stored as a single JSON line.

```python
from selectools.observe import JSONLTraceStore

store = JSONLTraceStore("traces.jsonl")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | (required) | Path to the JSONL file. Parent directories created automatically. |

**Characteristics:**
- Thread-safe (uses `threading.Lock`).
- Append-only writes -- fast saves with no seek overhead.
- Delete rewrites the entire file (expensive, by design -- JSONL is for archival).
- Human-readable -- each line is valid JSON.
- Easy to process with standard tools (`jq`, `grep`, Python line iteration).

**Line format:**

```json
{"run_id": "abc-123", "steps": 5, "total_ms": 1234.5, "created_at": "2026-03-27T10:00:00+00:00", "metadata": {"user_id": "alice"}, "trace": {...}}
```

### Choosing a Backend

| Backend | Best For | Persistence | Query Speed | Write Speed |
|---|---|---|---|---|
| `InMemoryTraceStore` | Dev, testing, short-lived processes | None | Fast | Fast |
| `SQLiteTraceStore` | Production, dashboards, analytics | Disk | Fast (indexed) | Fast (WAL) |
| `JSONLTraceStore` | Export, archival, log shipping | Disk | Slow (full scan) | Fast (append) |

---

## TraceFilter

Filter criteria for querying traces. All fields are optional -- unset fields are not applied.

```python
from selectools.observe import TraceFilter
from datetime import datetime, timezone, timedelta

# Traces from the last 24 hours with at least 3 steps
filters = TraceFilter(
    since=datetime.now(timezone.utc) - timedelta(hours=24),
    min_steps=3,
)
results = store.query(filters)

# Traces for a specific user
filters = TraceFilter(
    metadata_match={"user_id": "alice"},
)
results = store.query(filters)

# Combine all criteria
filters = TraceFilter(
    metadata_match={"environment": "production"},
    min_steps=2,
    max_steps=20,
    since=datetime(2026, 3, 1, tzinfo=timezone.utc),
    until=datetime(2026, 3, 31, tzinfo=timezone.utc),
)
results = store.query(filters)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `metadata_match` | `Optional[Dict[str, Any]]` | `None` | All key-value pairs must match the trace's metadata. |
| `min_steps` | `Optional[int]` | `None` | Minimum number of trace steps (inclusive). |
| `max_steps` | `Optional[int]` | `None` | Maximum number of trace steps (inclusive). |
| `since` | `Optional[datetime]` | `None` | Only traces created at or after this time. |
| `until` | `Optional[datetime]` | `None` | Only traces created at or before this time. |

---

## TraceSummary

A lightweight summary of a stored trace, returned by `list()` and `query()`. Avoids loading the full trace JSON for listing operations.

```python
for summary in store.list(limit=10):
    print(f"Run: {summary.run_id}")
    print(f"  Steps: {summary.steps}")
    print(f"  Duration: {summary.total_ms:.0f}ms")
    print(f"  Created: {summary.created_at}")
    print(f"  Metadata: {summary.metadata}")
```

### Fields

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | Unique run identifier. |
| `steps` | `int` | Number of trace steps. |
| `total_ms` | `float` | Total execution duration in milliseconds. |
| `created_at` | `datetime` | When the trace was saved (UTC). |
| `metadata` | `Dict[str, Any]` | Trace metadata (user_id, environment, etc.). |

---

## Integration with Serve

When using the [Serve Module](SERVE.md), traces can be persisted automatically and queried via HTTP. Wire a `TraceStore` into your served agent for full observability:

```python
from selectools import Agent, AgentConfig
from selectools.observe import SQLiteTraceStore
from selectools.serve import AgentRouter, create_app

# Agent with trace store
store = SQLiteTraceStore("traces.db")
agent = Agent(provider=provider, config=AgentConfig(model="gpt-4o"))

# Custom endpoint that saves traces
router = AgentRouter(agent)

# After each invoke, save the trace
original_invoke = router.handle_invoke

def invoke_with_trace(body):
    result = original_invoke(body)
    # The agent's last trace is available after run()
    if hasattr(agent, '_last_trace') and agent._last_trace:
        store.save(agent._last_trace)
    return result

router.handle_invoke = invoke_with_trace

app = create_app(agent)
app.serve()
```

### Querying Traces Programmatically

```python
# List recent traces
traces = store.list(limit=20)
for t in traces:
    print(f"{t.run_id}: {t.steps} steps, {t.total_ms:.0f}ms")

# Find slow traces
slow = store.query(TraceFilter(min_steps=10))
for t in slow:
    trace = store.load(t.run_id)
    for step in trace.steps:
        if step.duration_ms and step.duration_ms > 5000:
            print(f"  Slow step: {step.type.value} at {step.node_name} ({step.duration_ms:.0f}ms)")
```

---

## Patterns

### Automatic Trace Saving with Observer

Use an `AgentObserver` to save traces automatically after every run:

```python
from selectools import AgentObserver

class TraceSavingObserver(AgentObserver):
    def __init__(self, store: TraceStore):
        self.store = store

    def on_run_end(self, run_id, result, duration_ms):
        if result.trace:
            self.store.save(result.trace)

store = SQLiteTraceStore("traces.db")
agent = Agent(
    provider=provider,
    config=AgentConfig(model="gpt-4o"),
    observers=[TraceSavingObserver(store)],
)
```

### Cost Analytics

Query traces to compute cost analytics:

```python
from selectools.observe import TraceFilter, SQLiteTraceStore
from datetime import datetime, timezone, timedelta

store = SQLiteTraceStore("traces.db")

# Get all traces from the last 7 days
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
traces = store.query(TraceFilter(since=week_ago))

total_ms = sum(t.total_ms for t in traces)
avg_steps = sum(t.steps for t in traces) / max(len(traces), 1)

print(f"Runs: {len(traces)}")
print(f"Total duration: {total_ms / 1000:.1f}s")
print(f"Avg steps per run: {avg_steps:.1f}")
```

### Export to JSONL for Analysis

Use `JSONLTraceStore` to export traces for offline analysis:

```python
from selectools.observe import SQLiteTraceStore, JSONLTraceStore

sqlite_store = SQLiteTraceStore("traces.db")
export_store = JSONLTraceStore("export/traces_2026_03.jsonl")

# Export all traces from March
from datetime import datetime, timezone
filters = TraceFilter(
    since=datetime(2026, 3, 1, tzinfo=timezone.utc),
    until=datetime(2026, 3, 31, tzinfo=timezone.utc),
)

for summary in sqlite_store.query(filters):
    trace = sqlite_store.load(summary.run_id)
    export_store.save(trace)

print(f"Exported {len(sqlite_store.query(filters))} traces")
```

### Custom Backend

Implement the `TraceStore` protocol for any storage system:

```python
class RedisTraceStore:
    """Redis-backed trace store (example)."""

    def __init__(self, redis_client, prefix="trace:"):
        self.redis = redis_client
        self.prefix = prefix

    def save(self, trace: AgentTrace) -> str:
        run_id = trace.run_id
        data = json.dumps(trace.to_dict(), default=str)
        self.redis.set(f"{self.prefix}{run_id}", data)
        self.redis.zadd(f"{self.prefix}index", {run_id: time.time()})
        return run_id

    def load(self, run_id: str) -> AgentTrace:
        data = self.redis.get(f"{self.prefix}{run_id}")
        if data is None:
            raise ValueError(f"Trace {run_id!r} not found")
        return AgentTrace.from_dict(json.loads(data))

    def list(self, limit=50, offset=0):
        ids = self.redis.zrevrange(f"{self.prefix}index", offset, offset + limit - 1)
        return [self._to_summary(rid.decode()) for rid in ids]

    def query(self, filters: TraceFilter):
        # Implement filtering logic
        ...

    def delete(self, run_id: str) -> bool:
        result = self.redis.delete(f"{self.prefix}{run_id}")
        self.redis.zrem(f"{self.prefix}index", run_id)
        return result > 0
```

---

## API Reference

### InMemoryTraceStore.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_size` | `int` | `1000` | Maximum traces to store before LRU eviction. |

### SQLiteTraceStore.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | (required) | Path to SQLite database file. |

### JSONLTraceStore.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | (required) | Path to JSONL file. |

### TraceStore Methods

| Method | Description |
|---|---|
| `save(trace)` | Persist a trace. Returns `run_id` (str). |
| `load(run_id)` | Load full trace by run_id. Raises `ValueError` if not found. |
| `list(limit=50, offset=0)` | List `TraceSummary` objects, newest first. |
| `query(filters)` | Query traces matching a `TraceFilter`. Returns `List[TraceSummary]`. |
| `delete(run_id)` | Delete a trace. Returns `True` if deleted. |

### TraceFilter Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `metadata_match` | `Optional[Dict[str, Any]]` | `None` | Metadata key-value pairs to match. |
| `min_steps` | `Optional[int]` | `None` | Minimum step count. |
| `max_steps` | `Optional[int]` | `None` | Maximum step count. |
| `since` | `Optional[datetime]` | `None` | Created at or after. |
| `until` | `Optional[datetime]` | `None` | Created at or before. |

### TraceSummary Fields

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | Unique run identifier. |
| `steps` | `int` | Number of trace steps. |
| `total_ms` | `float` | Total duration in milliseconds. |
| `created_at` | `datetime` | When the trace was saved. |
| `metadata` | `Dict[str, Any]` | Trace metadata. |

---

## Examples

| Example | File | Description |
|---|---|---|
| 69 | [`69_trace_store.py`](https://github.com/johnnichev/selectools/blob/main/examples/69_trace_store.py) | Save, query, and export traces with all 3 backends |

---

## Further Reading

- [Agent Module](AGENT.md) -- The Agent class that produces traces
- [Serve Module](SERVE.md) -- Deploy agents as HTTP APIs with trace persistence
- [Audit Module](AUDIT.md) -- JSONL audit logging (complementary to trace storage)
- [Usage Module](USAGE.md) -- Token and cost tracking

---

**Next Steps:** Learn about deploying agents with trace persistence in the [Serve Module](SERVE.md).
