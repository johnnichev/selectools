"""Tests for trace_store — InMemoryTraceStore, SQLiteTraceStore, JSONLTraceStore."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from selectools.observe.trace_store import (
    InMemoryTraceStore,
    JSONLTraceStore,
    SQLiteTraceStore,
    TraceFilter,
    TraceStore,
    TraceSummary,
)
from selectools.trace import AgentTrace, StepType, TraceStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(metadata=None, n_steps=2):
    """Create a minimal trace with a few steps."""
    trace = AgentTrace(metadata=metadata or {})
    for i in range(n_steps):
        trace.add(TraceStep(type=StepType.LLM_CALL, duration_ms=10.0 * (i + 1)))
    return trace


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------


class TestTraceStoreProtocol:
    def test_inmemory_is_trace_store(self):
        assert isinstance(InMemoryTraceStore(), TraceStore)

    def test_sqlite_is_trace_store(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "test.db"))
        assert isinstance(store, TraceStore)

    def test_jsonl_is_trace_store(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "test.jsonl"))
        assert isinstance(store, TraceStore)


# ---------------------------------------------------------------------------
# InMemoryTraceStore
# ---------------------------------------------------------------------------


class TestInMemoryTraceStore:
    def test_save_and_load(self):
        store = InMemoryTraceStore()
        trace = _make_trace()
        run_id = store.save(trace)
        assert run_id == trace.run_id

        loaded = store.load(run_id)
        assert loaded.run_id == trace.run_id
        assert len(loaded.steps) == 2

    def test_load_nonexistent_raises(self):
        store = InMemoryTraceStore()
        with pytest.raises(ValueError, match="not found"):
            store.load("nonexistent-id")

    def test_list_returns_newest_first(self):
        store = InMemoryTraceStore()
        ids = []
        for i in range(5):
            t = _make_trace(n_steps=i + 1)
            ids.append(store.save(t))

        summaries = store.list(limit=3)
        assert len(summaries) == 3
        # Newest first
        assert summaries[0].run_id == ids[-1]
        assert summaries[1].run_id == ids[-2]

    def test_list_with_offset(self):
        store = InMemoryTraceStore()
        for _ in range(5):
            store.save(_make_trace())
        summaries = store.list(limit=2, offset=3)
        assert len(summaries) == 2

    def test_delete_existing(self):
        store = InMemoryTraceStore()
        trace = _make_trace()
        run_id = store.save(trace)
        assert store.delete(run_id) is True
        with pytest.raises(ValueError):
            store.load(run_id)

    def test_delete_nonexistent(self):
        store = InMemoryTraceStore()
        assert store.delete("no-such-id") is False

    def test_max_size_evicts_oldest(self):
        store = InMemoryTraceStore(max_size=3)
        ids = []
        for _ in range(5):
            ids.append(store.save(_make_trace()))
        with pytest.raises(ValueError):
            store.load(ids[0])
        with pytest.raises(ValueError):
            store.load(ids[1])
        store.load(ids[2])
        store.load(ids[3])
        store.load(ids[4])

    def test_query_with_min_steps(self):
        store = InMemoryTraceStore()
        store.save(_make_trace(n_steps=1))
        store.save(_make_trace(n_steps=5))
        store.save(_make_trace(n_steps=3))

        results = store.query(TraceFilter(min_steps=3))
        assert len(results) == 2
        assert all(s.steps >= 3 for s in results)

    def test_query_with_max_steps(self):
        store = InMemoryTraceStore()
        store.save(_make_trace(n_steps=1))
        store.save(_make_trace(n_steps=5))
        results = store.query(TraceFilter(max_steps=2))
        assert len(results) == 1

    def test_query_with_since_until(self):
        store = InMemoryTraceStore()
        store.save(_make_trace())

        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        results = store.query(TraceFilter(since=past, until=future))
        assert len(results) == 1

        results = store.query(TraceFilter(since=future))
        assert len(results) == 0

    def test_query_with_metadata_match(self):
        store = InMemoryTraceStore()
        store.save(_make_trace(metadata={"env": "prod"}))
        store.save(_make_trace(metadata={"env": "dev"}))
        store.save(_make_trace(metadata={"env": "prod", "version": "2"}))

        results = store.query(TraceFilter(metadata_match={"env": "prod"}))
        assert len(results) == 2

    def test_summary_fields(self):
        store = InMemoryTraceStore()
        trace = _make_trace(n_steps=3, metadata={"user": "test"})
        store.save(trace)

        summaries = store.list()
        assert len(summaries) == 1
        s = summaries[0]
        assert s.run_id == trace.run_id
        assert s.steps == 3
        assert s.total_ms == trace.total_duration_ms
        assert isinstance(s.created_at, datetime)
        assert s.metadata == {"user": "test"}

    def test_inmemory_preserves_exact_trace(self):
        store = InMemoryTraceStore()
        trace = _make_trace(metadata={"x": 1})
        store.save(trace)
        loaded = store.load(trace.run_id)
        assert loaded is trace


# ---------------------------------------------------------------------------
# SQLiteTraceStore
# ---------------------------------------------------------------------------


class TestSQLiteTraceStore:
    def test_save_and_load(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = _make_trace(metadata={"key": "val"})
        run_id = store.save(trace)
        assert run_id == trace.run_id

        loaded = store.load(run_id)
        assert loaded.run_id == trace.run_id
        assert len(loaded.steps) == 2

    def test_load_nonexistent_raises(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        with pytest.raises(ValueError, match="not found"):
            store.load("no-such-id")

    def test_list_returns_newest_first(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        ids = []
        for i in range(5):
            t = _make_trace(n_steps=i + 1)
            ids.append(store.save(t))

        summaries = store.list(limit=3)
        assert len(summaries) == 3
        assert summaries[0].run_id == ids[-1]

    def test_list_with_offset(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        for _ in range(5):
            store.save(_make_trace())
        summaries = store.list(limit=2, offset=3)
        assert len(summaries) == 2

    def test_delete_existing(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = _make_trace()
        run_id = store.save(trace)
        assert store.delete(run_id) is True
        with pytest.raises(ValueError):
            store.load(run_id)

    def test_delete_nonexistent(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        assert store.delete("no-such-id") is False

    def test_save_replaces_existing(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = _make_trace(n_steps=2)
        store.save(trace)
        trace.add(TraceStep(type=StepType.ERROR, duration_ms=5.0))
        store.save(trace)
        loaded = store.load(trace.run_id)
        assert len(loaded.steps) == 3

    def test_query_with_min_max_steps(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        store.save(_make_trace(n_steps=1))
        store.save(_make_trace(n_steps=5))
        store.save(_make_trace(n_steps=3))

        results = store.query(TraceFilter(min_steps=2, max_steps=4))
        assert len(results) == 1
        assert results[0].steps == 3

    def test_query_with_since_until(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        store.save(_make_trace())

        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        results = store.query(TraceFilter(since=past, until=future))
        assert len(results) == 1

        results = store.query(TraceFilter(since=future))
        assert len(results) == 0

    def test_query_with_metadata_match(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        store.save(_make_trace(metadata={"env": "prod"}))
        store.save(_make_trace(metadata={"env": "dev"}))
        store.save(_make_trace(metadata={"env": "prod", "region": "us"}))

        results = store.query(TraceFilter(metadata_match={"env": "prod"}))
        assert len(results) == 2

    def test_query_empty_filters(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        store.save(_make_trace())
        store.save(_make_trace())
        results = store.query(TraceFilter())
        assert len(results) == 2

    def test_row_to_summary_bad_metadata(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = _make_trace()
        store.save(trace)
        store._conn().execute(
            "UPDATE traces SET metadata = ? WHERE run_id = ?",
            ("not-json", trace.run_id),
        )
        store._conn().commit()
        summaries = store.list()
        assert len(summaries) == 1
        assert summaries[0].metadata == {}

    def test_row_to_summary_empty_metadata(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = _make_trace()
        store.save(trace)
        store._conn().execute(
            "UPDATE traces SET metadata = ? WHERE run_id = ?",
            ("", trace.run_id),
        )
        store._conn().commit()
        summaries = store.list()
        assert summaries[0].metadata == {}

    def test_round_trip_preserves_step_data(self, tmp_path):
        store = SQLiteTraceStore(str(tmp_path / "traces.db"))
        trace = AgentTrace(metadata={"env": "test"})
        trace.add(
            TraceStep(
                type=StepType.TOOL_EXECUTION,
                duration_ms=42.0,
                tool_name="calculator",
                tool_args={"expr": "1+1"},
                tool_result="2",
            )
        )
        store.save(trace)
        loaded = store.load(trace.run_id)
        assert len(loaded.steps) == 1
        assert loaded.steps[0].tool_name == "calculator"
        assert loaded.steps[0].tool_result == "2"


# ---------------------------------------------------------------------------
# JSONLTraceStore
# ---------------------------------------------------------------------------


class TestJSONLTraceStore:
    def test_save_and_load(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        trace = _make_trace(metadata={"user": "test"})
        run_id = store.save(trace)
        assert run_id == trace.run_id

        loaded = store.load(run_id)
        assert loaded.run_id == trace.run_id
        assert len(loaded.steps) == 2

    def test_load_nonexistent_raises(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        with pytest.raises(ValueError, match="not found"):
            store.load("no-such-id")

    def test_load_from_nonexistent_file(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        with pytest.raises(ValueError, match="not found"):
            store.load("nonexistent")

    def test_list_returns_newest_first(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        ids = []
        for i in range(4):
            t = _make_trace(n_steps=i + 1)
            ids.append(store.save(t))

        summaries = store.list(limit=2)
        assert len(summaries) == 2
        assert summaries[0].run_id == ids[-1]

    def test_list_with_offset(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        for _ in range(5):
            store.save(_make_trace())
        summaries = store.list(limit=2, offset=3)
        assert len(summaries) == 2

    def test_delete_existing(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        trace = _make_trace()
        run_id = store.save(trace)
        assert store.delete(run_id) is True
        with pytest.raises(ValueError):
            store.load(run_id)

    def test_delete_nonexistent(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        store.save(_make_trace())
        assert store.delete("no-such-id") is False

    def test_delete_from_nonexistent_file(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        assert store.delete("any-id") is False

    def test_query_with_filters(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        store.save(_make_trace(n_steps=1, metadata={"env": "dev"}))
        store.save(_make_trace(n_steps=5, metadata={"env": "prod"}))
        store.save(_make_trace(n_steps=3, metadata={"env": "prod"}))

        results = store.query(TraceFilter(min_steps=2, metadata_match={"env": "prod"}))
        assert len(results) == 2

    def test_query_with_max_steps(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        store.save(_make_trace(n_steps=1))
        store.save(_make_trace(n_steps=5))
        results = store.query(TraceFilter(max_steps=2))
        assert len(results) == 1

    def test_query_with_since_until(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        store.save(_make_trace())

        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        results = store.query(TraceFilter(since=past, until=future))
        assert len(results) == 1

        results = store.query(TraceFilter(since=future))
        assert len(results) == 0

    def test_iter_entries_skips_bad_json(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        store = JSONLTraceStore(str(path))
        store.save(_make_trace())
        with open(path, "a") as f:
            f.write("NOT JSON\n")
        summaries = store.list()
        assert len(summaries) == 1

    def test_iter_entries_skips_blank_lines(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        store = JSONLTraceStore(str(path))
        store.save(_make_trace())
        with open(path, "a") as f:
            f.write("\n\n\n")
        summaries = store.list()
        assert len(summaries) == 1

    def test_parent_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        store = JSONLTraceStore(str(nested / "traces.jsonl"))
        assert nested.exists()

    def test_round_trip_preserves_step_data(self, tmp_path):
        store = JSONLTraceStore(str(tmp_path / "traces.jsonl"))
        trace = AgentTrace(metadata={"env": "staging"})
        trace.add(
            TraceStep(
                type=StepType.LLM_CALL,
                duration_ms=100.0,
                model="gpt-4o",
                prompt_tokens=50,
                completion_tokens=25,
            )
        )
        store.save(trace)
        loaded = store.load(trace.run_id)
        assert loaded.steps[0].model == "gpt-4o"
        assert loaded.steps[0].prompt_tokens == 50
