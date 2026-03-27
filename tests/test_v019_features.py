"""Tests for v0.19.0 features: compose, retry/cache wrappers, trace store, streaming, type-safe."""

from __future__ import annotations

import asyncio
import os
import tempfile
import warnings

import pytest

from selectools.compose import compose
from selectools.observe import InMemoryTraceStore, JSONLTraceStore, SQLiteTraceStore, TraceFilter
from selectools.pipeline import Pipeline, Step, cache_step, retry, step
from selectools.trace import AgentTrace, StepType, TraceStep

# ===========================================================================
# compose()
# ===========================================================================


class TestCompose:
    def test_compose_two_functions(self):
        def upper(text: str) -> str:
            return text.upper()

        def exclaim(text: str) -> str:
            return text + "!"

        from selectools.tools.decorators import tool

        @tool(description="Uppercase")
        def tool_upper(text: str) -> str:
            return text.upper()

        @tool(description="Add exclamation")
        def tool_exclaim(text: str) -> str:
            return text + "!"

        composed = compose(tool_upper, tool_exclaim)
        assert composed.name == "tool_upper_then_tool_exclaim"
        result = composed.function(text="hello")
        assert result == "HELLO!"

    def test_compose_requires_two(self):
        from selectools.tools.decorators import tool

        @tool(description="Only one")
        def single(x: str) -> str:
            return x

        with pytest.raises(ValueError, match="at least 2"):
            compose(single)


# ===========================================================================
# retry() / cache_step()
# ===========================================================================


class TestRetryWrapper:
    def test_retry_wraps_step(self):
        @step
        def flaky(x: str) -> str:
            return x

        wrapped = retry(flaky, 3)
        assert wrapped.retry == 3

    def test_retry_in_pipeline(self):
        count = {"n": 0}

        def sometimes_fail(x: str) -> str:
            count["n"] += 1
            if count["n"] < 2:
                raise ValueError("not yet")
            return x.upper()

        pipeline = Pipeline(steps=[retry(sometimes_fail, 3)])
        result = pipeline.run("hello")
        assert result.output == "HELLO"


class TestCacheStep:
    def test_cache_returns_cached_value(self):
        call_count = {"n": 0}

        def expensive(x: str) -> str:
            call_count["n"] += 1
            return f"result:{x}"

        cached = cache_step(expensive, ttl=300)
        # First call
        r1 = cached("hello")
        # Second call — should be cached
        r2 = cached("hello")
        assert r1 == r2 == "result:hello"
        assert call_count["n"] == 1  # Only called once

    def test_cache_in_pipeline(self):
        call_count = {"n": 0}

        def expensive(x: str) -> str:
            call_count["n"] += 1
            return x.upper()

        pipeline = Pipeline(steps=[cache_step(expensive, ttl=300)])
        r1 = pipeline.run("hello")
        r2 = pipeline.run("hello")
        assert r1.output == "HELLO"
        assert r2.output == "HELLO"
        assert call_count["n"] == 1


# ===========================================================================
# Trace Store
# ===========================================================================


class TestInMemoryTraceStore:
    def test_save_and_load(self):
        store = InMemoryTraceStore()
        trace = AgentTrace(metadata={"test": True})
        trace.add(TraceStep(type=StepType.LLM_CALL))
        rid = store.save(trace)
        loaded = store.load(rid)
        assert len(loaded.steps) == 1
        assert loaded.metadata.get("test") is True

    def test_list(self):
        store = InMemoryTraceStore()
        for i in range(5):
            t = AgentTrace(metadata={"i": i})
            store.save(t)
        summaries = store.list(limit=3)
        assert len(summaries) == 3

    def test_delete(self):
        store = InMemoryTraceStore()
        t = AgentTrace()
        rid = store.save(t)
        assert store.delete(rid) is True
        assert store.delete(rid) is False

    def test_query_by_steps(self):
        store = InMemoryTraceStore()
        t1 = AgentTrace()
        t1.add(TraceStep(type=StepType.LLM_CALL))
        t2 = AgentTrace()
        for _ in range(5):
            t2.add(TraceStep(type=StepType.TOOL_EXECUTION))
        store.save(t1)
        store.save(t2)
        results = store.query(TraceFilter(min_steps=3))
        assert len(results) == 1

    def test_max_size_eviction(self):
        store = InMemoryTraceStore(max_size=3)
        ids = []
        for _ in range(5):
            ids.append(store.save(AgentTrace()))
        assert len(store.list(limit=100)) == 3
        with pytest.raises(ValueError):
            store.load(ids[0])  # Evicted


class TestSQLiteTraceStore:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteTraceStore(db_path)
            trace = AgentTrace(metadata={"env": "test"})
            trace.add(TraceStep(type=StepType.LLM_CALL))
            rid = store.save(trace)
            loaded = store.load(rid)
            assert len(loaded.steps) == 1
        finally:
            os.unlink(db_path)

    def test_list_and_delete(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteTraceStore(db_path)
            rid = store.save(AgentTrace())
            assert len(store.list()) == 1
            assert store.delete(rid) is True
            assert len(store.list()) == 0
        finally:
            os.unlink(db_path)


class TestJSONLTraceStore:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store = JSONLTraceStore(path)
            trace = AgentTrace(metadata={"format": "jsonl"})
            trace.add(TraceStep(type=StepType.TOOL_SELECTION))
            rid = store.save(trace)
            loaded = store.load(rid)
            assert len(loaded.steps) == 1
        finally:
            os.unlink(path)


# ===========================================================================
# Streaming composition
# ===========================================================================


class TestStreamingComposition:
    @pytest.mark.asyncio
    async def test_astream_yields_output(self):
        pipeline = Step(str.upper) | Step(lambda x: x + "!")
        chunks = []
        async for chunk in pipeline.astream("hello"):
            chunks.append(chunk)
        assert chunks == ["HELLO!"]

    @pytest.mark.asyncio
    async def test_astream_with_generator_last_step(self):
        def chunk_words(text: str):
            for word in text.split():
                yield word

        pipeline = Step(str.upper) | Step(chunk_words)
        chunks = []
        async for chunk in pipeline.astream("hello world"):
            chunks.append(chunk)
        assert chunks == ["HELLO", "WORLD"]

    @pytest.mark.asyncio
    async def test_astream_empty_pipeline(self):
        pipeline = Pipeline(steps=[])
        chunks = []
        async for chunk in pipeline.astream("passthrough"):
            chunks.append(chunk)
        assert chunks == ["passthrough"]


# ===========================================================================
# Type-safe contracts
# ===========================================================================


class TestTypeSafeContracts:
    def test_compatible_types_no_warning(self):
        @step
        def produce_str(x: str) -> str:
            return x.upper()

        @step
        def consume_str(x: str) -> str:
            return x + "!"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline = produce_str | consume_str
            type_warnings = [x for x in w if "type mismatch" in str(x.message)]
            assert len(type_warnings) == 0

    def test_incompatible_types_warns(self):
        @step
        def produce_int(x: str) -> int:
            return len(x)

        @step
        def consume_str(x: str) -> str:
            return x.upper()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline = produce_int | consume_str
            type_warnings = [x for x in w if "type mismatch" in str(x.message)]
            assert len(type_warnings) == 1
