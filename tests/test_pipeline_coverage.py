"""Additional pipeline tests for coverage gaps — parallel(), branch(), error paths, astream."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from selectools.pipeline import (
    Pipeline,
    Step,
    StepResult,
    _ensure_step,
    _filter_kwargs,
    _get_type_hints,
    _is_subtype,
    branch,
    cache_step,
    parallel,
    retry,
    step,
)

# ---------------------------------------------------------------------------
# _filter_kwargs edge cases
# ---------------------------------------------------------------------------


class TestFilterKwargs:
    def test_empty_kwargs(self):
        def fn(x: str) -> str:
            return x

        assert _filter_kwargs(fn, {}) == {}

    def test_var_keyword_passes_all(self):
        def fn(x: str, **kw) -> str:
            return x

        kwargs = {"a": 1, "b": 2}
        assert _filter_kwargs(fn, kwargs) == kwargs

    def test_no_var_keyword_filters(self):
        def fn(x: str, lang: str = "en") -> str:
            return x

        result = _filter_kwargs(fn, {"lang": "fr", "extra": "ignored"})
        assert result == {"lang": "fr"}

    def test_uninspectable_function(self):
        # Built-in len has a known signature in Python 3.9+, so it filters.
        # Just verify it doesn't crash on built-ins.
        result = _filter_kwargs(len, {"a": 1})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _is_subtype edge cases
# ---------------------------------------------------------------------------


class TestIsSubtype:
    def test_same_type(self):
        assert _is_subtype(str, str) is True

    def test_subclass(self):
        assert _is_subtype(bool, int) is True

    def test_any_input(self):
        assert _is_subtype(Any, str) is True

    def test_any_expected(self):
        assert _is_subtype(str, Any) is True

    def test_generic_origin(self):
        assert _is_subtype(List[int], Dict[str, int]) is True  # Can't validate generics


# ---------------------------------------------------------------------------
# _get_type_hints
# ---------------------------------------------------------------------------


class TestGetTypeHints:
    def test_annotated_function(self):
        def fn(x: str) -> int:
            return 0

        hints = _get_type_hints(fn)
        assert hints.get("input") is str
        assert hints.get("return") is int

    def test_unannotated_function(self):
        def fn(x):
            return x

        hints = _get_type_hints(fn)
        assert hints == {}

    def test_return_only(self):
        def fn(x) -> str:
            return ""

        hints = _get_type_hints(fn)
        assert hints.get("return") is str
        assert "input" not in hints


# ---------------------------------------------------------------------------
# _ensure_step
# ---------------------------------------------------------------------------


class TestEnsureStep:
    def test_step_passthrough(self):
        s = Step(lambda x: x)
        assert _ensure_step(s) is s

    def test_pipeline_passthrough(self):
        p = Pipeline(steps=[])
        result = _ensure_step(p)
        assert result is p

    def test_callable_wrapped(self):
        def fn(x):
            return x

        result = _ensure_step(fn)
        assert isinstance(result, Step)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="Expected callable"):
            _ensure_step(42)


# ---------------------------------------------------------------------------
# Step composition edge cases
# ---------------------------------------------------------------------------


class TestStepComposition:
    def test_ror_with_callable(self):
        """callable | step works via __ror__."""

        def lower(x: str) -> str:
            return x.lower()

        s = Step(lambda x: x + "!", name="exclaim")
        pipeline = lower | s
        assert isinstance(pipeline, Pipeline)
        result = pipeline.run("HELLO")
        assert result.output == "hello!"

    def test_ror_with_pipeline(self):
        """pipeline | step works."""
        p = Pipeline(steps=[Step(lambda x: x.upper(), name="upper")])
        s = Step(lambda x: x + "!", name="exclaim")
        combined = p | s
        assert isinstance(combined, Pipeline)
        result = combined.run("hello")
        assert result.output == "HELLO!"

    def test_pipeline_ror_with_pipeline(self):
        """pipeline1 __ror__ pipeline2 works."""
        p1 = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        p2 = Pipeline(steps=[Step(lambda x: x + "!", name="exclaim")])
        # Test __ror__: p1 | p2 should combine them
        # This goes through Pipeline.__or__ on p1
        combined = p1 | p2
        result = combined.run("hello")
        assert result.output == "HELLO!"

    def test_pipeline_append_callable(self):
        p = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        combined = p | (lambda x: x + "?")
        result = combined.run("hi")
        assert result.output == "HI?"

    def test_pipeline_invalid_step_type(self):
        with pytest.raises(TypeError, match="Pipeline step must be callable"):
            Pipeline(steps=[42])  # type: ignore

    def test_append_invalid_type(self):
        p = Pipeline(steps=[])
        with pytest.raises(TypeError, match="Cannot compose"):
            p._append(42)  # type: ignore


# ---------------------------------------------------------------------------
# Pipeline type contract warnings
# ---------------------------------------------------------------------------


class TestTypeContracts:
    def test_type_mismatch_warns(self):
        s1 = Step(lambda x: 42, name="to_int", output_type=int)
        s2 = Step(lambda x: x, name="expect_str", input_type=str)

        with pytest.warns(UserWarning, match="type mismatch"):
            Pipeline(steps=[s1, s2])

    def test_compatible_types_no_warning(self):
        s1 = Step(lambda x: "hello", name="to_str", output_type=str)
        s2 = Step(lambda x: x, name="take_str", input_type=str)
        # Should not warn
        Pipeline(steps=[s1, s2])


# ---------------------------------------------------------------------------
# Error handling — retry + on_error in pipeline
# ---------------------------------------------------------------------------


class TestPipelineErrorHandling:
    def test_retry_exhausted_with_skip(self):
        """When retry exhausted and on_error=skip, step is skipped."""
        call_count = {"n": 0}

        def always_fail(x: str) -> str:
            call_count["n"] += 1
            raise ValueError("always fails")

        s = Step(always_fail, retry=2, on_error="skip")
        pipeline = Pipeline(steps=[s, Step(lambda x: x + "!", name="exclaim")])
        result = pipeline.run("hello")
        # Should have called 1 + 2 retries = 3 times total
        assert call_count["n"] == 3
        # Input passes through to exclaim
        assert result.output == "hello!"

    def test_error_with_skip_no_retry(self):
        """on_error=skip without retry skips the step."""

        def fail(x: str) -> str:
            raise ValueError("fail")

        pipeline = Pipeline(
            steps=[
                Step(fail, on_error="skip"),
                Step(lambda x: x + "!", name="exclaim"),
            ]
        )
        result = pipeline.run("hello")
        assert result.output == "hello!"

    def test_retry_succeeds_on_second_attempt(self):
        state = {"calls": 0}

        def flaky(x: str) -> str:
            state["calls"] += 1
            if state["calls"] < 2:
                raise ValueError("not yet")
            return x.upper()

        pipeline = Pipeline(steps=[Step(flaky, retry=3)])
        result = pipeline.run("test")
        assert result.output == "TEST"
        assert any(t.get("retry") for t in result.trace)


# ---------------------------------------------------------------------------
# Async pipeline error handling
# ---------------------------------------------------------------------------


class TestAsyncPipelineErrors:
    @pytest.mark.asyncio
    async def test_arun_error_raises(self):
        async def fail(x: str) -> str:
            raise ValueError("async boom")

        pipeline = Pipeline(steps=[Step(fail)])
        with pytest.raises(ValueError, match="async boom"):
            await pipeline.arun("test")

    @pytest.mark.asyncio
    async def test_arun_retry_succeeds(self):
        state = {"calls": 0}

        async def flaky(x: str) -> str:
            state["calls"] += 1
            if state["calls"] < 2:
                raise ValueError("flaky")
            return x.upper()

        pipeline = Pipeline(steps=[Step(flaky, retry=3)])
        result = await pipeline.arun("hello")
        assert result.output == "HELLO"

    @pytest.mark.asyncio
    async def test_arun_retry_exhausted_raises(self):
        async def always_fail(x: str) -> str:
            raise ValueError("always")

        pipeline = Pipeline(steps=[Step(always_fail, retry=2)])
        with pytest.raises(ValueError, match="always"):
            await pipeline.arun("test")

    @pytest.mark.asyncio
    async def test_arun_retry_exhausted_with_skip(self):
        async def always_fail(x: str) -> str:
            raise ValueError("fail")

        pipeline = Pipeline(
            steps=[
                Step(always_fail, retry=1, on_error="skip"),
                Step(lambda x: x + "!", name="exclaim"),
            ]
        )
        result = await pipeline.arun("hello")
        assert result.output == "hello!"

    @pytest.mark.asyncio
    async def test_arun_skip_no_retry(self):
        async def fail(x: str) -> str:
            raise ValueError("fail")

        pipeline = Pipeline(
            steps=[
                Step(fail, on_error="skip"),
                Step(lambda x: x + "!", name="exclaim"),
            ]
        )
        result = await pipeline.arun("hello")
        assert result.output == "hello!"


# ---------------------------------------------------------------------------
# astream
# ---------------------------------------------------------------------------


class TestPipelineAstream:
    @pytest.mark.asyncio
    async def test_astream_basic(self):
        pipeline = Step(lambda x: x.upper(), name="up") | Step(lambda x: x + "!", name="exclaim")
        chunks = []
        async for chunk in pipeline.astream("hello"):
            chunks.append(chunk)
        assert chunks == ["HELLO!"]

    @pytest.mark.asyncio
    async def test_astream_empty_pipeline(self):
        pipeline = Pipeline(steps=[])
        chunks = []
        async for chunk in pipeline.astream("pass"):
            chunks.append(chunk)
        assert chunks == ["pass"]

    @pytest.mark.asyncio
    async def test_astream_async_last_step(self):
        async def async_upper(x: str) -> str:
            return x.upper()

        pipeline = Pipeline(steps=[Step(async_upper)])
        chunks = []
        async for chunk in pipeline.astream("hello"):
            chunks.append(chunk)
        assert chunks == ["HELLO"]

    @pytest.mark.asyncio
    async def test_astream_generator_last_step(self):
        def gen(x: str):
            for ch in x:
                yield ch.upper()

        pipeline = Pipeline(steps=[Step(gen)])
        chunks = []
        async for chunk in pipeline.astream("abc"):
            chunks.append(chunk)
        assert chunks == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_astream_async_generator_last_step(self):
        async def agen(x: str):
            for ch in x:
                yield ch.upper()

        pipeline = Pipeline(steps=[Step(agen)])
        chunks = []
        async for chunk in pipeline.astream("ab"):
            chunks.append(chunk)
        assert chunks == ["A", "B"]


# ---------------------------------------------------------------------------
# parallel() edge cases
# ---------------------------------------------------------------------------


class TestParallelCoverage:
    @pytest.mark.asyncio
    async def test_parallel_async_steps(self):
        async def a(x: str) -> str:
            return f"a:{x}"

        async def b(x: str) -> str:
            return f"b:{x}"

        p = parallel(a, b)
        result = await p("test")
        assert result["a"] == "a:test"
        assert result["b"] == "b:test"

    def test_parallel_with_kwargs(self):
        def fn_with_kw(x: str, tag: str = "default") -> str:
            return f"{tag}:{x}"

        def fn_no_kw(x: str) -> str:
            return x

        p = parallel(fn_with_kw, fn_no_kw)
        result = p("input", tag="v1")
        assert result["fn_with_kw"] == "v1:input"
        assert result["fn_no_kw"] == "input"


# ---------------------------------------------------------------------------
# branch() edge cases
# ---------------------------------------------------------------------------


class TestBranchCoverage:
    def test_branch_with_dict_input(self):
        @step()
        def handle(x) -> str:
            return f"handled:{x}"

        b = branch(a=handle)
        result = b({"branch": "a"})
        assert "handled" in result

    def test_branch_no_router_non_string_raises(self):
        @step()
        def handle(x) -> str:
            return "ok"

        b = branch(a=handle)
        with pytest.raises(ValueError, match="no router function"):
            b(42)

    def test_branch_key_not_found_no_default(self):
        @step()
        def only_a(x) -> str:
            return "a"

        b = branch(a=only_a)
        with pytest.raises(KeyError, match="no matching branch"):
            b("unknown_key")


# ---------------------------------------------------------------------------
# retry() and cache_step() wrappers
# ---------------------------------------------------------------------------


class TestRetryWrapper:
    def test_retry_wraps_step(self):
        @step()
        def fn(x: str) -> str:
            return x

        wrapped = retry(fn, attempts=5)
        assert isinstance(wrapped, Step)
        assert wrapped.retry == 5

    def test_retry_wraps_callable(self):
        def fn(x: str) -> str:
            return x

        wrapped = retry(fn, attempts=3)
        assert isinstance(wrapped, Step)
        assert wrapped.retry == 3


class TestCacheStep:
    def test_cache_step_returns_cached(self):
        call_count = {"n": 0}

        def expensive(x: str) -> str:
            call_count["n"] += 1
            return x.upper()

        cached = cache_step(expensive, ttl=300)
        assert cached("hello") == "HELLO"
        assert cached("hello") == "HELLO"
        assert call_count["n"] == 1  # Only called once

    def test_cache_step_different_inputs(self):
        call_count = {"n": 0}

        def expensive(x: str) -> str:
            call_count["n"] += 1
            return x.upper()

        cached = cache_step(expensive, ttl=300)
        cached("a")
        cached("b")
        assert call_count["n"] == 2

    def test_cache_step_lru_eviction(self):
        call_count = {"n": 0}

        def fn(x: str) -> str:
            call_count["n"] += 1
            return x

        cached = cache_step(fn, ttl=300, max_size=2)
        cached("a")
        cached("b")
        cached("c")  # evicts "a"
        cached("a")  # cache miss — "a" was evicted
        assert call_count["n"] == 4


# ---------------------------------------------------------------------------
# Pipeline.__call__ as graph node
# ---------------------------------------------------------------------------


class TestPipelineCall:
    def test_call_with_plain_value(self):
        pipeline = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        result = pipeline("hello")
        assert result == "HELLO"

    def test_call_with_graph_state_no_last_output(self):
        from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState

        pipeline = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        state = GraphState.from_prompt("hello world")
        # No STATE_KEY_LAST_OUTPUT set — should fall back to messages
        result_state = pipeline(state)
        assert isinstance(result_state, GraphState)


# ---------------------------------------------------------------------------
# Pipeline with nested pipeline
# ---------------------------------------------------------------------------


class TestNestedPipeline:
    def test_nested_pipeline_in_run(self):
        inner = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        outer = Pipeline(steps=[inner, Step(lambda x: x + "!", name="exclaim")])
        result = outer.run("hello")
        assert result.output == "HELLO!"

    @pytest.mark.asyncio
    async def test_nested_pipeline_in_arun(self):
        inner = Pipeline(steps=[Step(lambda x: x.upper(), name="up")])
        outer = Pipeline(steps=[inner, Step(lambda x: x + "!", name="exclaim")])
        result = await outer.arun("hello")
        assert result.output == "HELLO!"

    def test_sync_pipeline_with_async_step_in_run(self):
        """Sync run() calls asyncio.run() for async steps."""

        async def async_upper(x: str) -> str:
            return x.upper()

        pipeline = Pipeline(steps=[Step(async_upper)])
        result = pipeline.run("hello")
        assert result.output == "HELLO"
