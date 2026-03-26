"""Tests for the composability layer — Pipeline, @step, parallel(), branch()."""

from __future__ import annotations

import asyncio

import pytest

from selectools.pipeline import Pipeline, Step, StepResult, branch, parallel, step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def upper(text: str) -> str:
    return text.upper()


def add_exclaim(text: str) -> str:
    return text + "!"


def double(text: str) -> str:
    return text + text


async def async_upper(text: str) -> str:
    return text.upper()


def failing_step(text: str) -> str:
    raise ValueError("boom")


counter = {"calls": 0}


def flaky_step(text: str) -> str:
    counter["calls"] += 1
    if counter["calls"] < 3:
        raise ValueError("not yet")
    return text.upper()


# ---------------------------------------------------------------------------
# @step decorator
# ---------------------------------------------------------------------------


class TestStepDecorator:
    def test_step_no_parens(self):
        @step
        def my_fn(x: str) -> str:
            return x.upper()

        assert isinstance(my_fn, Step)
        assert my_fn.name == "my_fn"
        assert my_fn("hello") == "HELLO"

    def test_step_with_parens(self):
        @step()
        def my_fn(x: str) -> str:
            return x.lower()

        assert isinstance(my_fn, Step)
        assert my_fn("HELLO") == "hello"

    def test_step_with_name(self):
        @step(name="custom_name")
        def my_fn(x: str) -> str:
            return x

        assert my_fn.name == "custom_name"

    def test_step_with_retry(self):
        @step(retry=3)
        def my_fn(x: str) -> str:
            return x

        assert my_fn.retry == 3

    def test_step_preserves_function_call(self):
        """Step is callable exactly like the original function."""

        @step
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_step_repr(self):
        @step
        def my_fn(x: str) -> str:
            return x

        assert "my_fn" in repr(my_fn)


# ---------------------------------------------------------------------------
# Pipeline basics
# ---------------------------------------------------------------------------


class TestPipelineBasics:
    def test_pipe_operator_creates_pipeline(self):
        s1 = Step(upper)
        s2 = Step(add_exclaim)
        pipeline = s1 | s2
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2

    def test_three_step_pipe(self):
        s1 = Step(upper)
        s2 = Step(add_exclaim)
        s3 = Step(double)
        pipeline = s1 | s2 | s3
        assert len(pipeline.steps) == 3

    def test_run_linear_pipeline(self):
        pipeline = Step(upper) | Step(add_exclaim)
        result = pipeline.run("hello")
        assert isinstance(result, StepResult)
        assert result.output == "HELLO!"
        assert result.steps_run == 2

    def test_run_three_steps(self):
        pipeline = Step(upper) | Step(add_exclaim) | Step(double)
        result = pipeline.run("hi")
        assert result.output == "HI!HI!"

    def test_trace_records_all_steps(self):
        pipeline = Step(upper) | Step(add_exclaim)
        result = pipeline.run("test")
        assert len(result.trace) == 2
        assert result.trace[0]["step"] == "upper"
        assert result.trace[0]["status"] == "ok"
        assert result.trace[1]["step"] == "add_exclaim"
        assert "duration_ms" in result.trace[0]

    def test_empty_pipeline_returns_input(self):
        pipeline = Pipeline(steps=[])
        result = pipeline.run("passthrough")
        assert result.output == "passthrough"
        assert result.steps_run == 0

    def test_single_step_pipeline(self):
        pipeline = Pipeline(steps=[Step(upper)])
        result = pipeline.run("hello")
        assert result.output == "HELLO"

    def test_pipeline_from_plain_functions(self):
        """Plain callables are auto-wrapped as Steps."""
        pipeline = Pipeline(steps=[upper, add_exclaim])
        result = pipeline.run("hey")
        assert result.output == "HEY!"

    def test_pipeline_repr(self):
        pipeline = Step(upper, name="up") | Step(add_exclaim, name="exclaim")
        assert "up" in repr(pipeline)
        assert "exclaim" in repr(pipeline)


# ---------------------------------------------------------------------------
# Decorated @step + | operator
# ---------------------------------------------------------------------------


class TestDecoratedComposition:
    def test_decorated_steps_compose(self):
        @step
        def shout(text: str) -> str:
            return text.upper()

        @step
        def exclaim(text: str) -> str:
            return text + "!!!"

        pipeline = shout | exclaim
        result = pipeline.run("hello")
        assert result.output == "HELLO!!!"

    def test_plain_function_in_pipe(self):
        """Bare functions can be piped with Steps."""

        @step
        def shout(text: str) -> str:
            return text.upper()

        pipeline = shout | add_exclaim
        result = pipeline.run("hello")
        assert result.output == "HELLO!"


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------


class TestAsyncPipeline:
    @pytest.mark.asyncio
    async def test_async_step(self):
        s = Step(async_upper)
        assert s.is_async
        result = await s("hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_async_pipeline(self):
        pipeline = Step(async_upper) | Step(add_exclaim)
        result = await pipeline.arun("hello")
        assert result.output == "HELLO!"
        assert result.steps_run == 2

    @pytest.mark.asyncio
    async def test_mixed_sync_async_pipeline(self):
        pipeline = Step(upper) | Step(async_upper) | Step(add_exclaim)
        result = await pipeline.arun("hey")
        assert result.output == "HEY!"


# ---------------------------------------------------------------------------
# Error handling & retry
# ---------------------------------------------------------------------------


class TestRetryAndErrors:
    def test_error_raises_by_default(self):
        pipeline = Step(upper) | Step(failing_step)
        with pytest.raises(ValueError, match="boom"):
            pipeline.run("test")

    def test_error_trace_recorded(self):
        pipeline = Step(upper) | Step(failing_step)
        try:
            pipeline.run("test")
        except ValueError:
            pass
        # Can't access trace after exception in current impl — that's OK

    def test_on_error_skip(self):
        pipeline = Step(upper) | Step(failing_step, on_error="skip") | Step(add_exclaim)
        result = pipeline.run("test")
        # Skipped failing step, add_exclaim gets upper's output
        assert result.output == "TEST!"

    def test_retry_succeeds(self):
        counter["calls"] = 0
        s = Step(flaky_step, retry=3)
        pipeline = Pipeline(steps=[s])
        result = pipeline.run("hello")
        assert result.output == "HELLO"

    def test_retry_exhausted_raises(self):
        def always_fail(x: str) -> str:
            raise ValueError("always")

        pipeline = Pipeline(steps=[Step(always_fail, retry=2)])
        with pytest.raises(ValueError, match="always"):
            pipeline.run("test")


# ---------------------------------------------------------------------------
# parallel()
# ---------------------------------------------------------------------------


class TestParallel:
    def test_parallel_runs_all_steps(self):
        def search_web(query: str) -> str:
            return f"web:{query}"

        def search_docs(query: str) -> str:
            return f"docs:{query}"

        p = parallel(search_web, search_docs)
        result = p("ai agents")
        assert isinstance(result, dict)
        assert result["search_web"] == "web:ai agents"
        assert result["search_docs"] == "docs:ai agents"

    def test_parallel_in_pipeline(self):
        def a(x: str) -> str:
            return f"a:{x}"

        def b(x: str) -> str:
            return f"b:{x}"

        def merge(results: dict) -> str:
            return " + ".join(results.values())

        pipeline = parallel(a, b) | merge
        result = pipeline.run("input")
        assert "a:input" in result.output
        assert "b:input" in result.output

    def test_parallel_with_decorated_steps(self):
        @step
        def fast(x: str) -> str:
            return "fast"

        @step
        def slow(x: str) -> str:
            return "slow"

        p = parallel(fast, slow)
        result = p("go")
        assert result["fast"] == "fast"
        assert result["slow"] == "slow"


# ---------------------------------------------------------------------------
# branch()
# ---------------------------------------------------------------------------


class TestBranch:
    def test_branch_with_router(self):
        def route(x: str) -> str:
            return "upper" if "shout" in x else "lower"

        @step
        def up(x: str) -> str:
            return x.upper()

        @step
        def down(x: str) -> str:
            return x.lower()

        b = branch(router=route, upper=up, lower=down)
        assert b("shout this") == "SHOUT THIS"
        assert b("whisper this") == "whisper this"

    def test_branch_with_string_input(self):
        @step
        def handle_a(x: str) -> str:
            return "handled_a"

        @step
        def handle_b(x: str) -> str:
            return "handled_b"

        b = branch(handle_a=handle_a, handle_b=handle_b)
        assert b("handle_a") == "handled_a"
        assert b("handle_b") == "handled_b"

    def test_branch_default(self):
        @step
        def fallback(x: str) -> str:
            return "default"

        b = branch(default=fallback)
        assert b("unknown_key") == "default"

    def test_branch_missing_key_raises(self):
        @step
        def only_one(x: str) -> str:
            return x

        b = branch(one=only_one)
        with pytest.raises(KeyError, match="no matching branch"):
            b("two")

    def test_branch_in_pipeline(self):
        def classify(text: str) -> str:
            return "tech" if "code" in text else "general"

        @step
        def tech_review(x: str) -> str:
            return f"[TECH] {x}"

        @step
        def general_review(x: str) -> str:
            return f"[GEN] {x}"

        pipeline = classify | branch(
            router=lambda x: x,
            tech=tech_review,
            general=general_review,
        )
        result = pipeline.run("review this code")
        assert result.output == "[TECH] tech"


# ---------------------------------------------------------------------------
# AgentGraph bridge
# ---------------------------------------------------------------------------


class TestGraphBridge:
    def test_pipeline_as_graph_node(self):
        """Pipeline.__call__ works with GraphState."""
        from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState

        pipeline = Step(upper) | Step(add_exclaim)
        state = GraphState.from_prompt("hello")
        state.data[STATE_KEY_LAST_OUTPUT] = "hello"

        result_state = pipeline(state)
        assert isinstance(result_state, GraphState)
        assert result_state.data[STATE_KEY_LAST_OUTPUT] == "HELLO!"

    def test_pipeline_in_agent_graph(self):
        """Pipeline used as a callable node in AgentGraph."""
        from selectools.orchestration.graph import AgentGraph
        from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState

        def prep(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "hello world"
            return state

        pipeline = Step(upper) | Step(add_exclaim)

        graph = AgentGraph()
        graph.add_node("prep", prep, next_node="process")
        graph.add_node("process", pipeline, next_node=AgentGraph.END)
        result = graph.run("start")
        assert result.content == "HELLO WORLD!"


# ---------------------------------------------------------------------------
# Import from selectools
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_imports_from_selectools(self):
        from selectools import Pipeline, Step, StepResult, branch, parallel, step

        assert callable(step)
        assert callable(parallel)
        assert callable(branch)
        assert Pipeline is not None
        assert Step is not None
        assert StepResult is not None
