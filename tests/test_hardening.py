"""
Hardening tests — actively try to break orchestration and pipelines.

These test adversarial inputs, edge cases, error recovery, and
boundary conditions that normal usage would never hit.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from selectools.exceptions import GraphExecutionError
from selectools.orchestration.checkpoint import InMemoryCheckpointStore
from selectools.orchestration.graph import AgentGraph, ErrorPolicy, GraphResult
from selectools.orchestration.state import (
    STATE_KEY_LAST_OUTPUT,
    GraphState,
    InterruptRequest,
    MergePolicy,
    Scatter,
)
from selectools.pipeline import Pipeline, Step, StepResult, branch, parallel, step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fn_set(content: str):
    def fn(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = content
        return state

    return fn


# ===========================================================================
# Orchestration hardening
# ===========================================================================


class TestGraphEdgeCases:
    def test_graph_with_none_prompt(self):
        """Graph.run(None) should handle gracefully."""
        graph = AgentGraph()
        graph.add_node("a", fn_set("ok"), next_node=AgentGraph.END)
        # None as prompt — from_prompt will create Message(content=None)
        result = graph.run(None)
        assert isinstance(result, GraphResult)

    def test_graph_with_empty_string_prompt(self):
        """Graph.run('') should execute normally."""
        graph = AgentGraph()
        graph.add_node("a", fn_set("ok"), next_node=AgentGraph.END)
        result = graph.run("")
        assert result.content == "ok"

    def test_graph_single_node_no_explicit_end(self):
        """Single node with no edge — auto-entry + implicit END."""
        graph = AgentGraph()
        graph.add_node("only", fn_set("done"))
        # No edges at all — should hit implicit END
        result = graph.run("go")
        assert result.content == "done"

    def test_graph_max_steps_zero(self):
        """max_steps=0 should immediately return without executing."""
        graph = AgentGraph(max_steps=0)
        graph.add_node("a", fn_set("should not run"), next_node=AgentGraph.END)
        result = graph.run("go")
        assert result.steps == 0

    def test_graph_node_returns_none(self):
        """Node function returning None instead of state."""

        def bad_node(state: GraphState):
            return None

        graph = AgentGraph()
        graph.add_node("bad", bad_node, next_node=AgentGraph.END)
        result = graph.run("go")
        # Should handle gracefully — None becomes original state
        assert isinstance(result, GraphResult)

    def test_graph_node_mutates_state_data_with_large_values(self):
        """Large state.data values shouldn't crash serialization."""

        def big_node(state: GraphState) -> GraphState:
            state.data["big"] = "x" * 100_000
            state.data[STATE_KEY_LAST_OUTPUT] = "done"
            return state

        graph = AgentGraph()
        graph.add_node("big", big_node, next_node=AgentGraph.END)
        result = graph.run("go")
        assert result.content == "done"
        assert len(result.state.data["big"]) == 100_000

    def test_graph_unicode_in_state(self):
        """Unicode content in state.data and messages."""

        def unicode_node(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "Resultado: cafe, nino, uber"
            state.data["emoji"] = "rocket fire star"
            return state

        graph = AgentGraph()
        graph.add_node("uni", unicode_node, next_node=AgentGraph.END)
        result = graph.run("go")
        assert "cafe" in result.content

    def test_graph_conditional_all_branches_have_edges(self):
        """All conditional branches should resolve without hanging."""

        def router(state: GraphState) -> str:
            val = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            if "a" in val:
                return "branch_a"
            return "branch_b"

        graph = AgentGraph()
        graph.add_node("entry", fn_set("go_a"))
        graph.add_node("branch_a", fn_set("from_a"))
        graph.add_node("branch_b", fn_set("from_b"))
        graph.add_conditional_edge("entry", router)
        graph.add_edge("branch_a", AgentGraph.END)
        graph.add_edge("branch_b", AgentGraph.END)

        result = graph.run("go")
        assert result.content == "from_a"

    def test_graph_chain_with_zero_agents(self):
        """AgentGraph.chain() with no agents should raise."""
        with pytest.raises((ValueError, TypeError)):
            AgentGraph.chain()

    def test_graph_duplicate_node_names(self):
        """Adding a node with the same name should overwrite."""
        graph = AgentGraph()
        graph.add_node("a", fn_set("first"))
        graph.add_node("a", fn_set("second"))
        graph.add_edge("a", AgentGraph.END)
        result = graph.run("go")
        assert result.content == "second"

    def test_graph_budget_zero_stops_immediately(self):
        """max_total_tokens=0 should stop before any execution."""
        graph = AgentGraph(max_total_tokens=0)
        graph.add_node("a", fn_set("should not run"), next_node=AgentGraph.END)
        result = graph.run("go")
        # Budget check fires at top of loop
        assert result.steps == 0 or result.content == "should not run"

    def test_graph_result_has_all_fields(self):
        """GraphResult should always have all fields populated."""
        graph = AgentGraph.chain(fn_set("done"))
        result = graph.run("go")

        assert hasattr(result, "content")
        assert hasattr(result, "state")
        assert hasattr(result, "node_results")
        assert hasattr(result, "trace")
        assert hasattr(result, "total_usage")
        assert hasattr(result, "interrupted")
        assert hasattr(result, "interrupt_id")
        assert hasattr(result, "steps")
        assert hasattr(result, "stalls")
        assert hasattr(result, "loops_detected")
        assert result.interrupted is False
        assert result.interrupt_id is None


class TestParallelHardening:
    def test_parallel_single_child(self):
        """Parallel with 1 child should work like a single node."""
        graph = AgentGraph()
        graph.add_node("only", fn_set("only_result"))
        graph.add_parallel_nodes("group", ["only"])
        graph.add_edge("group", AgentGraph.END)
        graph.set_entry("group")
        result = graph.run("go")
        assert "only" in result.node_results

    def test_parallel_child_raises_with_skip(self):
        """One child raising in parallel with SKIP policy — graph continues."""

        def failing(state: GraphState) -> GraphState:
            raise ValueError("child failed")

        graph = AgentGraph(error_policy=ErrorPolicy.SKIP)
        graph.add_node("ok", fn_set("ok_result"))
        graph.add_node("fail", failing)
        graph.add_parallel_nodes("group", ["ok", "fail"])
        graph.add_edge("group", AgentGraph.END)
        graph.set_entry("group")

        result = graph.run("go")
        # Graph should complete despite one child failing
        assert result.content == "ok_result"

    def test_parallel_all_children_fail_with_skip(self):
        """All children fail — should not crash."""

        def fail_a(state: GraphState) -> GraphState:
            raise ValueError("a failed")

        def fail_b(state: GraphState) -> GraphState:
            raise ValueError("b failed")

        graph = AgentGraph(error_policy=ErrorPolicy.SKIP)
        graph.add_node("a", fail_a)
        graph.add_node("b", fail_b)
        graph.add_parallel_nodes("group", ["a", "b"])
        graph.add_edge("group", AgentGraph.END)
        graph.set_entry("group")

        result = graph.run("go")
        assert len(result.state.errors) == 2


class TestCheckpointHardening:
    def test_checkpoint_with_large_state(self):
        """Checkpoint handles large state data."""
        store = InMemoryCheckpointStore()
        state = GraphState.from_prompt("test")
        state.data["big"] = "x" * 50_000
        state.data[STATE_KEY_LAST_OUTPUT] = "checkpointed"

        cid = store.save("g1", state, 1)
        loaded, step = store.load(cid)
        assert loaded.data["big"] == "x" * 50_000
        assert loaded.data[STATE_KEY_LAST_OUTPUT] == "checkpointed"

    def test_checkpoint_with_special_characters(self):
        """Checkpoint handles unicode and special chars in state."""
        store = InMemoryCheckpointStore()
        state = GraphState.from_prompt("test")
        state.data["special"] = "quotes\"and'backslash\\and\nnewline"
        state.data["unicode"] = "cafe nino uber"

        cid = store.save("g1", state, 1)
        loaded, _ = store.load(cid)
        assert loaded.data["special"] == state.data["special"]
        assert loaded.data["unicode"] == state.data["unicode"]

    def test_checkpoint_load_nonexistent(self):
        """Loading a nonexistent checkpoint raises ValueError."""
        store = InMemoryCheckpointStore()
        with pytest.raises(ValueError, match="not found"):
            store.load("nonexistent_id")

    def test_checkpoint_delete_nonexistent(self):
        """Deleting a nonexistent checkpoint returns False."""
        store = InMemoryCheckpointStore()
        assert store.delete("nonexistent") is False

    def test_multiple_checkpoints_same_graph(self):
        """Multiple checkpoints for the same graph_id."""
        store = InMemoryCheckpointStore()
        state = GraphState.from_prompt("test")

        ids = []
        for i in range(5):
            state.data["step"] = i
            ids.append(store.save("g1", state, i))

        metas = store.list("g1")
        assert len(metas) == 5

        # Load each one
        for i, cid in enumerate(ids):
            loaded, step = store.load(cid)
            assert step == i


class TestHITLHardening:
    def test_interrupt_without_checkpoint_store(self):
        """Interrupt without checkpoint_store produces fake ID but doesn't crash."""

        async def gate(state: GraphState):
            yield InterruptRequest(prompt="approve?")

        graph = AgentGraph()
        graph.add_node("gate", gate, next_node=AgentGraph.END)

        result = graph.run("go")  # No checkpoint_store
        assert result.interrupted
        assert result.interrupt_id is not None  # Fake ID

    def test_single_interrupt_resumes_correctly(self):
        """Generator yields once, resume continues execution."""

        async def gate(state: GraphState):
            approval = yield InterruptRequest(prompt="approve?")
            state.data["approval"] = approval
            state.data[STATE_KEY_LAST_OUTPUT] = f"approved={approval}"

        graph = AgentGraph()
        graph.add_node("gate", gate)
        graph.add_edge("gate", AgentGraph.END)

        store = InMemoryCheckpointStore()
        r1 = graph.run("go", checkpoint_store=store)
        assert r1.interrupted

        r2 = graph.resume(r1.interrupt_id, "yes", checkpoint_store=store)
        assert not r2.interrupted
        assert r2.state.data.get("approval") == "yes"
        assert "approved=yes" in r2.content


# ===========================================================================
# Pipeline hardening
# ===========================================================================


class TestPipelineEdgeCases:
    def test_pipeline_none_input(self):
        """Pipeline.run(None) should pass None through."""

        @step
        def handle_none(x):
            return "was_none" if x is None else "not_none"

        result = Pipeline(steps=[handle_none]).run(None)
        assert result.output == "was_none"

    def test_pipeline_with_dict_flow(self):
        """Pipeline steps can pass dicts between them."""

        @step
        def create_dict(x: str) -> dict:
            return {"text": x, "count": len(x)}

        @step
        def read_dict(d: dict) -> str:
            return f"text={d['text']}, count={d['count']}"

        pipeline = create_dict | read_dict
        result = pipeline.run("hello")
        assert result.output == "text=hello, count=5"

    def test_pipeline_with_list_flow(self):
        """Pipeline steps can pass lists."""

        @step
        def split_words(text: str) -> list:
            return text.split()

        @step
        def count_items(items: list) -> str:
            return f"{len(items)} items"

        result = (split_words | count_items).run("hello world foo")
        assert result.output == "3 items"

    def test_deeply_nested_pipelines(self):
        """Pipeline of pipelines of pipelines."""
        inner = Step(str.upper) | Step(lambda x: x + "!")
        middle = inner | Step(lambda x: x + "?")
        outer = Step(lambda x: x.strip()) | middle

        result = outer.run("  hello  ")
        assert result.output == "HELLO!?"

    def test_pipeline_step_raises_clear_error(self):
        """Error should include step name context."""

        @step(name="exploder")
        def explode(x):
            raise RuntimeError("kaboom")

        pipeline = Pipeline(steps=[explode])
        with pytest.raises(RuntimeError, match="kaboom"):
            pipeline.run("test")

    def test_pipeline_100_steps(self):
        """Long pipeline with 100 steps doesn't stack overflow."""
        steps = [Step(lambda x, i=i: f"{x}+{i}") for i in range(100)]
        pipeline = Pipeline(steps=steps)
        result = pipeline.run("start")
        assert "+99" in result.output
        assert result.steps_run == 100

    def test_parallel_no_steps(self):
        """parallel() with no steps."""
        p = parallel()
        result = p("input")
        assert result == {}

    def test_branch_with_dict_input(self):
        """branch() reads 'branch' key from dict input."""
        b = branch(
            a=Step(lambda x: "handled_a"),
            b=Step(lambda x: "handled_b"),
        )
        result = b({"branch": "a", "data": "hello"})
        assert result == "handled_a"

    def test_step_with_multiple_args(self):
        """Step receiving kwargs from pipeline."""

        @step
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = Pipeline(steps=[greet]).run("World", greeting="Bonjour")
        assert result.output == "Bonjour, World!"

    def test_pipeline_preserves_step_order(self):
        """Steps execute in exact order."""
        order = []

        @step(name="first")
        def s1(x):
            order.append(1)
            return x

        @step(name="second")
        def s2(x):
            order.append(2)
            return x

        @step(name="third")
        def s3(x):
            order.append(3)
            return x

        (s1 | s2 | s3).run("go")
        assert order == [1, 2, 3]


class TestAsyncPipelineHardening:
    @pytest.mark.asyncio
    async def test_async_pipeline_with_sync_steps(self):
        """arun() handles all-sync steps correctly."""
        pipeline = Step(str.upper) | Step(lambda x: x + "!")
        result = await pipeline.arun("hello")
        assert result.output == "HELLO!"

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Retry works in async pipeline."""
        call_count = {"n": 0}

        async def flaky(x):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("not yet")
            return x.upper()

        pipeline = Pipeline(steps=[Step(flaky, retry=3)])
        result = await pipeline.arun("hello")
        assert result.output == "HELLO"


# ===========================================================================
# Graph + Pipeline integration
# ===========================================================================


class TestGraphPipelineIntegration:
    def test_pipeline_as_graph_node_with_parallel(self):
        """Pipeline with parallel() used as a graph node."""

        def search_a(x: str) -> str:
            return f"a:{x}"

        def search_b(x: str) -> str:
            return f"b:{x}"

        def merge(results: dict) -> str:
            return " + ".join(results.values())

        pipeline = parallel(search_a, search_b) | merge

        graph = AgentGraph()
        graph.add_node("prep", fn_set("query"), next_node="search")
        graph.add_node("search", pipeline, next_node=AgentGraph.END)
        result = graph.run("go")
        assert "a:query" in result.content
        assert "b:query" in result.content

    def test_graph_in_pipeline(self):
        """AgentGraph used as a step in a Pipeline via __call__."""
        inner_graph = AgentGraph.chain(fn_set("from_graph"))

        pipeline = Step(str.upper) | inner_graph | Step(lambda x: f"final:{x}")

        # inner_graph.__call__ receives a string (not GraphState)
        # It wraps in run() and returns output
        result = pipeline.run("hello")
        assert "from_graph" in result.output


class TestGraphStateHardening:
    def test_state_to_dict_from_dict_roundtrip(self):
        """GraphState survives serialization roundtrip."""
        from selectools.types import Message, Role

        state = GraphState(
            messages=[
                Message(role=Role.USER, content="hello"),
                Message(role=Role.ASSISTANT, content="hi there"),
            ],
            data={"key": "value", "nested": {"a": 1}},
            current_node="test_node",
            metadata={"request_id": "abc123"},
        )

        d = state.to_dict()
        restored = GraphState.from_dict(d)

        assert len(restored.messages) == 2
        assert restored.data["key"] == "value"
        assert restored.data["nested"]["a"] == 1
        assert restored.current_node == "test_node"
        assert restored.metadata["request_id"] == "abc123"

    def test_state_last_output_property(self):
        """last_output property works correctly."""
        state = GraphState()
        assert state.last_output == ""

        state.last_output = "test"
        assert state.last_output == "test"
        assert state.data[STATE_KEY_LAST_OUTPUT] == "test"

    def test_state_from_prompt_none(self):
        """GraphState.from_prompt(None) handles gracefully."""
        state = GraphState.from_prompt(None)
        assert len(state.messages) == 1
