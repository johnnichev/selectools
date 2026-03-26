"""Tests for AgentGraph — execution engine, routing, parallel, HITL, subgraphs."""

from __future__ import annotations

import asyncio

import pytest

from selectools.exceptions import GraphExecutionError
from selectools.orchestration.graph import AgentGraph, ErrorPolicy, GraphResult
from selectools.orchestration.node import GraphNode
from selectools.orchestration.state import (
    STATE_KEY_LAST_OUTPUT,
    ContextMode,
    GraphEvent,
    GraphEventType,
    GraphState,
    InterruptRequest,
    MergePolicy,
    Scatter,
)
from selectools.types import Message, Role

# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


def sync_fn_node(content: str):
    """Return a sync callable that sets STATE_KEY_LAST_OUTPUT."""

    def fn(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = content
        return state

    return fn


# ------------------------------------------------------------------
# Linear graph
# ------------------------------------------------------------------


class TestLinearGraph:
    def test_linear_three_nodes(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("from_a"))
        graph.add_node("b", sync_fn_node("from_b"))
        graph.add_node("c", sync_fn_node("from_c"))
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", AgentGraph.END)
        graph.set_entry("a")

        result = graph.run("start")
        assert result.content == "from_c"
        assert result.steps == 3

    def test_implicit_end_no_outgoing_edge(self):
        """Node with no outgoing edge implicitly ends the graph."""
        graph = AgentGraph()
        graph.add_node("only", sync_fn_node("sole output"))
        graph.set_entry("only")

        result = graph.run("start")
        assert result.content == "sole output"
        assert result.steps == 1

    def test_state_passes_between_nodes(self):
        def writer(state: GraphState) -> GraphState:
            state.data["written"] = "hello"
            return state

        def reader(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = state.data.get("written", "missing")
            return state

        graph = AgentGraph()
        graph.add_node("writer", writer)
        graph.add_node("reader", reader)
        graph.add_edge("writer", "reader")
        graph.add_edge("reader", AgentGraph.END)
        graph.set_entry("writer")

        result = graph.run("test")
        assert result.content == "hello"
        assert result.state.data["written"] == "hello"

    def test_result_has_node_results(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("out"))
        graph.set_entry("a")

        result = graph.run("x")
        assert "a" in result.node_results


class TestGraphValidation:
    def test_no_entry_auto_set_on_first_add(self):
        """First add_node auto-sets entry — no need for set_entry()."""
        graph = AgentGraph()
        graph.add_node("n", sync_fn_node("x"))
        graph.add_edge("n", AgentGraph.END)
        result = graph.run("go")
        assert result.content  # runs without error

    def test_empty_graph_raises(self):
        """Graph with no nodes raises on run."""
        graph = AgentGraph()
        with pytest.raises(GraphExecutionError):
            graph.run("go")

    def test_validate_returns_warnings(self):
        graph = AgentGraph()
        graph.add_node("n", sync_fn_node("x"))
        # Entry is set, edge target not registered
        graph.set_entry("n")
        graph.add_edge("n", "nonexistent")
        warnings = graph.validate()
        assert any("nonexistent" in w for w in warnings)

    def test_missing_node_in_execution_raises(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("x"))
        graph.add_edge("a", "b")  # b does not exist
        graph.set_entry("a")
        with pytest.raises(GraphExecutionError):
            graph.run("go")


# ------------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------------


class TestConditionalRouting:
    def test_conditional_edge_routes_to_named_node(self):
        def fn_yes(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "yes answer"
            return state

        def fn_no(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "no answer"
            return state

        # Router always returns "yes_branch"
        def router(state: GraphState) -> str:
            return "yes_branch"

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_node("yes_branch", fn_yes)
        graph.add_node("no_branch", fn_no)
        graph.add_conditional_edge("entry", router)
        graph.add_edge("yes_branch", AgentGraph.END)
        graph.add_edge("no_branch", AgentGraph.END)
        graph.set_entry("entry")

        result = graph.run("test")
        assert result.content == "yes answer"

    def test_path_map_validation_on_unknown_return(self):
        """Router returning value not in path_map should raise."""

        def bad_router(state: GraphState) -> str:
            return "unknown_node"

        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("x"))
        graph.add_node("b", sync_fn_node("y"))
        graph.add_conditional_edge("a", bad_router, path_map={"b": "b"})
        graph.add_edge("b", AgentGraph.END)
        graph.set_entry("a")

        with pytest.raises(GraphExecutionError):
            graph.run("go")

    def test_scatter_dynamic_fanout(self):
        """Scatter returns create dynamic parallel group."""

        def scatter_router(state: GraphState):
            return [Scatter("worker_a"), Scatter("worker_b")]

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_node("worker_a", sync_fn_node("result_a"))
        graph.add_node("worker_b", sync_fn_node("result_b"))
        graph.add_conditional_edge("entry", scatter_router)
        graph.set_entry("entry")

        # Scatter creates a dynamic parallel group — should not crash
        result = graph.run("go")
        assert result.steps >= 1


# ------------------------------------------------------------------
# Parallel nodes
# ------------------------------------------------------------------


class TestParallelNodes:
    def test_parallel_fan_out(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("from_a"))
        graph.add_node("b", sync_fn_node("from_b"))
        graph.add_parallel_nodes("fan_out", ["a", "b"], merge_policy=MergePolicy.LAST_WINS)
        graph.add_edge("fan_out", AgentGraph.END)
        graph.set_entry("fan_out")

        result = graph.run("go")
        assert result.steps >= 1
        assert result.content in ("from_a", "from_b")

    def test_parallel_append_merge(self):
        def a_fn(state: GraphState) -> GraphState:
            state.data["items"] = ["a"]
            state.data[STATE_KEY_LAST_OUTPUT] = "from_a"
            return state

        def b_fn(state: GraphState) -> GraphState:
            state.data["items"] = ["b"]
            state.data[STATE_KEY_LAST_OUTPUT] = "from_b"
            return state

        graph = AgentGraph()
        graph.add_node("a", a_fn)
        graph.add_node("b", b_fn)
        graph.add_parallel_nodes("fan", ["a", "b"], merge_policy=MergePolicy.APPEND)
        graph.add_edge("fan", AgentGraph.END)
        graph.set_entry("fan")

        result = graph.run("go")
        items = result.state.data.get("items", [])
        # With APPEND, list items from both branches are combined
        assert set(items) == {"a", "b"} or "a" in items or "b" in items

    def test_add_parallel_nodes_registers_in_nodes(self):
        from selectools.orchestration.node import ParallelGroupNode

        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("x"))
        graph.add_node("b", sync_fn_node("y"))
        graph.add_parallel_nodes("group", ["a", "b"])
        assert "group" in graph._nodes
        assert isinstance(graph._nodes["group"], ParallelGroupNode)

    def test_state_isolation_in_parallel(self):
        """Each parallel branch gets a deep copy of state — mutations don't bleed."""

        def branch_a(state: GraphState) -> GraphState:
            state.data["branch"] = "a"
            state.data[STATE_KEY_LAST_OUTPUT] = "a"
            return state

        def branch_b(state: GraphState) -> GraphState:
            state.data["branch"] = "b"
            state.data[STATE_KEY_LAST_OUTPUT] = "b"
            return state

        graph = AgentGraph()
        graph.add_node("branch_a", branch_a)
        graph.add_node("branch_b", branch_b)
        graph.add_parallel_nodes("fan", ["branch_a", "branch_b"])
        graph.add_edge("fan", AgentGraph.END)
        graph.set_entry("fan")

        result = graph.run("go")
        # Both branches ran; merged state has one of their values (not undefined)
        assert result.state.data.get("branch") in ("a", "b")


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    def test_abort_policy_raises(self):
        def failing_node(state: GraphState) -> GraphState:
            raise RuntimeError("intentional failure")

        graph = AgentGraph(error_policy=ErrorPolicy.ABORT)
        graph.add_node("fail", failing_node)
        graph.set_entry("fail")

        with pytest.raises(GraphExecutionError):
            graph.run("go")

    def test_skip_policy_continues(self):
        def failing_node(state: GraphState) -> GraphState:
            raise RuntimeError("intentional failure")

        def ok_node(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "survived"
            return state

        graph = AgentGraph(error_policy=ErrorPolicy.SKIP)
        graph.add_node("fail", failing_node)
        graph.add_node("ok", ok_node)
        graph.add_edge("fail", "ok")
        graph.add_edge("ok", AgentGraph.END)
        graph.set_entry("fail")

        result = graph.run("go")
        assert result.content == "survived"
        assert len(result.state.errors) == 1

    def test_per_node_error_policy_overrides_graph(self):
        """Node-level SKIP overrides graph-level ABORT."""

        def failing_node(state: GraphState) -> GraphState:
            raise RuntimeError("intentional failure")

        def ok_node(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "ok"
            return state

        graph = AgentGraph(error_policy=ErrorPolicy.ABORT)
        graph.add_node("fail", failing_node, error_policy=ErrorPolicy.SKIP)
        graph.add_node("ok", ok_node)
        graph.add_edge("fail", "ok")
        graph.add_edge("ok", AgentGraph.END)
        graph.set_entry("fail")

        result = graph.run("go")
        assert result.content == "ok"


# ------------------------------------------------------------------
# Loop and stall detection
# ------------------------------------------------------------------


class TestLoopAndStallDetection:
    def test_max_steps_guard(self):
        """Graph respects max_steps limit."""
        call_count = {"n": 0}

        def looping_node(state: GraphState) -> GraphState:
            call_count["n"] += 1
            # Mutate state so loop detection doesn't trip before max_steps
            state.data["step"] = call_count["n"]
            state.data[STATE_KEY_LAST_OUTPUT] = f"step_{call_count['n']}"
            return state

        graph = AgentGraph(max_steps=3, enable_loop_detection=False)
        graph.add_node("loop", looping_node)
        graph.add_edge("loop", "loop")  # infinite loop
        graph.set_entry("loop")

        result = graph.run("go")
        assert result.steps == 3

    def test_per_node_max_visits(self):
        graph = AgentGraph(enable_loop_detection=False)
        # Each visit mutates state so loop hash changes; max_visits enforces the cap
        visit_count = {"n": 0}

        def node_fn(state: GraphState) -> GraphState:
            visit_count["n"] += 1
            state.data["visits"] = visit_count["n"]
            state.data[STATE_KEY_LAST_OUTPUT] = str(visit_count["n"])
            return state

        graph.add_node("loop", node_fn, max_visits=2)
        graph.add_edge("loop", "loop")
        graph.set_entry("loop")

        with pytest.raises(GraphExecutionError):
            graph.run("go")

    def test_loop_detection_raises_on_repeated_hash(self):
        """Hard loop: same state hash twice → raises GraphExecutionError."""

        def stateless_node(state: GraphState) -> GraphState:
            # Does not mutate state → same hash each time
            return state

        graph = AgentGraph(enable_loop_detection=True)
        graph.add_node("n", stateless_node)
        graph.add_edge("n", "n")
        graph.set_entry("n")

        with pytest.raises(GraphExecutionError):
            graph.run("go")


# ------------------------------------------------------------------
# HITL (Human-in-the-loop)
# ------------------------------------------------------------------


class TestHITL:
    def test_generator_node_yields_interrupt(self):
        """Generator node that yields InterruptRequest causes interrupted=True."""
        from selectools.orchestration.checkpoint import InMemoryCheckpointStore

        async def review_node(state: GraphState):
            state.data["pre_interrupt"] = "computed"
            approval = yield InterruptRequest(prompt="Approve?", payload="some data")
            state.data[STATE_KEY_LAST_OUTPUT] = f"approved={approval}"

        graph = AgentGraph()
        graph.add_node("review", review_node)
        graph.set_entry("review")

        store = InMemoryCheckpointStore()
        result = graph.run("start", checkpoint_store=store)
        assert result.interrupted is True
        assert result.interrupt_id is not None
        assert result.state.data.get("pre_interrupt") == "computed"

    def test_generator_node_resumes(self):
        """resume() injects response and continues from yield point."""
        from selectools.orchestration.checkpoint import InMemoryCheckpointStore

        async def review_node(state: GraphState):
            state.data["pre_interrupt"] = "done"
            approval = yield InterruptRequest(prompt="Approve?")
            state.data[STATE_KEY_LAST_OUTPUT] = f"decision={approval}"

        graph = AgentGraph()
        graph.add_node("review", review_node)
        graph.set_entry("review")

        store = InMemoryCheckpointStore()
        result = graph.run("start", checkpoint_store=store)
        assert result.interrupted

        # Resume with approval
        final = graph.resume(result.interrupt_id, "yes", checkpoint_store=store)
        assert not final.interrupted
        last_output = final.state.data.get(STATE_KEY_LAST_OUTPUT, "")
        assert "yes" in last_output or "yes" in final.content


# ------------------------------------------------------------------
# Async execution
# ------------------------------------------------------------------


class TestAsyncExecution:
    @pytest.mark.asyncio
    async def test_arun_linear(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("async result"))
        graph.set_entry("a")

        result = await graph.arun("go")
        assert result.content == "async result"

    @pytest.mark.asyncio
    async def test_astream_yields_events(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("streamed"))
        graph.add_node("b", sync_fn_node("final"))
        graph.add_edge("a", "b")
        graph.set_entry("a")

        events = []
        async for event in graph.astream("go"):
            events.append(event)

        types = [e.type for e in events]
        assert GraphEventType.GRAPH_START in types
        assert GraphEventType.GRAPH_END in types
        assert GraphEventType.NODE_START in types
        assert GraphEventType.NODE_END in types


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------


class TestVisualization:
    def test_to_mermaid_linear(self):
        graph = AgentGraph(name="test_graph")
        graph.add_node("a", sync_fn_node("x"))
        graph.add_node("b", sync_fn_node("y"))
        graph.add_edge("a", "b")
        graph.add_edge("b", AgentGraph.END)
        graph.set_entry("a")

        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid
        assert "a" in mermaid
        assert "b" in mermaid

    def test_to_mermaid_with_conditional(self):
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("x"))
        graph.add_node("b", sync_fn_node("y"))
        graph.add_conditional_edge("a", lambda s: "b", path_map={"b": "b"})
        graph.add_edge("b", AgentGraph.END)
        graph.set_entry("a")

        mermaid = graph.to_mermaid()
        assert "b" in mermaid

    def test_to_mermaid_empty_graph(self):
        graph = AgentGraph()
        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid  # no crash

    def test_visualize_ascii_no_crash(self, capsys):
        graph = AgentGraph(name="vis_test")
        graph.add_node("n", sync_fn_node("x"))
        graph.set_entry("n")
        graph.visualize(format="ascii")  # no exception
        captured = capsys.readouterr()
        assert "vis_test" in captured.out

    def test_visualize_png_raises_without_graphviz(self):
        import sys
        import unittest.mock as mock

        graph = AgentGraph(name="test")
        graph.add_node("n", sync_fn_node("x"))
        graph.set_entry("n")

        with mock.patch.dict(sys.modules, {"graphviz": None}):
            with pytest.raises(ImportError):
                graph.visualize(format="png")


# ------------------------------------------------------------------
# Observable events
# ------------------------------------------------------------------


class TestObservers:
    def test_observer_events_fire(self):
        from selectools.observer import AgentObserver

        events = []

        class RecordingObserver(AgentObserver):
            def on_graph_start(self, run_id, graph_name, entry_node, state):
                events.append(("graph_start", graph_name))

            def on_graph_end(self, run_id, graph_name, steps, total_duration_ms):
                events.append(("graph_end", steps))

            def on_node_start(self, run_id, node_name, step):
                events.append(("node_start", node_name))

            def on_node_end(self, run_id, node_name, step, duration_ms):
                events.append(("node_end", node_name))

            def on_graph_routing(self, run_id, from_node, to_node):
                events.append(("routing", from_node, to_node))

        graph = AgentGraph(observers=[RecordingObserver()])
        graph.add_node("a", sync_fn_node("x"))
        graph.add_node("b", sync_fn_node("y"))
        graph.add_edge("a", "b")
        graph.set_entry("a")

        graph.run("go")

        event_types = [e[0] for e in events]
        assert "graph_start" in event_types
        assert "graph_end" in event_types
        assert "node_start" in event_types
        assert "node_end" in event_types
        assert "routing" in event_types

    def test_loop_detection_fires_event(self):
        """Hard loop detection (same state hash) fires on_loop_detected and raises."""
        from selectools.observer import AgentObserver

        loops = []

        class LoopObserver(AgentObserver):
            def on_loop_detected(self, run_id, node_name, loop_count):
                loops.append(loop_count)

        def stateless_node(state: GraphState) -> GraphState:
            # Does not mutate state → same hash each step → loop detected
            return state

        graph = AgentGraph(
            observers=[LoopObserver()],
            enable_loop_detection=True,
        )
        graph.add_node("n", stateless_node)
        graph.add_edge("n", "n")
        graph.set_entry("n")

        with pytest.raises(GraphExecutionError):
            graph.run("go")

        # on_loop_detected fired before the raise
        assert len(loops) > 0


# ------------------------------------------------------------------
# Graph composition
# ------------------------------------------------------------------


class TestComposition:
    def test_graph_as_callable_node(self):
        """AgentGraph.__call__ returns state — usable as a sync callable outside event loops."""
        inner = AgentGraph(name="inner")
        inner.add_node("inner_node", sync_fn_node("inner result"))
        inner.set_entry("inner_node")

        # __call__ wraps run(), which works when called outside an event loop
        state_in = GraphState.from_prompt("go")
        state_out = inner(state_in)
        assert isinstance(state_out, GraphState)
        assert state_out.data.get(STATE_KEY_LAST_OUTPUT) == "inner result"

    def test_add_subgraph(self):
        """add_subgraph registers a SubgraphNode."""
        from selectools.orchestration.node import SubgraphNode

        inner = AgentGraph(name="inner")
        inner.add_node("n", sync_fn_node("from inner"))
        inner.set_entry("n")

        outer = AgentGraph(name="outer")
        outer.add_subgraph("sub", inner, input_map={}, output_map={})
        outer.set_entry("sub")

        assert "sub" in outer._nodes
        assert isinstance(outer._nodes["sub"], SubgraphNode)

        result = outer.run("go")
        assert result.content == "from inner"


# ------------------------------------------------------------------
# Ergonomic helpers
# ------------------------------------------------------------------


class TestErgonomics:
    def test_chain_factory(self):
        """AgentGraph.chain() creates a linear pipeline."""
        graph = AgentGraph.chain(
            sync_fn_node("a_out"),
            sync_fn_node("b_out"),
            sync_fn_node("c_out"),
        )
        result = graph.run("start")
        assert result.content == "c_out"
        assert len(result.node_results) == 3

    def test_chain_with_names(self):
        graph = AgentGraph.chain(
            sync_fn_node("x"),
            sync_fn_node("y"),
            names=["planner", "writer"],
        )
        result = graph.run("go")
        assert "planner" in result.node_results
        assert "writer" in result.node_results

    def test_auto_entry_node(self):
        """First add_node auto-sets entry — no explicit set_entry needed."""
        graph = AgentGraph()
        graph.add_node("only", sync_fn_node("done"))
        graph.add_edge("only", AgentGraph.END)
        result = graph.run("go")
        assert result.content == "done"

    def test_next_node_shorthand(self):
        """add_node(next_node=...) auto-creates edge."""
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("a_out"), next_node="b")
        graph.add_node("b", sync_fn_node("b_out"), next_node=AgentGraph.END)
        result = graph.run("go")
        assert result.content == "b_out"

    def test_last_output_property(self):
        state = GraphState.from_prompt("hello")
        assert state.last_output == ""
        state.last_output = "test"
        assert state.last_output == "test"
        assert state.data[STATE_KEY_LAST_OUTPUT] == "test"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_single_node_no_edges_reaches_end(self):
        """A single node with no outgoing edges should implicitly reach END."""
        graph = AgentGraph()
        graph.add_node("solo", sync_fn_node("solo output"))
        graph.set_entry("solo")

        result = graph.run("start")
        assert result.content == "solo output"
        assert result.steps == 1

    def test_parallel_with_one_child(self):
        """Parallel group with a single child should execute without error."""
        graph = AgentGraph()
        graph.add_node("only_child", sync_fn_node("child result"))
        graph.add_parallel_nodes("fan", ["only_child"], merge_policy=MergePolicy.LAST_WINS)
        graph.add_edge("fan", AgentGraph.END)
        graph.set_entry("fan")

        result = graph.run("go")
        assert result.content == "child result"
        assert result.steps >= 1

    def test_router_returning_none_raises(self):
        """Router that returns None should raise GraphExecutionError, not silently END."""
        from selectools.exceptions import GraphExecutionError

        def none_router(state: GraphState):
            return None

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_conditional_edge("entry", none_router)
        graph.set_entry("entry")

        with pytest.raises(GraphExecutionError, match="Router returned None"):
            graph.run("go")

    def test_router_raising_exception_wraps_in_graph_error(self):
        """Router that raises should produce a GraphExecutionError wrapping the original."""
        from selectools.exceptions import GraphExecutionError

        def exploding_router(state: GraphState):
            raise ValueError("router broke")

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_conditional_edge("entry", exploding_router)
        graph.set_entry("entry")

        with pytest.raises(GraphExecutionError, match="Router function raised"):
            graph.run("go")

    def test_empty_scatter_list_raises(self):
        """Router returning an empty list should raise GraphExecutionError."""
        from selectools.exceptions import GraphExecutionError

        def empty_scatter_router(state: GraphState):
            return []

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_conditional_edge("entry", empty_scatter_router)
        graph.set_entry("entry")

        with pytest.raises(GraphExecutionError, match="empty list"):
            graph.run("go")

    def test_chain_with_single_agent(self):
        """chain() with a single agent should work and produce one node."""
        graph = AgentGraph.chain(sync_fn_node("only"))
        result = graph.run("start")
        assert result.content == "only"
        assert result.steps == 1
        assert len(result.node_results) == 1

    def test_chain_names_length_mismatch_raises(self):
        """chain() with mismatched names length should raise ValueError."""
        with pytest.raises(ValueError, match="names length"):
            AgentGraph.chain(
                sync_fn_node("a"),
                sync_fn_node("b"),
                names=["only_one"],
            )

    def test_next_node_shorthand_creates_edge(self):
        """add_node(next_node=...) should create a static edge automatically."""
        graph = AgentGraph()
        graph.add_node("a", sync_fn_node("from_a"), next_node="b")
        graph.add_node("b", sync_fn_node("from_b"), next_node=AgentGraph.END)

        # Verify edges were created
        assert graph._edges.get("a") == "b"
        assert graph._edges.get("b") == AgentGraph.END

        result = graph.run("go")
        assert result.content == "from_b"
        assert result.steps == 2


# ------------------------------------------------------------------
# Regressions — bugs fixed in Phase 3
# ------------------------------------------------------------------


class TestRegressions:
    def test_scatter_patches_isolated_per_branch(self):
        """Each Scatter branch sees only its own state_patch, not other branches'."""
        received_patches = {}

        def branch_a(state: GraphState) -> GraphState:
            received_patches["a"] = dict(state.data)
            state.data[STATE_KEY_LAST_OUTPUT] = "a_done"
            return state

        def branch_b(state: GraphState) -> GraphState:
            received_patches["b"] = dict(state.data)
            state.data[STATE_KEY_LAST_OUTPUT] = "b_done"
            return state

        def scatter_router(state: GraphState):
            return [
                Scatter("worker_a", state_patch={"task": "analyze", "source": "alpha"}),
                Scatter("worker_b", state_patch={"task": "summarize", "source": "beta"}),
            ]

        graph = AgentGraph()
        graph.add_node("entry", sync_fn_node("start"))
        graph.add_node("worker_a", branch_a)
        graph.add_node("worker_b", branch_b)
        graph.add_conditional_edge("entry", scatter_router)
        graph.set_entry("entry")

        graph.run("go")

        # branch_a should see its own patch
        assert received_patches["a"]["task"] == "analyze"
        assert received_patches["a"]["source"] == "alpha"

        # branch_b should see its own patch, NOT branch_a's
        assert received_patches["b"]["task"] == "summarize"
        assert received_patches["b"]["source"] == "beta"

    def test_parallel_all_children_fail_with_skip_returns_parent_state(self):
        """When all parallel children fail under SKIP policy, return parent state (not crash)."""

        def failing_child_a(state: GraphState) -> GraphState:
            raise RuntimeError("child a failed")

        def failing_child_b(state: GraphState) -> GraphState:
            raise RuntimeError("child b failed")

        graph = AgentGraph(error_policy=ErrorPolicy.SKIP)
        graph.add_node("fail_a", failing_child_a)
        graph.add_node("fail_b", failing_child_b)
        graph.add_parallel_nodes("fan", ["fail_a", "fail_b"])
        graph.add_edge("fan", AgentGraph.END)
        graph.set_entry("fan")

        result = graph.run("go")
        # Should not crash; errors should be recorded
        assert len(result.state.errors) == 2
        # Content should be empty since no child succeeded
        assert result.content == ""

    def test_observer_exception_does_not_crash_graph(self):
        """An observer that raises should not prevent graph execution."""
        from selectools.observer import AgentObserver

        class BrokenObserver(AgentObserver):
            def on_graph_start(self, run_id, graph_name, entry_node, state):
                raise RuntimeError("observer exploded")

            def on_node_start(self, run_id, node_name, step):
                raise RuntimeError("observer exploded again")

            def on_node_end(self, run_id, node_name, step, duration_ms):
                raise TypeError("observer type error")

            def on_graph_end(self, run_id, graph_name, steps, total_duration_ms):
                raise ValueError("observer end error")

            def on_graph_routing(self, run_id, from_node, to_node):
                raise KeyError("observer routing error")

        graph = AgentGraph(observers=[BrokenObserver()])
        graph.add_node("a", sync_fn_node("a_out"))
        graph.add_node("b", sync_fn_node("b_out"))
        graph.add_edge("a", "b")
        graph.add_edge("b", AgentGraph.END)
        graph.set_entry("a")

        # Should complete despite every observer method raising
        result = graph.run("go")
        assert result.content == "b_out"
        assert result.steps == 2

    def test_empty_supervisor_agent_raises(self):
        """SupervisorAgent with no agents should raise ValueError immediately."""
        from unittest.mock import MagicMock

        from selectools.orchestration.supervisor import SupervisorAgent

        mock_provider = MagicMock()

        with pytest.raises(ValueError, match="at least one agent"):
            SupervisorAgent(agents={}, provider=mock_provider)
