"""Integration tests for orchestration: observers, StepTypes, exports."""

from __future__ import annotations

import pytest

from selectools.observer import AgentObserver, LoggingObserver, SimpleStepObserver
from selectools.orchestration import (
    STATE_KEY_LAST_OUTPUT,
    AgentGraph,
    CheckpointStore,
    ContextMode,
    ErrorPolicy,
    FileCheckpointStore,
    GraphEvent,
    GraphEventType,
    GraphNode,
    GraphResult,
    GraphState,
    InMemoryCheckpointStore,
    InterruptRequest,
    MergePolicy,
    ModelSplit,
    ParallelGroupNode,
    Scatter,
    SQLiteCheckpointStore,
    SubgraphNode,
    SupervisorAgent,
    SupervisorStrategy,
    goto,
    merge_states,
    update,
)
from selectools.trace import StepType


def sync_fn(content):
    def fn(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = content
        return state

    return fn


class TestPublicExports:
    def test_all_classes_importable_from_orchestration(self):
        """All orchestration exports are importable from selectools.orchestration."""
        assert AgentGraph is not None
        assert GraphState is not None
        assert GraphResult is not None
        assert GraphNode is not None
        assert ParallelGroupNode is not None
        assert SubgraphNode is not None
        assert InterruptRequest is not None
        assert Scatter is not None
        assert MergePolicy is not None
        assert ContextMode is not None
        assert ErrorPolicy is not None
        assert InMemoryCheckpointStore is not None
        assert FileCheckpointStore is not None
        assert SQLiteCheckpointStore is not None
        assert SupervisorAgent is not None
        assert SupervisorStrategy is not None
        assert ModelSplit is not None
        assert STATE_KEY_LAST_OUTPUT == "__last_output__"

    def test_all_exports_importable_from_selectools(self):
        """All orchestration exports importable from top-level selectools package."""
        import selectools

        assert hasattr(selectools, "AgentGraph")
        assert hasattr(selectools, "GraphState")
        assert hasattr(selectools, "GraphResult")
        assert hasattr(selectools, "InterruptRequest")
        assert hasattr(selectools, "SupervisorAgent")
        assert hasattr(selectools, "InMemoryCheckpointStore")
        assert hasattr(selectools, "FileCheckpointStore")
        assert hasattr(selectools, "SQLiteCheckpointStore")
        assert hasattr(selectools, "MergePolicy")
        assert hasattr(selectools, "ErrorPolicy")

    def test_version_updated(self):
        import selectools

        assert selectools.__version__ == "0.18.0"


class TestNewStepTypes:
    def test_all_graph_step_types_exist(self):
        graph_types = [
            StepType.GRAPH_NODE_START,
            StepType.GRAPH_NODE_END,
            StepType.GRAPH_ROUTING,
            StepType.GRAPH_CHECKPOINT,
            StepType.GRAPH_INTERRUPT,
            StepType.GRAPH_RESUME,
            StepType.GRAPH_PARALLEL_START,
            StepType.GRAPH_PARALLEL_END,
            StepType.GRAPH_STALL,
            StepType.GRAPH_LOOP_DETECTED,
        ]
        for st in graph_types:
            assert isinstance(st, StepType)

    def test_total_step_type_count(self):
        assert len(list(StepType)) == 27

    def test_graph_step_type_string_equality(self):
        assert StepType.GRAPH_NODE_START == "graph_node_start"
        assert StepType.GRAPH_ROUTING == "graph_routing"
        assert StepType.GRAPH_INTERRUPT == "graph_interrupt"
        assert StepType.GRAPH_LOOP_DETECTED == "graph_loop_detected"


class TestNewObserverMethods:
    def test_agent_observer_has_all_13_methods(self):
        obs = AgentObserver()
        graph_methods = [
            "on_graph_start",
            "on_graph_end",
            "on_graph_error",
            "on_node_start",
            "on_node_end",
            "on_graph_routing",
            "on_graph_interrupt",
            "on_graph_resume",
            "on_parallel_start",
            "on_parallel_end",
            "on_stall_detected",
            "on_loop_detected",
            "on_supervisor_replan",
        ]
        for method in graph_methods:
            assert hasattr(obs, method), f"Missing: {method}"
            assert callable(getattr(obs, method))

    def test_logging_observer_implements_all_graph_methods(self):
        obs = LoggingObserver()
        graph_methods = [
            "on_graph_start",
            "on_graph_end",
            "on_graph_error",
            "on_node_start",
            "on_node_end",
            "on_graph_routing",
            "on_graph_interrupt",
            "on_graph_resume",
            "on_parallel_start",
            "on_parallel_end",
            "on_stall_detected",
            "on_loop_detected",
            "on_supervisor_replan",
        ]
        for method in graph_methods:
            assert hasattr(obs, method), f"LoggingObserver missing: {method}"

    def test_logging_observer_graph_methods_callable(self):
        """LoggingObserver graph methods call _emit without raising."""
        import logging

        obs = LoggingObserver()
        # Should not raise
        obs.on_graph_start("run1", "my_graph", "entry_node", {})
        obs.on_graph_end("run1", "my_graph", 5, 1234.5)
        obs.on_graph_error("run1", "my_graph", "node_a", RuntimeError("fail"))
        obs.on_node_start("run1", "node_a", 1)
        obs.on_node_end("run1", "node_a", 1, 100.0)
        obs.on_graph_routing("run1", "node_a", "node_b")
        obs.on_graph_interrupt("run1", "node_a", "ckpt_id_123")
        obs.on_graph_resume("run1", "node_a", "ckpt_id_123")
        obs.on_parallel_start("run1", "fan_out", ["a", "b"])
        obs.on_parallel_end("run1", "fan_out", 2)
        obs.on_stall_detected("run1", "node_a", 1)
        obs.on_loop_detected("run1", "node_a", 1)
        obs.on_supervisor_replan("run1", 2, "new plan json")

    def test_simple_step_observer_routes_graph_events(self):
        received = []

        def cb(event_type, run_id, **kwargs):
            received.append(event_type)

        obs = SimpleStepObserver(cb)
        obs.on_graph_start("r", "g", "e", {})
        obs.on_node_start("r", "n", 1)
        obs.on_stall_detected("r", "n", 1)
        obs.on_loop_detected("r", "n", 1)
        obs.on_supervisor_replan("r", 2, "plan")

        assert "graph_start" in received
        assert "node_start" in received
        assert "stall_detected" in received
        assert "loop_detected" in received
        assert "supervisor_replan" in received

    def test_async_observer_has_all_13_async_methods(self):
        from selectools.observer import AsyncAgentObserver

        obs = AsyncAgentObserver()
        async_methods = [
            "a_on_graph_start",
            "a_on_graph_end",
            "a_on_graph_error",
            "a_on_node_start",
            "a_on_node_end",
            "a_on_graph_routing",
            "a_on_graph_interrupt",
            "a_on_graph_resume",
            "a_on_parallel_start",
            "a_on_parallel_end",
            "a_on_stall_detected",
            "a_on_loop_detected",
            "a_on_supervisor_replan",
        ]
        for method in async_methods:
            assert hasattr(obs, method), f"AsyncAgentObserver missing: {method}"
            assert callable(getattr(obs, method))


class TestTraceHierarchy:
    def test_graph_trace_has_graph_step_types(self):
        """Graph-level trace contains GRAPH_NODE_START, GRAPH_ROUTING steps."""
        graph = AgentGraph()
        graph.add_node("a", sync_fn("x"))
        graph.add_node("b", sync_fn("y"))
        graph.add_edge("a", "b")
        graph.set_entry("a")

        result = graph.run("go")
        step_types = {s.type for s in result.trace.steps}
        assert StepType.GRAPH_NODE_START in step_types
        assert StepType.GRAPH_ROUTING in step_types

    def test_checkpoint_step_in_trace(self):
        store = InMemoryCheckpointStore()
        graph = AgentGraph()
        graph.add_node("a", sync_fn("x"))
        graph.set_entry("a")

        result = graph.run("go", checkpoint_store=store)
        step_types = {s.type for s in result.trace.steps}
        assert StepType.GRAPH_CHECKPOINT in step_types


class TestObserverIntegration:
    def test_all_graph_events_fire_in_order(self):
        """Verify on_graph_start fires before on_node_start fires before on_graph_end."""
        order = []

        class OrderObserver(AgentObserver):
            def on_graph_start(self, run_id, graph_name, entry_node, state):
                order.append("graph_start")

            def on_node_start(self, run_id, node_name, step):
                order.append(f"node_start:{node_name}")

            def on_node_end(self, run_id, node_name, step, duration_ms):
                order.append(f"node_end:{node_name}")

            def on_graph_routing(self, run_id, from_node, to_node):
                order.append(f"routing:{from_node}→{to_node}")

            def on_graph_end(self, run_id, graph_name, steps, total_duration_ms):
                order.append("graph_end")

        graph = AgentGraph(observers=[OrderObserver()])
        graph.add_node("first", sync_fn("x"))
        graph.add_node("second", sync_fn("y"))
        graph.add_edge("first", "second")
        graph.set_entry("first")

        graph.run("go")

        assert order[0] == "graph_start"
        assert order[-1] == "graph_end"
        assert "node_start:first" in order
        assert "node_end:first" in order
        assert "node_start:second" in order

    def test_parallel_events_fire(self):
        parallel_events = []

        class ParallelObserver(AgentObserver):
            def on_parallel_start(self, run_id, group_name, child_nodes):
                parallel_events.append(("start", group_name))

            def on_parallel_end(self, run_id, group_name, child_count):
                parallel_events.append(("end", group_name))

        graph = AgentGraph(observers=[ParallelObserver()])
        graph.add_node("a", sync_fn("a"))
        graph.add_node("b", sync_fn("b"))
        graph.add_parallel_nodes("fan", ["a", "b"])
        graph.set_entry("fan")

        graph.run("go")

        types = [e[0] for e in parallel_events]
        assert "start" in types
        assert "end" in types
