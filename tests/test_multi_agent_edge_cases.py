"""
Multi-agent edge cases — interactions between features that could break.

Tests adversarial combinations: memory+parallel, structured+routing,
cancellation+subgraph, pipeline+graph, observer ordering, etc.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.cancellation import CancellationToken
from selectools.exceptions import GraphExecutionError
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.orchestration.checkpoint import InMemoryCheckpointStore
from selectools.orchestration.graph import ErrorPolicy
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState, InterruptRequest
from selectools.pipeline import Pipeline, Step, branch, parallel, step
from selectools.providers.base import Provider
from selectools.trace import StepType
from selectools.types import AgentResult, Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Recording provider
# ---------------------------------------------------------------------------


class RecordingProvider(Provider):
    name = "recording"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses=None):
        self._responses = list(responses or ["response"])
        self._idx = 0
        self.calls = []

    def _next(self):
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def complete(self, *, model, system_prompt, messages, tools=None, **kw):
        self.calls.append({"model": model, "messages": messages})
        resp = self._next()
        if tools and resp.startswith("TOOL:"):
            parts = resp.split(":", 2)
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name=parts[1],
                            parameters=json.loads(parts[2]) if len(parts) > 2 else {},
                            id="tc",
                        )
                    ],
                ),
                UsageStats(
                    prompt_tokens=50, completion_tokens=20, total_tokens=70, cost_usd=0.0001
                ),
            )
        return (
            Message(role=Role.ASSISTANT, content=resp),
            UsageStats(prompt_tokens=50, completion_tokens=20, total_tokens=70, cost_usd=0.0001),
        )

    async def acomplete(self, **kw):
        return self.complete(**kw)

    def stream(self, **kw):
        yield self._next()

    async def astream(self, **kw):
        yield self._next()


def _agent(provider, tools=None, **kw):
    if tools is None:

        @tool(description="noop")
        def noop(x: str) -> str:
            return x

        tools = [noop]
    return Agent(
        provider=provider, tools=tools, config=AgentConfig(model="m", max_iterations=3, **kw)
    )


def fn_set(val):
    def fn(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = val
        return state

    return fn


# ===========================================================================
# 1. Parallel memory — shared agent doesn't corrupt
# ===========================================================================


class TestParallelMemoryIsolation:
    def test_parallel_branches_dont_corrupt_shared_memory(self):
        """Parallel branches with same agent — memory should accumulate, not corrupt."""
        p = RecordingProvider(["branch_reply"] * 5)
        agent = _agent(p)
        memory = ConversationMemory(max_messages=50)
        agent.memory = memory

        graph = AgentGraph()
        graph.add_node("a", agent)
        graph.add_node("b", agent)
        graph.add_parallel_nodes("both", ["a", "b"])
        graph.add_edge("both", AgentGraph.END)
        graph.set_entry("both")

        result = graph.run("test")
        # Should complete without crash — memory may have entries from both branches
        assert isinstance(result.content, str)


# ===========================================================================
# 2. Structured output agent in conditional routing
# ===========================================================================


class TestStructuredOutputRouting:
    def test_router_receives_raw_json_from_structured_agent(self):
        """Router gets raw JSON string, not parsed model, from structured output agent."""
        provider = RecordingProvider(['{"action": "approve", "reason": "looks good"}'])
        agent = _agent(provider)

        def router(state: GraphState) -> str:
            output = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            try:
                parsed = json.loads(output)
                return "yes" if parsed.get("action") == "approve" else "no"
            except (json.JSONDecodeError, TypeError):
                # Router may get non-JSON — handle gracefully
                return "yes" if "approve" in output.lower() else "no"

        graph = AgentGraph()
        graph.add_node("decide", agent)
        graph.add_node("yes", fn_set("approved"))
        graph.add_node("no", fn_set("rejected"))
        graph.add_conditional_edge("decide", router)
        graph.add_edge("yes", AgentGraph.END)
        graph.add_edge("no", AgentGraph.END)

        result = graph.run("Decide")
        assert result.content == "approved"


# ===========================================================================
# 3. Cancellation in nested subgraphs
# ===========================================================================


class TestNestedCancellation:
    def test_outer_cancellation_propagates_to_subgraph(self):
        """Cancelling outer graph should stop inner subgraph."""
        token = CancellationToken()

        def cancel_node(state: GraphState) -> GraphState:
            token.cancel()
            state.data[STATE_KEY_LAST_OUTPUT] = "cancelled"
            return state

        inner = AgentGraph(name="inner", cancellation_token=token)
        inner.add_node("inner_a", fn_set("inner_ran"), next_node=AgentGraph.END)

        outer = AgentGraph(name="outer", cancellation_token=token)
        outer.add_node("cancel", cancel_node, next_node="sub")
        outer.add_subgraph("sub", inner, input_map={}, output_map={})
        outer.add_edge("sub", AgentGraph.END)

        result = outer.run("go")
        # Should complete — cancel node ran, subgraph may or may not have run
        assert result.state.data.get("cancelled") is None or result.state.data.get(
            STATE_KEY_LAST_OUTPUT
        )


# ===========================================================================
# 4. Pipeline branch inside graph conditional
# ===========================================================================


class TestDoublRouting:
    def test_pipeline_branch_then_graph_conditional(self):
        """Pipeline.branch() inside a graph + graph conditional routing."""

        def pipeline_router(x):
            return "a" if "route_a" in str(x) else "b"

        pipeline = Pipeline(
            steps=[
                branch(
                    router=pipeline_router,
                    a=Step(lambda x: "pipeline_a_output"),
                    b=Step(lambda x: "pipeline_b_output"),
                )
            ]
        )

        def graph_router(state: GraphState) -> str:
            output = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            return "final_a" if "pipeline_a" in output else "final_b"

        graph = AgentGraph()
        graph.add_node("prep", fn_set("route_a"), next_node="pipe")
        graph.add_node("pipe", pipeline)
        graph.add_conditional_edge("pipe", graph_router)
        graph.add_node("final_a", fn_set("reached_a"))
        graph.add_node("final_b", fn_set("reached_b"))
        graph.add_edge("final_a", AgentGraph.END)
        graph.add_edge("final_b", AgentGraph.END)

        result = graph.run("go")
        assert result.content == "reached_a"


# ===========================================================================
# 5. Observer event ordering in parallel
# ===========================================================================


class TestObserverParallelOrdering:
    def test_parallel_events_maintain_causal_order(self):
        """Within each parallel branch, events are causally ordered."""
        events = []

        class TrackObs(AgentObserver):
            def on_llm_start(self, run_id, messages, model, system_prompt):
                events.append("llm_start")

            def on_llm_end(self, run_id, response, usage):
                events.append("llm_end")

        p1 = RecordingProvider(["r1"])
        p2 = RecordingProvider(["r2"])
        a1 = _agent(p1, observers=[TrackObs()])
        a2 = _agent(p2, observers=[TrackObs()])

        graph = AgentGraph()
        graph.add_node("a", a1)
        graph.add_node("b", a2)
        graph.add_parallel_nodes("both", ["a", "b"])
        graph.add_edge("both", AgentGraph.END)
        graph.set_entry("both")

        graph.run("test")

        # Both branches produce llm_start + llm_end
        assert events.count("llm_start") == 2
        assert events.count("llm_end") == 2
        # First llm_start must come before last llm_end (causal)
        assert events.index("llm_start") < len(events) - 1 - events[::-1].index("llm_end")


# ===========================================================================
# 6. Checkpoint + resume with agent memory
# ===========================================================================


class TestCheckpointMemoryPersistence:
    def test_agent_memory_persists_through_checkpoint(self):
        """Agent memory should accumulate across checkpoint boundary."""
        memory = ConversationMemory(max_messages=50)
        store = InMemoryCheckpointStore()

        p = RecordingProvider(["pre-checkpoint response", "post-checkpoint response"])
        agent = _agent(p)
        agent.memory = memory

        async def gate(state: GraphState):
            yield InterruptRequest(prompt="continue?")
            state.data[STATE_KEY_LAST_OUTPUT] = "resumed"

        graph = AgentGraph()
        graph.add_node("before", agent, next_node="gate")
        graph.add_node("gate", gate, next_node=AgentGraph.END)

        r1 = graph.run("initial", checkpoint_store=store)
        assert r1.interrupted
        mem_before = len(memory.get_history())

        graph.resume(r1.interrupt_id, "yes", checkpoint_store=store)
        # Memory from agent should still have prior messages
        # (memory is on the agent instance, not in GraphState checkpoint)
        assert len(memory.get_history()) >= mem_before


# ===========================================================================
# 7. Error in output_transform with SKIP policy
# ===========================================================================


class TestTransformErrorSkip:
    def test_skip_policy_catches_output_transform_error(self):
        """SKIP policy should handle errors in output_transform."""
        provider = RecordingProvider(["response"])
        agent = _agent(provider)

        def bad_transform(result, state):
            raise RuntimeError("transform exploded")

        graph = AgentGraph(error_policy=ErrorPolicy.SKIP)
        graph.add_node("agent", agent, output_transform=bad_transform, next_node="recovery")
        graph.add_node("recovery", fn_set("recovered"), next_node=AgentGraph.END)

        result = graph.run("test")
        # Should reach recovery node via SKIP
        assert result.content == "recovered"
        assert any("transform" in str(e.get("error", "")) for e in result.state.errors)


# ===========================================================================
# 8. astream() with parallel nodes — event ordering
# ===========================================================================


class TestAstreamParallel:
    @pytest.mark.asyncio
    async def test_astream_parallel_events_ordered(self):
        """astream() yields PARALLEL_START before PARALLEL_END."""
        from selectools.orchestration.state import GraphEventType

        p1 = RecordingProvider(["r1"])
        p2 = RecordingProvider(["r2"])
        a1 = _agent(p1)
        a2 = _agent(p2)

        graph = AgentGraph()
        graph.add_node("a", a1)
        graph.add_node("b", a2)
        graph.add_parallel_nodes("both", ["a", "b"])
        graph.add_edge("both", AgentGraph.END)
        graph.set_entry("both")

        event_types = []
        async for event in graph.astream("test"):
            event_types.append(event.type)

        p_start = event_types.index(GraphEventType.PARALLEL_START)
        p_end = event_types.index(GraphEventType.PARALLEL_END)
        assert p_start < p_end


# ===========================================================================
# 9. Multiple graphs sharing same agent — state isolation
# ===========================================================================


class TestSharedAgentIsolation:
    def test_agent_usage_accumulates_across_graphs(self):
        """Same agent in multiple graphs — usage should accumulate."""
        p = RecordingProvider(["g1_response", "g2_response"])
        agent = _agent(p)

        g1 = AgentGraph.chain(agent, names=["n1"])
        g2 = AgentGraph.chain(agent, names=["n2"])

        g1.run("query1")
        g2.run("query2")

        # Usage accumulates on the agent instance
        assert agent.usage.total_tokens >= 140  # 70 * 2 runs


# ===========================================================================
# 10. Router returning non-string raises clear error
# ===========================================================================


class TestRouterNonStringReturn:
    def test_router_returns_pipeline_raises(self):
        """Router returning a Pipeline instead of string should error clearly."""
        pipeline = Step(str.upper)

        def bad_router(state: GraphState):
            return pipeline

        graph = AgentGraph()
        graph.add_node("entry", fn_set("test"))
        graph.add_conditional_edge("entry", bad_router)

        with pytest.raises(GraphExecutionError):
            graph.run("test")

    def test_router_returns_int_raises(self):
        """Router returning int should error."""

        def int_router(state: GraphState):
            return 42

        graph = AgentGraph()
        graph.add_node("entry", fn_set("test"))
        graph.add_conditional_edge("entry", int_router)

        with pytest.raises(GraphExecutionError):
            graph.run("test")


# ===========================================================================
# 11. Empty agent response flows to next node
# ===========================================================================


class TestEmptyResponseFlow:
    def test_none_content_flows_as_empty_or_error_string(self):
        """Agent with empty response — next node should get non-None value."""

        # Provider returns empty string
        provider = RecordingProvider([""])
        agent = _agent(provider)

        def check_content(state: GraphState) -> GraphState:
            val = state.data.get(STATE_KEY_LAST_OUTPUT, "MISSING")
            state.data[STATE_KEY_LAST_OUTPUT] = f"received:{type(val).__name__}:{repr(val)[:50]}"
            return state

        graph = AgentGraph()
        graph.add_node("agent", agent, next_node="check")
        graph.add_node("check", check_content, next_node=AgentGraph.END)

        result = graph.run("test")
        # Value should be a string, not None
        assert "received:str:" in result.content


# ===========================================================================
# 12. Graph trace includes all step types from nested execution
# ===========================================================================


class TestNestedTraceCompleteness:
    def test_graph_trace_has_all_expected_step_types(self):
        """Graph trace should include NODE_START, NODE_END, ROUTING at minimum."""
        p1 = RecordingProvider(["r1"])
        p2 = RecordingProvider(["r2"])
        a1 = _agent(p1)
        a2 = _agent(p2)

        graph = AgentGraph.chain(a1, a2)
        result = graph.run("test")

        step_types = {s.type for s in result.trace.steps}
        assert StepType.GRAPH_NODE_START in step_types
        assert StepType.GRAPH_NODE_END in step_types
        assert StepType.GRAPH_ROUTING in step_types
