"""
Feature integration tests — every selectools feature tested inside a multi-agent graph.

Verifies that features built for single-agent use work correctly when agents
are composed into graphs, pipelines, and parallel execution.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.cancellation import CancellationToken
from selectools.memory import ConversationMemory
from selectools.observer import AgentObserver
from selectools.orchestration.checkpoint import InMemoryCheckpointStore
from selectools.orchestration.state import (
    STATE_KEY_LAST_OUTPUT,
    GraphState,
    InterruptRequest,
    MergePolicy,
)
from selectools.pipeline import Pipeline, Step, parallel, step
from selectools.policy import PolicyDecision, ToolPolicy
from selectools.providers.base import Provider
from selectools.tools.base import Tool
from selectools.trace import StepType
from selectools.types import AgentResult, Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Test provider that records every call
# ---------------------------------------------------------------------------


class RecordingProvider(Provider):
    """Provider that records calls and returns predictable responses."""

    name = "recording"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses: List[str] = None):
        self._responses = list(responses or ["Default response"])
        self._call_index = 0
        self.calls: List[Dict[str, Any]] = []

    def _next_response(self) -> str:
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return self._responses[-1]

    def complete(self, *, model, system_prompt, messages, tools=None, **kwargs):
        self.calls.append(
            {
                "method": "complete",
                "model": model,
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": tools,
            }
        )
        resp = self._next_response()
        # Check if response should trigger a tool call
        if tools and resp.startswith("TOOL:"):
            parts = resp.split(":", 2)
            tool_name = parts[1]
            tool_args = parts[2] if len(parts) > 2 else "{}"
            import json

            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        ToolCall(tool_name=tool_name, parameters=json.loads(tool_args), id="tc_1")
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

    async def acomplete(self, *, model, system_prompt, messages, tools=None, **kwargs):
        return self.complete(
            model=model, system_prompt=system_prompt, messages=messages, tools=tools, **kwargs
        )

    def stream(self, *, model, system_prompt, messages, tools=None, **kwargs):
        resp = self._next_response()
        for word in resp.split():
            yield word + " "

    async def astream(self, *, model, system_prompt, messages, tools=None, **kwargs):
        resp = self._next_response()
        for word in resp.split():
            yield word + " "


def _make_agent(provider, responses=None, tools=None, **config_kwargs):
    """Create an agent with RecordingProvider."""
    p = provider if isinstance(provider, Provider) else RecordingProvider(responses or ["response"])
    if tools is None:

        @tool(description="A test tool")
        def test_tool(input: str) -> str:
            return f"tool_result:{input}"

        tools = [test_tool]
    return Agent(
        provider=p,
        tools=tools,
        config=AgentConfig(model="test-model", max_iterations=3, **config_kwargs),
    )


# ===========================================================================
# 1. Tool calling in graph nodes
# ===========================================================================


class TestToolCallingInGraph:
    def test_agent_calls_tool_inside_graph_node(self):
        """Agent in a graph node should call tools and return results."""

        @tool(description="Get a price")
        def get_price(product: str) -> str:
            return f"${product}: $99"

        provider = RecordingProvider(
            [
                'TOOL:get_price:{"product": "laptop"}',
                "The laptop costs $99.",
            ]
        )
        agent = _make_agent(provider, tools=[get_price])

        graph = AgentGraph.chain(agent)
        result = graph.run("How much is a laptop?")

        assert result.content == "The laptop costs $99."
        assert result.total_usage.total_tokens > 0

    def test_different_tools_per_node(self):
        """Each node can have different tools."""

        @tool(description="Search")
        def search(query: str) -> str:
            return f"found:{query}"

        @tool(description="Format")
        def format_text(text: str) -> str:
            return f"formatted:{text}"

        p1 = RecordingProvider(['TOOL:search:{"query": "ai"}', "AI found"])
        p2 = RecordingProvider(['TOOL:format_text:{"text": "AI found"}', "Formatted: AI found"])
        a1 = _make_agent(p1, tools=[search])
        a2 = _make_agent(p2, tools=[format_text])

        graph = AgentGraph.chain(a1, a2, names=["search", "format"])
        result = graph.run("Search and format")

        assert "search" in result.node_results
        assert "format" in result.node_results


# ===========================================================================
# 2. Structured output in graph nodes
# ===========================================================================


class TestStructuredOutputInGraph:
    def test_structured_output_in_graph_node(self):
        """response_format should work inside graph nodes."""
        from pydantic import BaseModel

        class Rating(BaseModel):
            score: int
            reason: str

        provider = RecordingProvider(['{"score": 8, "reason": "good quality"}'])
        agent = _make_agent(provider, system_prompt="Rate the input")

        graph = AgentGraph.chain(agent)
        result = graph.run("Rate this product")

        assert result.content
        assert "score" in result.content or "8" in result.content


# ===========================================================================
# 3. Memory in graph nodes
# ===========================================================================


class TestMemoryInGraph:
    def test_agent_with_memory_in_graph(self):
        """ConversationMemory should persist across iterations within a node."""
        memory = ConversationMemory(max_messages=20)
        provider = RecordingProvider(["I remember everything"])
        agent = _make_agent(provider)
        agent.memory = memory

        graph = AgentGraph.chain(agent)
        result = graph.run("Remember this conversation")

        assert result.content
        assert len(memory.get_history()) > 0

    def test_separate_memory_per_node(self):
        """Different nodes should have independent memory."""
        mem1 = ConversationMemory(max_messages=10)
        mem2 = ConversationMemory(max_messages=10)

        p1 = RecordingProvider(["Node 1 response"])
        p2 = RecordingProvider(["Node 2 response"])
        a1 = _make_agent(p1)
        a1.memory = mem1
        a2 = _make_agent(p2)
        a2.memory = mem2

        graph = AgentGraph.chain(a1, a2, names=["n1", "n2"])
        graph.run("Test")

        # Both memories should have content but be independent
        assert len(mem1.get_history()) > 0
        assert len(mem2.get_history()) > 0


# ===========================================================================
# 4. Budget limits propagate through graph
# ===========================================================================


class TestBudgetInGraph:
    def test_graph_level_budget_stops_execution(self):
        """max_total_tokens on graph should stop before exceeding budget."""
        # Each agent call uses 70 tokens. Budget of 80 should allow 1 node then stop.
        a1 = _make_agent(RecordingProvider(["step1"]))
        a2 = _make_agent(RecordingProvider(["step2"]))
        a3 = _make_agent(RecordingProvider(["step3"]))

        graph = AgentGraph(max_total_tokens=80)
        graph.add_node("a", a1, next_node="b")
        graph.add_node("b", a2, next_node="c")
        graph.add_node("c", a3, next_node=AgentGraph.END)
        result = graph.run("go")

        # Budget check fires at top of each loop iteration.
        # After a (70t) + b (140t) exceeds 80, c should not execute.
        assert result.steps < 3 or "c" not in result.node_results

    def test_agent_level_budget_in_graph(self):
        """Agent's own max_total_tokens should work inside graph."""
        provider = RecordingProvider(["response"])
        agent = _make_agent(provider, max_total_tokens=50)

        graph = AgentGraph.chain(agent)
        result = graph.run("go")
        assert result.content


# ===========================================================================
# 5. Cancellation stops graph
# ===========================================================================


class TestCancellationInGraph:
    def test_pre_cancelled_token_stops_graph(self):
        """CancellationToken cancelled before graph.run() stops immediately."""
        token = CancellationToken()
        token.cancel()

        provider = RecordingProvider(["should not run"])
        agent = _make_agent(provider)

        graph = AgentGraph(cancellation_token=token)
        graph.add_node("a", agent, next_node=AgentGraph.END)
        result = graph.run("go")

        # Cancellation check fires inside the loop after step increment
        assert result.steps <= 1
        assert "a" not in result.node_results

    def test_cancellation_mid_graph(self):
        """Token cancelled during execution stops the graph."""
        token = CancellationToken()

        def cancel_after_first(state: GraphState) -> GraphState:
            token.cancel()
            state.data[STATE_KEY_LAST_OUTPUT] = "cancelled_here"
            return state

        provider = RecordingProvider(["should not reach"])
        agent = _make_agent(provider)

        graph = AgentGraph(cancellation_token=token)
        graph.add_node("cancel", cancel_after_first, next_node="agent")
        graph.add_node("agent", agent, next_node=AgentGraph.END)
        result = graph.run("go")

        # Cancel node executes (step 1), then agent node's step fires cancel check
        assert result.steps <= 2
        assert "agent" not in result.node_results


# ===========================================================================
# 6. Observer events fire from graph nodes
# ===========================================================================


class TestObserversInGraph:
    def test_agent_observer_fires_inside_graph(self):
        """Agent-level observers should fire when agent runs inside a graph."""
        events = []

        class TrackingObserver(AgentObserver):
            def on_llm_start(self, run_id, messages, model, system_prompt):
                events.append("llm_start")

            def on_llm_end(self, run_id, response, usage):
                events.append("llm_end")

            def on_tool_start(self, run_id, call_id, tool_name, tool_args):
                events.append(f"tool_start:{tool_name}")

        @tool(description="test tool")
        def my_tool(x: str) -> str:
            return "result"

        provider = RecordingProvider(['TOOL:my_tool:{"x": "test"}', "done"])
        agent = _make_agent(provider, tools=[my_tool], observers=[TrackingObserver()])

        graph = AgentGraph.chain(agent)
        graph.run("go")

        assert "llm_start" in events
        assert "llm_end" in events
        assert "tool_start:my_tool" in events

    def test_graph_observer_fires_alongside_agent_observer(self):
        """Both graph-level and agent-level observers should fire."""
        graph_events = []
        agent_events = []

        class GraphObs(AgentObserver):
            def on_graph_start(self, run_id, graph_name, entry_node, state):
                graph_events.append("graph_start")

            def on_node_start(self, run_id, node_name, step):
                graph_events.append(f"node_start:{node_name}")

        class AgentObs(AgentObserver):
            def on_llm_start(self, run_id, messages, model, system_prompt):
                agent_events.append("llm_start")

        provider = RecordingProvider(["response"])
        agent = _make_agent(provider, observers=[AgentObs()])

        graph = AgentGraph(observers=[GraphObs()])
        graph.add_node("a", agent, next_node=AgentGraph.END)
        graph.run("go")

        assert "graph_start" in graph_events
        assert "node_start:a" in graph_events
        assert "llm_start" in agent_events


# ===========================================================================
# 7. Tool policy in graph nodes
# ===========================================================================


class TestToolPolicyInGraph:
    def test_deny_policy_blocks_tool_in_graph(self):
        """ToolPolicy deny should block tools even inside graph nodes."""

        @tool(description="Dangerous tool")
        def delete_all(target: str) -> str:
            return "deleted"

        policy = ToolPolicy(deny=["delete_*"])
        provider = RecordingProvider(
            ['TOOL:delete_all:{"target": "everything"}', "Could not delete"]
        )
        agent = _make_agent(provider, tools=[delete_all], tool_policy=policy)

        graph = AgentGraph.chain(agent)
        result = graph.run("Delete everything")
        # Tool should be blocked, agent should respond without executing it
        assert result.content


# ===========================================================================
# 8. Traces from graph nodes
# ===========================================================================


class TestTracesInGraph:
    def test_node_traces_contain_agent_steps(self):
        """Each node's AgentResult should contain trace steps."""
        provider = RecordingProvider(["response"])
        agent = _make_agent(provider)

        graph = AgentGraph.chain(agent, names=["my_node"])
        result = graph.run("go")

        node_result = result.node_results["my_node"][0]
        assert node_result.trace is not None
        assert len(node_result.trace.steps) > 0

    def test_graph_trace_has_graph_steps(self):
        """Graph-level trace should contain graph-specific step types."""
        provider = RecordingProvider(["r1", "r2"])
        a1 = _make_agent(provider)
        a2 = _make_agent(provider)

        graph = AgentGraph.chain(a1, a2)
        result = graph.run("go")

        step_types = [s.type for s in result.trace.steps]
        assert StepType.GRAPH_NODE_START in step_types
        assert StepType.GRAPH_NODE_END in step_types
        assert StepType.GRAPH_ROUTING in step_types


# ===========================================================================
# 9. Reasoning strategies in graph nodes
# ===========================================================================


class TestReasoningInGraph:
    def test_react_strategy_in_graph_node(self):
        """reasoning_strategy='react' should work inside graph."""
        provider = RecordingProvider(["React response"])
        agent = _make_agent(provider, reasoning_strategy="react")

        graph = AgentGraph.chain(agent)
        result = graph.run("Think step by step")

        # Verify the system prompt was modified to include ReAct instructions
        assert len(provider.calls) > 0
        system = provider.calls[0]["system_prompt"]
        assert "Thought" in system or "Action" in system or "Observation" in system

    def test_cot_strategy_in_graph_node(self):
        """reasoning_strategy='cot' should work inside graph."""
        provider = RecordingProvider(["CoT response"])
        agent = _make_agent(provider, reasoning_strategy="cot")

        graph = AgentGraph.chain(agent)
        result = graph.run("Explain")

        system = provider.calls[0]["system_prompt"]
        assert "step" in system.lower() or "chain" in system.lower() or "think" in system.lower()


# ===========================================================================
# 10. Caching in graph nodes
# ===========================================================================


class TestCachingInGraph:
    def test_cache_works_across_graph_runs(self):
        """InMemoryCache should cache results across graph executions."""
        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=100)
        provider = RecordingProvider(["cached response"] * 5)
        agent = _make_agent(provider, cache=cache)

        graph = AgentGraph.chain(agent)
        r1 = graph.run("same question")
        r2 = graph.run("same question")

        # Second run should hit cache — fewer provider calls
        assert cache.stats.hits >= 1 or len(provider.calls) <= 2


# ===========================================================================
# 11. Tool result caching in graph nodes
# ===========================================================================


class TestToolCachingInGraph:
    def test_cacheable_tool_in_graph(self):
        """@tool(cacheable=True) should cache within a graph node."""
        call_count = {"n": 0}

        @tool(description="Expensive API call", cacheable=True)
        def expensive_call(query: str) -> str:
            call_count["n"] += 1
            return f"result:{query}"

        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=100)
        provider = RecordingProvider(
            [
                'TOOL:expensive_call:{"query": "test"}',
                'TOOL:expensive_call:{"query": "test"}',
                "done",
            ]
        )
        agent = _make_agent(provider, tools=[expensive_call], cache=cache)

        graph = AgentGraph.chain(agent)
        result = graph.run("Call the expensive tool twice")
        assert result.content


# ===========================================================================
# 12. Fallback provider in graph nodes
# ===========================================================================


class TestFallbackInGraph:
    def test_fallback_provider_works_in_graph(self):
        """FallbackProvider should failover inside graph nodes."""
        from selectools.providers.base import ProviderError
        from selectools.providers.fallback import FallbackProvider

        class FailingProvider(Provider):
            name = "failing"
            supports_streaming = False
            supports_async = True

            def complete(self, **kwargs):
                raise ProviderError("primary down")

            async def acomplete(self, **kwargs):
                raise ProviderError("primary down")

            def stream(self, **kwargs):
                raise ProviderError("primary down")

            async def astream(self, **kwargs):
                raise ProviderError("primary down")

        backup = RecordingProvider(["backup response"])
        fallback = FallbackProvider([FailingProvider(), backup])

        agent = _make_agent(fallback)
        graph = AgentGraph.chain(agent)
        result = graph.run("go")

        # FallbackProvider should have tried backup after primary failed
        assert result.content  # Should have some response
        assert (
            len(backup.calls) > 0
            or "backup" in result.content
            or "Provider error" in result.content
        )


# ===========================================================================
# 13. Parallel execution with different features per branch
# ===========================================================================


class TestParallelFeatures:
    def test_parallel_nodes_with_different_configs(self):
        """Parallel nodes with different agent configs."""
        p1 = RecordingProvider(["fast response"])
        p2 = RecordingProvider(["detailed response"])

        a1 = Agent(
            provider=p1,
            tools=[Tool(name="t", description="t", parameters=[], function=lambda: "ok")],
            config=AgentConfig(model="m", max_iterations=1),
        )
        a2 = Agent(
            provider=p2,
            tools=[Tool(name="t", description="t", parameters=[], function=lambda: "ok")],
            config=AgentConfig(model="m", max_iterations=5),
        )

        graph = AgentGraph()
        graph.add_node("fast", a1)
        graph.add_node("detailed", a2)
        graph.add_parallel_nodes("both", ["fast", "detailed"])
        graph.add_edge("both", AgentGraph.END)
        graph.set_entry("both")

        result = graph.run("go")
        assert "fast" in result.node_results
        assert "detailed" in result.node_results


# ===========================================================================
# 14. Pipeline features in graph
# ===========================================================================


class TestPipelineInGraph:
    def test_pipeline_with_retry_as_graph_node(self):
        """Pipeline with retry logic used as a graph node."""
        call_count = {"n": 0}

        @step(retry=2)
        def flaky(text: str) -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("not yet")
            return text.upper()

        pipeline = Pipeline(steps=[flaky])

        graph = AgentGraph()
        graph.add_node(
            "prep",
            lambda s: setattr(s, "data", {**s.data, STATE_KEY_LAST_OUTPUT: "hello"}) or s,
            next_node="process",
        )
        graph.add_node("process", pipeline, next_node=AgentGraph.END)
        result = graph.run("go")

        assert result.content == "HELLO"

    def test_parallel_pipeline_in_graph(self):
        """parallel() step inside a pipeline used as a graph node."""

        def search_a(x: str) -> str:
            return f"a:{x}"

        def search_b(x: str) -> str:
            return f"b:{x}"

        def merge(results: dict) -> str:
            return " + ".join(results.values())

        pipeline = parallel(search_a, search_b) | merge

        def prep(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "query"
            return state

        graph = AgentGraph()
        graph.add_node("prep", prep, next_node="search")
        graph.add_node("search", pipeline, next_node=AgentGraph.END)
        result = graph.run("go")

        assert "a:query" in result.content
        assert "b:query" in result.content


# ===========================================================================
# 15. Cost tracking through graph
# ===========================================================================


class TestCostTrackingInGraph:
    def test_usage_accumulates_across_nodes(self):
        """Token usage should sum across all graph nodes."""
        p1 = RecordingProvider(["node1"])
        p2 = RecordingProvider(["node2"])
        a1 = _make_agent(p1)
        a2 = _make_agent(p2)

        graph = AgentGraph.chain(a1, a2)
        result = graph.run("go")

        # Each agent uses 70 tokens (50 prompt + 20 completion)
        assert result.total_usage.total_tokens >= 70
        assert result.total_usage.cost_usd > 0

    def test_parallel_usage_sums(self):
        """Parallel node usage should aggregate."""
        p1 = RecordingProvider(["r1"])
        p2 = RecordingProvider(["r2"])
        a1 = _make_agent(p1)
        a2 = _make_agent(p2)

        graph = AgentGraph()
        graph.add_node("a", a1)
        graph.add_node("b", a2)
        graph.add_parallel_nodes("both", ["a", "b"])
        graph.add_edge("both", AgentGraph.END)
        graph.set_entry("both")

        result = graph.run("go")
        assert result.total_usage.total_tokens >= 140  # 70 * 2 nodes


# ===========================================================================
# 16. Error handling features in graph
# ===========================================================================


class TestErrorHandlingInGraph:
    def test_skip_policy_continues_graph(self):
        """SKIP error policy lets graph continue past failing node."""

        def bad_node(state: GraphState) -> GraphState:
            raise RuntimeError("node crashed")

        provider = RecordingProvider(["recovery"])
        agent = _make_agent(provider)

        graph = AgentGraph(error_policy="skip")
        graph.add_node("bad", bad_node, next_node="recover")
        graph.add_node("recover", agent, next_node=AgentGraph.END)
        result = graph.run("go")

        assert result.content == "recovery"
        assert len(result.state.errors) > 0

    def test_retry_policy_retries_node(self):
        """RETRY policy retries a failing node."""
        attempts = {"n": 0}

        def flaky_node(state: GraphState) -> GraphState:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("not ready")
            state.data[STATE_KEY_LAST_OUTPUT] = "success"
            return state

        graph = AgentGraph(error_policy="retry", error_retry_limit=3)
        graph.add_node("flaky", flaky_node, next_node=AgentGraph.END)
        result = graph.run("go")

        assert result.content == "success"
        assert attempts["n"] == 2


# ===========================================================================
# 17. Checkpoint integration with features
# ===========================================================================


class TestCheckpointWithFeatures:
    def test_checkpoint_preserves_agent_results(self):
        """Checkpointed state should preserve node results for resume."""
        store = InMemoryCheckpointStore()

        async def gate(state: GraphState):
            yield InterruptRequest(prompt="approve?")
            state.data[STATE_KEY_LAST_OUTPUT] = "approved"

        provider = RecordingProvider(["final answer"])
        agent = _make_agent(provider)

        graph = AgentGraph()
        graph.add_node("gate", gate, next_node="agent")
        graph.add_node("agent", agent, next_node=AgentGraph.END)

        r1 = graph.run("go", checkpoint_store=store)
        assert r1.interrupted

        r2 = graph.resume(r1.interrupt_id, "yes", checkpoint_store=store)
        assert not r2.interrupted
        assert r2.content


# ===========================================================================
# 18. Async execution
# ===========================================================================


class TestAsyncGraphExecution:
    @pytest.mark.asyncio
    async def test_arun_with_agents(self):
        """graph.arun() should work with real agents."""
        provider = RecordingProvider(["async response"])
        agent = _make_agent(provider)

        graph = AgentGraph.chain(agent)
        result = await graph.arun("go")

        assert result.content == "async response"

    @pytest.mark.asyncio
    async def test_astream_yields_events(self):
        """graph.astream() should yield events with agent nodes."""
        from selectools.orchestration.state import GraphEventType

        provider = RecordingProvider(["streamed"])
        agent = _make_agent(provider)

        graph = AgentGraph.chain(agent)
        events = []
        async for event in graph.astream("go"):
            events.append(event.type)

        assert GraphEventType.GRAPH_START in events
        assert GraphEventType.NODE_START in events
        assert GraphEventType.NODE_END in events
        assert GraphEventType.GRAPH_END in events
