"""
Performance benchmarks — measure selectools overhead.

Measures the time selectools adds ON TOP of the LLM call,
not the LLM call itself. Uses a mock provider with zero latency.

Run: python tests/benchmarks/bench_overhead.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState
from selectools.pipeline import Pipeline, Step, parallel, step
from selectools.providers.stubs import LocalProvider


@tool(description="Fast tool")
def fast_tool(x: str) -> str:
    return f"result:{x}"


def bench(name, fn, iterations=100):
    """Run fn N times and report stats."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    p99 = sorted(times)[int(len(times) * 0.99)]
    mean = statistics.mean(times)
    print(f"  {name:40s}  mean={mean:6.2f}ms  p50={p50:6.2f}ms  p95={p95:6.2f}ms  p99={p99:6.2f}ms")
    return {"name": name, "mean": mean, "p50": p50, "p95": p95, "p99": p99}


def main():
    provider = LocalProvider()

    print("=" * 80)
    print("Selectools Performance Benchmarks")
    print("=" * 80)
    print(f"(100 iterations each, LocalProvider with ~0ms LLM latency)\n")

    # 1. Single agent run overhead
    print("--- Agent Core ---")
    agent = Agent(tools=[fast_tool], provider=provider, config=AgentConfig(max_iterations=1))
    bench("agent.run() single iteration", lambda: agent.run("test"))

    # 2. Agent with tool call
    agent2 = Agent(tools=[fast_tool], provider=provider, config=AgentConfig(max_iterations=3))
    bench("agent.run() with tool call", lambda: agent2.run("test"))

    # 3. AgentGraph overhead
    print("\n--- Orchestration ---")

    def fn_node(state: GraphState) -> GraphState:
        state.data[STATE_KEY_LAST_OUTPUT] = "done"
        return state

    graph1 = AgentGraph.chain(fn_node)
    bench("graph.run() 1 callable node", lambda: graph1.run("test"))

    graph3 = AgentGraph.chain(fn_node, fn_node, fn_node)
    bench("graph.run() 3 callable nodes", lambda: graph3.run("test"))

    graph_agent = AgentGraph.chain(agent)
    bench("graph.run() 1 agent node", lambda: graph_agent.run("test"))

    graph3_agent = AgentGraph.chain(agent, agent, agent)
    bench("graph.run() 3 agent nodes", lambda: graph3_agent.run("test"))

    # 4. Parallel overhead
    graph_par = AgentGraph()
    graph_par.add_node("a", fn_node)
    graph_par.add_node("b", fn_node)
    graph_par.add_node("c", fn_node)
    graph_par.add_parallel_nodes("all", ["a", "b", "c"])
    graph_par.add_edge("all", AgentGraph.END)
    graph_par.set_entry("all")
    bench("graph.run() 3 parallel nodes", lambda: graph_par.run("test"))

    # 5. Pipeline overhead
    print("\n--- Pipeline ---")

    @step
    def upper(x: str) -> str:
        return x.upper()

    @step
    def exclaim(x: str) -> str:
        return x + "!"

    pipe1 = Pipeline(steps=[upper])
    bench("pipeline.run() 1 step", lambda: pipe1.run("test"))

    pipe3 = upper | exclaim | upper
    bench("pipeline.run() 3 steps", lambda: pipe3.run("test"))

    pipe10 = Pipeline(steps=[Step(lambda x: x + ".") for _ in range(10)])
    bench("pipeline.run() 10 steps", lambda: pipe10.run("test"))

    # 6. Checkpoint overhead
    print("\n--- Checkpoint ---")
    from selectools.orchestration.checkpoint import InMemoryCheckpointStore

    store = InMemoryCheckpointStore()
    state = GraphState.from_prompt("test")
    state.data["key"] = "value" * 100

    bench("checkpoint save (InMemory)", lambda: store.save("g1", state, 1))
    cid = store.save("g1", state, 1)
    bench("checkpoint load (InMemory)", lambda: store.load(cid))

    # 7. Trace store overhead
    print("\n--- Trace Store ---")
    from selectools.observe import InMemoryTraceStore
    from selectools.trace import AgentTrace, StepType, TraceStep

    trace_store = InMemoryTraceStore()
    trace = AgentTrace(metadata={"bench": True})
    for _ in range(10):
        trace.add(TraceStep(type=StepType.LLM_CALL))

    bench("trace store save (InMemory)", lambda: trace_store.save(trace))
    rid = trace_store.save(trace)
    bench("trace store load (InMemory)", lambda: trace_store.load(rid))

    print(f"\n{'=' * 80}")
    print("Benchmark complete.")
    print("Note: These measure FRAMEWORK overhead only (LocalProvider ≈ 0ms LLM latency).")
    print("Real-world latency is dominated by LLM API calls (100-2000ms each).")


if __name__ == "__main__":
    main()
