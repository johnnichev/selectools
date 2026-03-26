"""
Example 55: Linear AgentGraph pipeline

Demonstrates a simple 3-node linear graph:
  planner → writer → reviewer → END

Uses LocalProvider so no API keys are needed.
"""

from selectools import Agent, AgentConfig
from selectools.orchestration import STATE_KEY_LAST_OUTPUT, AgentGraph, GraphState
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"


def make_agent(name: str, responses: list) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=responses),
        tools=[search],
    )


def main():
    # Create three agents for the pipeline
    planner = make_agent("planner", ["Plan: 1) Research 2) Draft 3) Review"])
    writer = make_agent("writer", ["Draft article about AI agents..."])
    reviewer = make_agent("reviewer", ["Looks good! Minor improvements needed."])

    # Build a linear graph
    graph = AgentGraph(name="blog_pipeline")
    graph.add_node("planner", planner)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)

    graph.add_edge("planner", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", AgentGraph.END)
    graph.set_entry("planner")

    # Run the pipeline
    result = graph.run("Write a blog post about AI agents")

    print("=== Linear Graph Result ===")
    print(f"Final output: {result.content}")
    print(f"Steps executed: {result.steps}")
    print(f"Nodes visited: {list(result.node_results.keys())}")
    print(f"Total tokens: {result.total_usage.total_tokens}")

    # Show execution trace
    print("\n=== Execution Trace ===")
    for step in result.trace.steps[:5]:  # first 5 steps
        print(f"  [{step.type}] {step.node_name or ''}")


if __name__ == "__main__":
    main()
