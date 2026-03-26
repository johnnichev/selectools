"""
Example 61: Nested subgraphs with AgentGraph

Demonstrates SubgraphNode — an AgentGraph embedded as a node in another graph.
Uses input_map and output_map for explicit state key translation.

Uses LocalProvider so no API keys are needed.
"""

from selectools import Agent, AgentConfig
from selectools.orchestration import STATE_KEY_LAST_OUTPUT, AgentGraph, GraphState
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def analyze(text: str) -> str:
    """Analyze text content."""
    return f"Analysis of: {text[:50]}..."


def make_agent(response: str) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=[response]),
        tools=[analyze],
    )


def build_review_subgraph() -> AgentGraph:
    """Inner graph: draft → revise → approve."""
    drafter = make_agent("Inner draft: content created")
    reviser = make_agent("Inner revision: content improved")
    approver = make_agent("Inner approval: content approved")

    inner = AgentGraph(name="review_subgraph")
    inner.add_node("draft", drafter)
    inner.add_node("revise", reviser)
    inner.add_node("approve", approver)
    inner.add_edge("draft", "revise")
    inner.add_edge("revise", "approve")
    inner.add_edge("approve", AgentGraph.END)
    inner.set_entry("draft")
    return inner


def main():
    prep_agent = make_agent("Preparation complete: topic selected and research gathered")
    publish_agent = make_agent("Published! Content is live.")

    review_subgraph = build_review_subgraph()

    # Outer graph: prep → [subgraph] → publish
    outer = AgentGraph(name="content_pipeline")
    outer.add_node("prep", prep_agent)
    outer.add_subgraph(
        "review",
        review_subgraph,
        input_map={},  # pass state as-is to subgraph
        output_map={},  # merge subgraph output back
    )
    outer.add_node("publish", publish_agent)
    outer.add_edge("prep", "review")
    outer.add_edge("review", "publish")
    outer.add_edge("publish", AgentGraph.END)
    outer.set_entry("prep")

    print("=== Subgraph Composition ===")
    print("Outer graph structure:")
    print(outer.to_mermaid())
    print()

    result = outer.run("Create and publish a tech article")
    print(f"Final output: {result.content}")
    print(f"Total steps: {result.steps}")
    print(f"History entries: {len(result.state.history)}")

    # Show all nodes that contributed
    print(f"\nNodes with results: {list(result.node_results.keys())}")


if __name__ == "__main__":
    main()
