"""
Example 56: Parallel fan-out with AgentGraph

Demonstrates parallel execution of multiple agents with state merging:
  entry → [researcher_a, researcher_b, researcher_c] → summarizer → END

Uses LocalProvider so no API keys are needed.
"""

from selectools import Agent, AgentConfig
from selectools.orchestration import STATE_KEY_LAST_OUTPUT, AgentGraph, GraphState, MergePolicy
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def fetch_data(source: str) -> str:
    """Fetch data from a source."""
    return f"Data from {source}: [sample content]"


def make_agent(responses: list) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=responses),
        tools=[fetch_data],
    )


def main():
    # Create parallel research agents
    researcher_a = make_agent(["Research from source A: AI safety findings"])
    researcher_b = make_agent(["Research from source B: Alignment techniques"])
    researcher_c = make_agent(["Research from source C: Recent breakthroughs"])
    summarizer = make_agent(["Summary: Combining all research into key findings..."])

    graph = AgentGraph(name="parallel_research")

    # Register individual researcher nodes
    graph.add_node("researcher_a", researcher_a)
    graph.add_node("researcher_b", researcher_b)
    graph.add_node("researcher_c", researcher_c)
    graph.add_node("summarizer", summarizer)

    # Register a parallel group that fans out to all three researchers
    graph.add_parallel_nodes(
        "research_phase",
        ["researcher_a", "researcher_b", "researcher_c"],
        merge_policy=MergePolicy.APPEND,
    )

    graph.add_edge("research_phase", "summarizer")
    graph.add_edge("summarizer", AgentGraph.END)
    graph.set_entry("research_phase")

    result = graph.run("Research AI safety from multiple sources")

    print("=== Parallel Graph Result ===")
    print(f"Final output: {result.content}")
    print(f"Steps: {result.steps}")
    print(f"Node results: {list(result.node_results.keys())}")


if __name__ == "__main__":
    main()
