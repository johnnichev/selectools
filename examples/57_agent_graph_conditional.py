"""
Example 57: Conditional routing with AgentGraph

Demonstrates conditional edges with path_map validation:
  drafter → router → (revise | publish) → END

Uses LocalProvider so no API keys are needed.
"""

from selectools import Agent, AgentConfig
from selectools.orchestration import STATE_KEY_LAST_OUTPUT, AgentGraph, GraphState
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def evaluate_quality(text: str) -> str:
    """Evaluate the quality of text."""
    return "quality: high" if len(text) > 20 else "quality: low"


def make_agent(responses: list) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=responses),
        tools=[evaluate_quality],
    )


def quality_router(state: GraphState) -> str:
    """Route based on draft quality."""
    last_output = state.data.get(STATE_KEY_LAST_OUTPUT, "")
    if "needs revision" in last_output.lower() or len(last_output) < 50:
        return "revise"
    return "publish"


def main():
    drafter = make_agent(["Draft: This is a well-crafted article about AI safety..."])
    reviser = make_agent(["Revised draft with improvements applied..."])
    publisher = make_agent(["Published! Article is now live."])

    graph = AgentGraph(name="review_pipeline")

    graph.add_node("drafter", drafter)
    graph.add_node("revise", reviser)
    graph.add_node("publish", publisher)

    graph.add_conditional_edge(
        "drafter",
        quality_router,
        path_map={"revise": "revise", "publish": "publish"},
    )
    graph.add_edge("revise", "publish")
    graph.add_edge("publish", AgentGraph.END)
    graph.set_entry("drafter")

    result = graph.run("Write and publish an article")

    print("=== Conditional Graph Result ===")
    print(f"Output: {result.content}")
    print(f"Steps: {result.steps}")

    # Visualize the graph
    print("\n=== Mermaid Diagram ===")
    print(graph.to_mermaid())


if __name__ == "__main__":
    main()
