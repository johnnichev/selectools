"""
Example 58: Human-in-the-loop with AgentGraph

Demonstrates generator nodes with yield InterruptRequest:
- Graph pauses at reviewer node
- Human provides approval decision
- Graph resumes from exact yield point (no double execution!)

Uses LocalProvider so no API keys are needed.
"""

from selectools import Agent, AgentConfig
from selectools.orchestration import (
    STATE_KEY_LAST_OUTPUT,
    AgentGraph,
    GraphState,
    InMemoryCheckpointStore,
    InterruptRequest,
)
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def draft_content(topic: str) -> str:
    """Draft content on a topic."""
    return f"Draft: {topic} — [content here]"


def make_agent(responses: list) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=responses),
        tools=[draft_content],
    )


async def review_node(state: GraphState):
    """Generator node that pauses for human approval."""
    # Expensive work done before interrupt — stored in state, not re-done on resume
    if "analysis" not in state.data:
        draft = state.data.get(STATE_KEY_LAST_OUTPUT, "")
        state.data["analysis"] = (
            f"Analysis: draft is {len(draft)} chars, looks {'good' if len(draft) > 20 else 'short'}"
        )

    print(f"\n[HITL] Analysis ready: {state.data['analysis']}")
    print("[HITL] Graph pausing for human input...")

    # Yield an interrupt — execution pauses here
    decision = yield InterruptRequest(
        prompt="Do you approve this draft? (yes/no)",
        payload={
            "draft": state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            "analysis": state.data["analysis"],
        },
    )

    # Continues here after graph.resume() — decision contains human's response
    state.data["approved"] = decision == "yes"
    state.data[STATE_KEY_LAST_OUTPUT] = (
        f"Review complete: {'approved' if decision == 'yes' else 'rejected'}"
    )
    print(f"[HITL] Decision received: {decision}")


def main():
    drafter = make_agent(["Draft article: AI safety is crucial for future development..."])
    publisher = make_agent(["Published! Article is now live on the blog."])

    graph = AgentGraph(name="hitl_pipeline")
    graph.add_node("drafter", drafter)
    graph.add_node("reviewer", review_node)  # generator node
    graph.add_node("publisher", publisher)

    graph.add_edge("drafter", "reviewer")
    graph.add_conditional_edge(
        "reviewer",
        lambda state: "publisher" if state.data.get("approved") else AgentGraph.END,
        path_map={"publisher": "publisher"},
    )
    graph.add_edge("publisher", AgentGraph.END)
    graph.set_entry("drafter")

    # Use a checkpoint store (required for HITL)
    store = InMemoryCheckpointStore()

    print("=== First Run (pauses at reviewer) ===")
    result = graph.run("Write a blog post about AI safety", checkpoint_store=store)

    if result.interrupted:
        print(f"Graph paused! interrupt_id: {result.interrupt_id}")
        print(f"Payload: {result.state.data.get('analysis', '')}")

        # Simulate human decision
        human_decision = "yes"
        print(f"\nHuman decided: {human_decision!r}")

        print("\n=== Resuming Graph ===")
        final = graph.resume(result.interrupt_id, human_decision, checkpoint_store=store)
        print(f"Final output: {final.content}")
        print(f"Approved: {final.state.data.get('approved')}")
    else:
        print(f"Completed without interrupt: {result.content}")


if __name__ == "__main__":
    main()
