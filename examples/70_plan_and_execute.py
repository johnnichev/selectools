"""
Example 70: PlanAndExecuteAgent

The planner Agent generates a JSON execution plan. Executor agents handle
each step in sequence. Results are aggregated into a final output.

Pattern: planner → [executor_0, executor_1, ...] → aggregated result

Run: python examples/70_plan_and_execute.py
"""

import asyncio
from typing import List

from selectools import tool
from selectools.agent import Agent, AgentConfig
from selectools.patterns import PlanAndExecuteAgent, PlanStep
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.usage import UsageStats

# ── Mock setup (no API keys needed) ──────────────────────────────────────────
# In production replace _ScriptedProvider with OpenAIProvider / AnthropicProvider

_RESEARCH_PLAN = '[{"executor": "researcher", "task": "research vector databases"}, {"executor": "writer", "task": "write the blog post"}, {"executor": "reviewer", "task": "review and improve"}]'


@tool(description="Placeholder tool — not called in this example")
def _noop(x: str) -> str:
    return x


class _ScriptedProvider(LocalProvider):
    """Returns scripted responses in order, cycling if exhausted."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._index = 0

    def complete(self, *, model, system_prompt, messages, **kwargs):  # type: ignore[override]
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return Message(role=Role.ASSISTANT, content=text), UsageStats()


def _make_agent(*responses: str) -> Agent:
    """Create an agent that returns scripted responses in order."""
    return Agent(tools=[_noop], provider=_ScriptedProvider(list(responses)), config=AgentConfig())


def main():
    print("=" * 60)
    print("PlanAndExecuteAgent — Example")
    print("=" * 60)

    # Planner returns a JSON execution plan
    planner = _make_agent(_RESEARCH_PLAN)

    # Executor agents handle individual steps
    researcher = _make_agent(
        "Research: Vector databases store high-dimensional embeddings for similarity search. Key players: Pinecone, Qdrant, Weaviate, FAISS."
    )
    writer = _make_agent(
        "Draft: Vector databases are specialized storage systems optimized for embedding similarity search, enabling efficient retrieval in RAG pipelines."
    )
    reviewer = _make_agent(
        "Review complete. The draft is clear, accurate, and well-structured. Approved."
    )

    agent = PlanAndExecuteAgent(
        planner=planner,
        executors={
            "researcher": researcher,
            "writer": writer,
            "reviewer": reviewer,
        },
    )

    # ── Synchronous execution ─────────────────────────────────────────────
    print("\n[sync] Running plan-and-execute...")
    result = agent.run("Write a technical blog post about vector databases")
    print(f"Final output: {result.content[:200]}")
    print(f"Plan stored: {result.state.data.get('__plan__')}")

    # ── Async execution ───────────────────────────────────────────────────
    async def run_async():
        planner2 = _make_agent(_RESEARCH_PLAN)
        researcher2 = _make_agent("Research findings: ...")
        writer2 = _make_agent("Draft: ...")
        reviewer2 = _make_agent("Approved.")

        agent2 = PlanAndExecuteAgent(
            planner=planner2,
            executors={"researcher": researcher2, "writer": writer2, "reviewer": reviewer2},
        )
        result = await agent2.arun("Write a blog post about embeddings")
        print(f"\n[async] Final output: {result.content[:200]}")

    print("\n[async] Running plan-and-execute...")
    asyncio.run(run_async())

    print("\n✓ PlanAndExecuteAgent example complete")


if __name__ == "__main__":
    main()
