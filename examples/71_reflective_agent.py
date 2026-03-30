"""
Example 71: ReflectiveAgent

The actor Agent produces an initial draft. The critic Agent evaluates it and
provides feedback. The actor revises based on the critique. This cycle repeats
until the critic includes the stop_condition word ("approved") or max_reflections
is reached.

Pattern: actor → critic → actor → critic → ... → ReflectiveResult

Run: python examples/71_reflective_agent.py
"""

import asyncio
from typing import List

from selectools import tool
from selectools.agent import Agent, AgentConfig
from selectools.patterns import ReflectiveAgent
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role
from selectools.usage import UsageStats


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
    print("ReflectiveAgent — Example")
    print("=" * 60)

    # Actor improves the draft across rounds; critic approves on round 2
    writer = _make_agent(
        "Initial draft: Our product is good.",  # round 1 draft
        "Revised draft: Our product delivers 3x faster results with zero setup, "  # round 2 draft
        "backed by 2566 tests and a free, self-hosted observability stack.",
    )
    reviewer = _make_agent(
        "The draft is too vague. Please add specific numbers and benefits.",  # round 1 critique
        "This is clear and compelling. Approved — ready to publish.",  # round 2 critique
    )

    agent = ReflectiveAgent(
        actor=writer,
        critic=reviewer,
        max_reflections=3,
        stop_condition="approved",
    )

    # ── Synchronous execution ─────────────────────────────────────────────
    print("\n[sync] Running reflection loop...")
    result = agent.run("Write a one-sentence product pitch for selectools")

    print(f"\nFinal draft: {result.final_draft}")
    print(f"Approved: {result.approved}")
    print(f"Total rounds: {result.total_rounds}")
    for r in result.rounds:
        print(f"\n  Round {r.round_number + 1}:")
        print(f"    Draft:    {r.draft[:100]}")
        print(f"    Critique: {r.critique[:100]}")
        print(f"    Approved: {r.approved}")

    # ── Async execution ───────────────────────────────────────────────────
    async def run_async():
        writer2 = _make_agent(
            "First try.",
            "Better version: selectools ships AI agents in minutes.",
        )
        reviewer2 = _make_agent(
            "Too short, needs more detail.",
            "Perfect. Approved.",
        )
        agent2 = ReflectiveAgent(actor=writer2, critic=reviewer2, max_reflections=3)
        result2 = await agent2.arun("Write a tagline")
        print(f"\n[async] Final draft: {result2.final_draft}")
        print(f"[async] Approved after {result2.total_rounds} round(s): {result2.approved}")

    print("\n[async] Running reflection loop...")
    asyncio.run(run_async())

    print("\n✓ ReflectiveAgent example complete")


if __name__ == "__main__":
    main()
