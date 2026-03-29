"""
Example 72: DebateAgent

Multiple agents argue opposing positions over max_rounds rounds. Each agent
sees the prior round transcript so they can respond to each other. A judge
Agent synthesizes a final conclusion after all rounds complete.

Pattern: [agent_a, agent_b, ...] → rounds → judge → DebateResult

Run: python examples/72_debate_agent.py
"""

import asyncio
from typing import List

from selectools import tool
from selectools.agent import Agent, AgentConfig
from selectools.patterns import DebateAgent
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
    print("DebateAgent — Example")
    print("=" * 60)

    # Two debaters argue over 2 rounds; judge delivers a verdict
    optimist = _make_agent(
        # Round 1: opening argument
        "AI will eliminate tedious work and free humans for creative pursuits. "
        "History shows that automation creates more jobs than it destroys.",
        # Round 2: rebuttal
        "The transition will be managed through reskilling programs and new industries. "
        "AI-augmented workers already outperform both humans and AI alone.",
    )
    skeptic = _make_agent(
        # Round 1: opening argument
        "The pace of AI adoption outstrips society's ability to retrain workers. "
        "White-collar jobs are now at risk in ways past automation never threatened.",
        # Round 2: rebuttal
        "Reskilling takes years and not everyone can pivot. The winners will be capital owners, "
        "not displaced workers. We need policy guardrails now, not optimism.",
    )
    judge = _make_agent(
        # Called once after all rounds
        "Both sides raise valid points. AI will displace certain roles but also create new ones. "
        "The key variable is the *speed* of transition — policy must bridge the gap. "
        "Conclusion: cautious optimism with proactive labour policy.",
    )

    agent = DebateAgent(
        agents={"optimist": optimist, "skeptic": skeptic},
        judge=judge,
        max_rounds=2,
    )

    # ── Synchronous execution ─────────────────────────────────────────────
    print("\n[sync] Running debate...")
    result = agent.run("Will AI cause widespread unemployment?")

    print(f"\nConclusion: {result.conclusion}")
    print(f"Total rounds: {result.total_rounds}")
    for r in result.rounds:
        print(f"\n  Round {r.round_number + 1}:")
        for name, arg in r.arguments.items():
            print(f"    {name}: {arg[:100]}")

    # ── Async execution ───────────────────────────────────────────────────
    async def run_async():
        pro = _make_agent(
            "Remote work boosts productivity and expands talent pools.",
            "Asynchronous communication actually improves documentation culture.",
        )
        con = _make_agent(
            "Collaboration suffers and junior employees miss mentorship opportunities.",
            "Zoom fatigue and isolation harm mental health and long-term retention.",
        )
        arbiter = _make_agent(
            "Hybrid models offer the best of both worlds. Teams should define their own norms.",
        )
        agent2 = DebateAgent(
            agents={"pro": pro, "con": con},
            judge=arbiter,
            max_rounds=2,
        )
        result2 = await agent2.arun("Should companies mandate return to office?")
        print(f"\n[async] Conclusion: {result2.conclusion}")
        print(f"[async] Rounds completed: {result2.total_rounds}")

    print("\n[async] Running debate...")
    asyncio.run(run_async())

    print("\n✓ DebateAgent example complete")


if __name__ == "__main__":
    main()
