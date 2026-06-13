#!/usr/bin/env python3
"""
Unified memory via config — MemoryConfig(unified=True) (beta, v1.1).

UnifiedMemory shipped standalone in v0.24.0 (see 106_unified_memory.py).
This example wires it into an Agent through AgentConfig: the agent injects
context assembled from the long-term, entity, and episodic tiers before each
call, and persists every completed turn via UnifiedMemory.add_turn() — which
records the episode and auto-promotes aging-out short-term items whose
importance clears the threshold (rule-based scoring, no extra LLM calls).

No API key needed. Runs entirely offline with LocalProvider.

Prerequisites: pip install selectools
Run: python examples/112_unified_memory_config.py
"""

from selectools import Agent, AgentConfig, MemoryConfig, Message, Role, tool
from selectools.providers.stubs import LocalProvider


@tool(description="Look up a fact")
def lookup(query: str) -> str:
    return f"The answer to '{query}' is 42."


def main() -> None:
    agent = Agent(
        tools=[lookup],
        provider=LocalProvider(),
        config=AgentConfig(
            max_iterations=1,
            memory=MemoryConfig(
                unified=True,
                importance_threshold=0.7,
                short_term_limit=2,  # tiny window so promotion triggers quickly
                long_term_limit=1000,
                episodic_retention_days=30,
                auto_promote=True,
            ),
        ),
    )
    unified = agent.unified_memory
    assert unified is not None

    print("=== Turn 1: an identity fact (importance 0.9) ===")
    agent.run([Message(role=Role.USER, content="My name is Alice Smith")])
    print(unified)

    print("\n=== Turn 2: mundane — turn 1 ages out and is promoted ===")
    agent.run([Message(role=Role.USER, content="What should I eat for lunch?")])
    for entry in unified.long_term.store.query(limit=10):
        print(f"  promoted [{entry.category}] importance={entry.importance:.2f}  {entry.content}")

    print("\n=== Turn 3: promoted knowledge is injected back as context ===")
    agent.run([Message(role=Role.USER, content="Do you know my name?")])
    context = unified.assemble_context(max_tokens=4000, include_conversation=False)
    print(context)

    print("\n=== Federated recall across tiers ===")
    for result in unified.recall("Alice name", limit=3):
        print(f"  {result.score:.2f} [{result.source}] {result.content}")


if __name__ == "__main__":
    main()
