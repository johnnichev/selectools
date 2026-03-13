#!/usr/bin/env python3
"""
Entity Memory — Extract and track named entities across conversation turns.

EntityMemory merges entities into a deduplicated registry and builds a context
block for the system prompt. This example manually feeds entities to demonstrate
the registry offline without a real LLM call.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: pip install selectools
Run: python examples/35_entity_memory.py
"""

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role
from selectools.entity_memory import Entity, EntityMemory
from selectools.providers.stubs import LocalProvider


def main() -> None:
    provider = LocalProvider()
    entity_mem = EntityMemory(provider=provider, max_entities=20)

    # --- Simulate Turn 1: user mentions people and a company ---

    print("=== Turn 1: Introduce entities ===")
    turn1_entities = [
        Entity(name="Alice", entity_type="person", attributes={"role": "engineer"}),
        Entity(name="Acme Corp", entity_type="organization", attributes={"industry": "tech"}),
    ]
    entity_mem.update(turn1_entities)

    for e in entity_mem.entities:
        print(f"  {e.name} [{e.entity_type}] mentions={e.mention_count} attrs={e.attributes}")

    # --- Simulate Turn 2: mention Alice again and add a technology ---

    print("\n=== Turn 2: Re-mention Alice, add Python ===")
    turn2_entities = [
        Entity(name="Alice", entity_type="person", attributes={"team": "backend"}),
        Entity(name="Python 3.12", entity_type="technology", attributes={"use": "scripting"}),
    ]
    entity_mem.update(turn2_entities)

    for e in entity_mem.entities:
        print(f"  {e.name} [{e.entity_type}] mentions={e.mention_count} attrs={e.attributes}")

    # --- Build and display context ---

    print("\n=== Context block for system prompt ===")
    context = entity_mem.build_context()
    print(context)

    # --- Wire it into an agent via AgentConfig ---

    print("\n=== Running agent with entity_memory ===")
    agent = Agent(
        tools=[],
        provider=provider,
        config=AgentConfig(max_iterations=1, entity_memory=entity_mem),
        memory=ConversationMemory(max_messages=10),
    )
    result = agent.run([Message(role=Role.USER, content="What do you know about Alice?")])
    print(f"Agent: {result.content}")

    # --- Serialization round-trip ---

    print("\n=== Serialization round-trip ===")
    data = entity_mem.to_dict()
    restored = EntityMemory.from_dict(data, provider=provider)
    print(f"Restored {len(restored.entities)} entities")
    for e in restored.entities:
        print(f"  {e.name} [{e.entity_type}]")


if __name__ == "__main__":
    main()
