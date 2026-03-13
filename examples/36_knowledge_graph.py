#!/usr/bin/env python3
"""
Knowledge Graph Memory — Track relationship triples across conversation turns.

KnowledgeGraphMemory stores subject-relation-object triples in a TripleStore.
Relevant triples are queried each turn and injected into the system prompt.
This example manually adds triples to demonstrate the graph offline.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: pip install selectools
Run: python examples/36_knowledge_graph.py
"""

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role
from selectools.knowledge_graph import InMemoryTripleStore, KnowledgeGraphMemory, Triple
from selectools.providers.stubs import LocalProvider


def main() -> None:
    provider = LocalProvider()
    store = InMemoryTripleStore(max_triples=100)
    kg = KnowledgeGraphMemory(provider=provider, storage=store, max_context_triples=10)

    print("=== Adding triples to the knowledge graph ===\n")
    triples = [
        Triple(subject="Alice", relation="works_at", object="Acme Corp"),
        Triple(subject="Alice", relation="knows", object="Bob"),
        Triple(subject="Bob", relation="manages", object="DataPipeline"),
        Triple(subject="Acme Corp", relation="uses", object="Python"),
        Triple(subject="DataPipeline", relation="written_in", object="Python"),
        Triple(subject="Alice", relation="prefers", object="dark mode", confidence=0.9),
    ]
    store.add_many(triples)
    print(f"Graph contains {store.count()} triples\n")

    # --- Query for triples relevant to a topic ---

    print("=== Query: 'Alice' ===")
    alice_triples = kg.query_relevant("Tell me about Alice")
    for t in alice_triples:
        print(f"  {t.subject} --[{t.relation}]--> {t.object}")

    print("\n=== Query: 'Python' ===")
    python_triples = kg.query_relevant("What uses Python?")
    for t in python_triples:
        print(f"  {t.subject} --[{t.relation}]--> {t.object}")

    # --- Build context for the system prompt ---

    print("\n=== Context block (all triples) ===")
    print(kg.build_context())

    print("\n=== Context block (query-filtered: 'Bob') ===")
    print(kg.build_context(query="Bob"))

    # --- Wire it into an agent ---

    print("\n=== Running agent with knowledge_graph ===")
    agent = Agent(
        tools=[],
        provider=provider,
        config=AgentConfig(max_iterations=1, knowledge_graph=kg),
        memory=ConversationMemory(max_messages=10),
    )
    result = agent.run(
        [Message(role=Role.USER, content="What is Alice's relationship with Acme Corp?")]
    )
    print(f"Agent: {result.content}")

    # --- Serialization round-trip ---

    print("\n=== Serialization round-trip ===")
    data = kg.to_dict()
    restored = KnowledgeGraphMemory.from_dict(data, provider=provider)
    print(f"Restored graph has {restored.store.count()} triples")


if __name__ == "__main__":
    main()
