#!/usr/bin/env python3
"""
Knowledge Memory — Persistent cross-session facts with daily logs.

KnowledgeMemory stores daily log entries and persistent facts in MEMORY.md.
When configured on an agent, a ``remember`` tool is auto-registered and the
build_context() output is injected into the system prompt each turn.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: pip install selectools
Run: python examples/37_knowledge_memory.py
"""

import shutil
import tempfile

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role
from selectools.knowledge import KnowledgeMemory
from selectools.providers.stubs import LocalProvider


def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="selectools_knowledge_")
    km = KnowledgeMemory(directory=tmpdir, recent_days=2, max_context_chars=3000)

    # --- Store some facts directly via the API ---

    print("=== Storing knowledge entries ===\n")
    print(km.remember("User prefers dark mode", category="preference"))
    print(km.remember("Project deadline is 2025-03-21", category="fact", persistent=True))
    print(km.remember("Standup meeting every day at 9 AM", category="schedule", persistent=True))
    print(km.remember("Discussed migration to Python 3.12", category="context"))

    # --- Read back what was stored ---

    print("\n=== Recent daily logs ===")
    print(km.get_recent_logs() or "(empty)")

    print("\n=== Persistent facts (MEMORY.md) ===")
    print(km.get_persistent_facts() or "(empty)")

    # --- Build the context block injected into the system prompt ---

    print("\n=== Context block for prompt injection ===")
    print(km.build_context())

    # --- Wire into an agent: the remember tool is auto-registered ---

    print("\n=== Running agent with knowledge_memory ===")
    agent = Agent(
        tools=[],  # no explicit tools -- remember is auto-added
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=1, knowledge_memory=km),
        memory=ConversationMemory(max_messages=10),
    )

    # Verify the remember tool was auto-registered
    tool_names = [t.name for t in agent.tools]
    print(f"Registered tools: {tool_names}")

    result = agent.run([Message(role=Role.USER, content="Remember that I like tea, not coffee.")])
    print(f"Agent: {result.content}")

    # --- Serialization round-trip ---

    print("\n=== Serialization round-trip ===")
    data = km.to_dict()
    restored = KnowledgeMemory.from_dict(data)
    print(f"Restored KnowledgeMemory at: {restored.directory}")

    # Clean up
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nTemporary knowledge files cleaned up.")


if __name__ == "__main__":
    main()
