#!/usr/bin/env python3
"""
Persistent Sessions — Save and restore conversation memory across agent instances.

Demonstrates JsonFileSessionStore: the agent's conversation history is persisted
to disk and restored when a new agent is created with the same session_id.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: pip install selectools
Run: python examples/33_persistent_sessions.py
"""

import shutil
import tempfile

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.providers.stubs import LocalProvider
from selectools.sessions import JsonFileSessionStore


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    weather = {"paris": "18C, sunny", "london": "12C, cloudy", "tokyo": "25C, humid"}
    return weather.get(city.lower(), f"No data for {city}")


def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="selectools_sessions_")
    store = JsonFileSessionStore(directory=tmpdir)
    session_id = "demo-session"

    print("=== Session 1: First conversation ===\n")
    memory1 = ConversationMemory(max_messages=20)
    agent1 = Agent(
        tools=[get_weather],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, session_store=store, session_id=session_id),
        memory=memory1,
    )
    result1 = agent1.run([Message(role=Role.USER, content="What is the weather in Paris?")])
    print(f"Agent: {result1.content}")
    print(f"Memory has {len(memory1)} messages")
    print(f"Session saved: {store.exists(session_id)}\n")

    # --- Simulate a restart by creating a brand-new agent ---
    print("=== Session 2: New agent, same session_id ===\n")
    restored_memory = store.load(session_id)
    print(f"Restored memory has {len(restored_memory)} messages from previous session")

    agent2 = Agent(
        tools=[get_weather],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, session_store=store, session_id=session_id),
        memory=restored_memory,
    )
    result2 = agent2.run([Message(role=Role.USER, content="Now check London.")])
    print(f"Agent: {result2.content}")
    print(f"Memory now has {len(restored_memory)} messages (includes both sessions)\n")

    print("=== Stored sessions ===")
    for meta in store.list():
        print(f"  id={meta.session_id}  messages={meta.message_count}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nTemporary session files cleaned up.")


if __name__ == "__main__":
    main()
