#!/usr/bin/env python3
"""
Summarize-on-Trim — Automatically summarize old messages when memory is trimmed.

When the conversation exceeds max_messages, the oldest messages are removed.
With summarize_on_trim=True the agent asks an LLM to condense them into a
short summary that is prepended as context to future turns, preserving key
facts without consuming message slots.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: None
    pip install selectools

Run:
    python examples/34_summarize_on_trim.py
"""

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role
from selectools.providers.stubs import LocalProvider


def main() -> None:
    # Small memory window so trimming happens quickly
    memory = ConversationMemory(max_messages=4)

    agent = Agent(
        tools=[],
        provider=LocalProvider(),
        config=AgentConfig(
            max_iterations=1,
            summarize_on_trim=True,
            # LocalProvider is used for both chat and summarization
        ),
        memory=memory,
    )

    prompts = [
        "My name is Alice and I work at Acme Corp.",
        "I prefer dark mode and Python 3.12.",
        "My project deadline is next Friday.",
        "Remind me about the standup at 9 AM.",
    ]

    for i, text in enumerate(prompts, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {text}")
        result = agent.run([Message(role=Role.USER, content=text)])
        print(f"Agent: {result.content}")
        print(f"Messages in memory: {len(memory)}")

        if memory.summary:
            print(f"Running summary: {memory.summary}")
        print()

    # After several turns the oldest messages are gone but the summary keeps context
    print("=== Final memory state ===")
    print(f"Messages retained: {len(memory)}")
    print(f"Summary: {memory.summary or '(none yet -- increase turns to trigger trimming)'}")

    print("\n=== Retained messages ===")
    for msg in memory.get_history():
        role = msg.role.value.upper()
        preview = msg.content[:70] + "..." if len(msg.content) > 70 else msg.content
        print(f"  {role}: {preview}")


if __name__ == "__main__":
    main()
