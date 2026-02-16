#!/usr/bin/env python3
"""
Multi-turn Conversation Memory with automatic context preservation.

Prerequisites: OPENAI_API_KEY (examples 01-03)
Run: python examples/04_conversation_memory.py
"""

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.models import OpenAI
from selectools.providers.openai_provider import OpenAIProvider


@tool(description="Get information about a topic")
def get_info(topic: str) -> str:
    """Simulated information retrieval."""
    info_db = {
        "python": "Python is a high-level programming language known for readability.",
        "selectools": "Selectools is a lightweight tool-calling library for AI agents.",
        "memory": "Memory in AI agents helps maintain context across conversations.",
    }
    return info_db.get(topic.lower(), f"No information found about {topic}")


@tool(description="Remember a fact for later")
def remember_fact(fact: str) -> str:
    """Simulated fact storage."""
    return f"I'll remember that: {fact}"


def main() -> None:
    # Create a conversation memory with a limit of 20 messages
    memory = ConversationMemory(max_messages=20)

    # Create an agent with memory
    agent = Agent(
        tools=[get_info, remember_fact],
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
        config=AgentConfig(max_iterations=5, temperature=0.7),
        memory=memory,  # Pass memory to agent
    )

    print("=== Multi-Turn Conversation Demo ===\n")

    # Turn 1: Ask about Python
    print("Turn 1: User asks about Python")
    response1 = agent.run([Message(role=Role.USER, content="Tell me about Python")])
    print(f"Agent: {response1.content}\n")
    print(f"Memory now has {len(memory)} messages\n")

    # Turn 2: Follow-up question (memory maintains context)
    print("Turn 2: Follow-up question")
    response2 = agent.run([Message(role=Role.USER, content="What about Selectools?")])
    print(f"Agent: {response2.content}\n")
    print(f"Memory now has {len(memory)} messages\n")

    # Turn 3: Reference previous conversation
    print("Turn 3: Reference previous context")
    response3 = agent.run(
        [Message(role=Role.USER, content="Can you compare the two things we just discussed?")]
    )
    print(f"Agent: {response3.content}\n")
    print(f"Memory now has {len(memory)} messages\n")

    # Show full conversation history
    print("=== Full Conversation History ===")
    for i, msg in enumerate(memory.get_history(), 1):
        role_name = msg.role.value.upper()
        content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"{i}. {role_name}: {content_preview}")

    # Demonstrate memory serialization
    print("\n=== Memory Serialization ===")
    memory_dict = memory.to_dict()
    print(f"Max messages: {memory_dict['max_messages']}")
    print(f"Current count: {memory_dict['message_count']}")
    print(f"Max tokens: {memory_dict['max_tokens']}")


if __name__ == "__main__":
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    main()
