#!/usr/bin/env python3
"""
Test ConversationMemory with real OpenAI API calls.

This script verifies that ConversationMemory works correctly with OpenAI,
including multi-turn conversations where the LLM can reference previous context.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python scripts/test_memory_with_openai.py
"""

import os
import sys
from pathlib import Path

# Add src to path for local testing
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.models import OpenAI
from selectools.providers.openai_provider import OpenAIProvider


@tool(description="Store a fact for later reference")
def remember_fact(fact: str) -> str:
    """Store a fact that the user wants to remember."""
    return f"I've stored this fact: {fact}"


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # Simulated weather data
    weather_data = {
        "san francisco": "Foggy, 62°F",
        "new york": "Sunny, 45°F",
        "london": "Rainy, 50°F",
        "tokyo": "Clear, 55°F",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def test_basic_memory():
    """Test that memory persists across turns."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Memory Persistence")
    print("=" * 60)

    memory = ConversationMemory(max_messages=20)
    agent = Agent(
        tools=[remember_fact, get_weather],
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),  # Using mini for cost
        memory=memory,
        config=AgentConfig(max_iterations=5, temperature=0.7, verbose=True),
    )

    # Turn 1: Tell the agent something
    print("\n--- Turn 1 ---")
    print("User: My favorite color is blue")
    response1 = agent.run([Message(role=Role.USER, content="My favorite color is blue")])
    print(f"Agent: {response1.content}")
    print(f"Memory size: {len(memory)} messages")

    # Turn 2: Ask the agent to recall
    print("\n--- Turn 2 ---")
    print("User: What's my favorite color?")
    response2 = agent.run([Message(role=Role.USER, content="What's my favorite color?")])
    print(f"Agent: {response2.content}")
    print(f"Memory size: {len(memory)} messages")

    # Verify the agent remembered
    if "blue" in response2.content.lower():
        print("\n✅ PASS: Agent correctly remembered the favorite color!")
        return True
    else:
        print("\n❌ FAIL: Agent did not remember the favorite color")
        print(f"Expected 'blue' in response, got: {response2.content}")
        return False


def test_multi_turn_context():
    """Test that agent maintains context across multiple turns."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Turn Context")
    print("=" * 60)

    memory = ConversationMemory(max_messages=20)
    agent = Agent(
        tools=[remember_fact, get_weather],
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
        memory=memory,
        config=AgentConfig(max_iterations=5, temperature=0.7),
    )

    # Turn 1: Ask about weather in one city
    print("\n--- Turn 1 ---")
    print("User: What's the weather in San Francisco?")
    response1 = agent.run([Message(role=Role.USER, content="What's the weather in San Francisco?")])
    print(f"Agent: {response1.content}")

    # Turn 2: Ask about another city
    print("\n--- Turn 2 ---")
    print("User: How about New York?")
    response2 = agent.run([Message(role=Role.USER, content="How about New York?")])
    print(f"Agent: {response2.content}")

    # Turn 3: Ask to compare (requires context from both previous turns)
    print("\n--- Turn 3 ---")
    print("User: Which city is warmer?")
    response3 = agent.run([Message(role=Role.USER, content="Which city is warmer?")])
    print(f"Agent: {response3.content}")
    print(f"Final memory size: {len(memory)} messages")

    # Verify the agent used context from both cities
    has_sf = "san francisco" in response3.content.lower() or "62" in response3.content
    has_ny = "new york" in response3.content.lower() or "45" in response3.content

    if has_sf and has_ny:
        print("\n✅ PASS: Agent used context from both previous turns!")
        return True
    else:
        print("\n❌ FAIL: Agent did not properly use context from previous turns")
        print(f"Response: {response3.content}")
        return False


def test_memory_without_memory():
    """Test that agent still works without memory (backward compatibility)."""
    print("\n" + "=" * 60)
    print("TEST 3: Backward Compatibility (No Memory)")
    print("=" * 60)

    agent = Agent(
        tools=[get_weather],
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
        # No memory parameter
        config=AgentConfig(max_iterations=5, temperature=0.7),
    )

    print("\n--- Single Turn ---")
    print("User: What's the weather in Tokyo?")
    response = agent.run([Message(role=Role.USER, content="What's the weather in Tokyo?")])
    print(f"Agent: {response.content}")

    if "tokyo" in response.content.lower() or "55" in response.content:
        print("\n✅ PASS: Agent works without memory (backward compatible)!")
        return True
    else:
        print("\n❌ FAIL: Agent did not respond correctly")
        return False


def test_memory_limits():
    """Test that memory respects max_messages limit."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory Limits")
    print("=" * 60)

    memory = ConversationMemory(max_messages=4)  # Very small limit
    agent = Agent(
        tools=[remember_fact],
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
        memory=memory,
        config=AgentConfig(max_iterations=3, temperature=0.7),
    )

    # Add multiple turns to exceed limit
    for i in range(1, 4):
        print(f"\n--- Turn {i} ---")
        msg = f"Message number {i}"
        print(f"User: {msg}")
        response = agent.run([Message(role=Role.USER, content=msg)])
        print(f"Agent: {response.content[:60]}...")
        print(f"Memory size: {len(memory)} messages")

    # Memory should not exceed 4 messages
    if len(memory) <= 4:
        print(f"\n✅ PASS: Memory respected limit (max=4, actual={len(memory)})!")
        return True
    else:
        print(f"\n❌ FAIL: Memory exceeded limit (max=4, actual={len(memory)})")
        return False


def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Testing ConversationMemory with OpenAI")
    print("=" * 60)
    print("\nThis will make real API calls to OpenAI (using gpt-4o-mini for cost)")
    print("Estimated cost: ~$0.01-0.02")

    # Skip prompt if running non-interactively
    if sys.stdin.isatty():
        input("\nPress Enter to continue or Ctrl+C to cancel...")
    else:
        print("\nRunning in non-interactive mode, proceeding automatically...")

    results = []

    try:
        # Run tests
        results.append(("Basic Memory Persistence", test_basic_memory()))
        results.append(("Multi-Turn Context", test_multi_turn_context()))
        results.append(("Backward Compatibility", test_memory_without_memory()))
        results.append(("Memory Limits", test_memory_limits()))

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! ConversationMemory works with OpenAI!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
