"""
Cost Tracking — token counting, cost estimation, and usage summaries.

Prerequisites: OPENAI_API_KEY (examples 01-03)
Run: python examples/05_cost_tracking.py
"""

import selectools
from selectools import Agent, AgentConfig, Message, Role, tool

# Define some example tools


@tool(description="Search the web for information")
def web_search(query: str) -> str:
    """Simulate a web search."""
    return f"Search results for '{query}': Found 10 results about {query}."


@tool(description="Calculate mathematical expressions")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # nosec B307 - example only
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool(description="Get current weather for a location")
def get_weather(location: str) -> str:
    """Get weather information."""
    return f"Weather in {location}: Sunny, 72°F"


def main() -> None:
    """Run the cost tracking demo."""
    print("=" * 60)
    print("Cost Tracking Demo")
    print("=" * 60)

    # Create agent with cost warning threshold
    from selectools.models import OpenAI

    config = AgentConfig(
        model=OpenAI.GPT_4O_MINI.id,  # Use cheaper model for demo
        verbose=True,  # Show token counts in real-time
        cost_warning_threshold=0.01,  # Warn if cost exceeds $0.01
        max_iterations=10,
    )

    agent = Agent(
        tools=[web_search, calculator, get_weather],
        config=config,
    )

    # Example 1: Simple query
    print("\n" + "=" * 60)
    print("Example 1: Simple Query")
    print("=" * 60)

    response = agent.run(
        [Message(role=Role.USER, content="What's 25 * 4 and what's the weather in San Francisco?")]
    )

    print(f"\nAgent Response: {response.content}")
    print(f"\nTotal Cost: ${agent.total_cost:.6f}")
    print(f"Total Tokens: {agent.total_tokens:,}")

    # Example 2: Multiple turns (reusing agent)
    print("\n" + "=" * 60)
    print("Example 2: Multiple Turns")
    print("=" * 60)

    response2 = agent.run(
        [Message(role=Role.USER, content="Search for 'Python programming tutorials'")]
    )

    print(f"\nAgent Response: {response2.content}")

    # Show cumulative usage
    print(agent.get_usage_summary())

    # Example 3: Per-tool breakdown
    print("\n" + "=" * 60)
    print("Example 3: Per-Tool Usage Breakdown")
    print("=" * 60)

    usage_dict = agent.usage.to_dict()
    print(f"Total Iterations: {usage_dict['iterations']}")
    print(f"Total Tokens: {usage_dict['total_tokens']:,}")
    print(f"Total Cost: ${usage_dict['total_cost_usd']:.6f}")
    print("\nTool Usage:")
    for tool_name, count in usage_dict["tool_usage"].items():
        tokens = usage_dict["tool_tokens"][tool_name]
        print(f"  - {tool_name}: {count} calls, {tokens:,} tokens")

    # Example 4: Reset usage for new conversation
    print("\n" + "=" * 60)
    print("Example 4: Reset Usage")
    print("=" * 60)

    print("Resetting usage stats...")
    agent.reset_usage()

    print(f"Total Cost after reset: ${agent.total_cost:.6f}")
    print(f"Total Tokens after reset: {agent.total_tokens:,}")

    # Example 5: Cost warning threshold
    print("\n" + "=" * 60)
    print("Example 5: Cost Warning Threshold")
    print("=" * 60)

    # Make multiple calls to trigger warning
    for i in range(3):
        agent.run(
            [Message(role=Role.USER, content=f"Search for 'topic {i}' and calculate {i} * {i}")]
        )

    print(f"\nFinal Total Cost: ${agent.total_cost:.6f}")
    print("(Warning should have been printed if threshold exceeded)")


if __name__ == "__main__":
    # Note: This demo requires OPENAI_API_KEY environment variable
    # Set it before running: export OPENAI_API_KEY='your-key-here'
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-key-here'")
