"""
Observability Hooks — lifecycle callbacks and tool validation at registration time.

Prerequisites: OPENAI_API_KEY (examples 01-05)
Run: python examples/12_observability_hooks.py
"""

import time
from typing import Any, Dict

from selectools import (
    Agent,
    AgentConfig,
    Message,
    Role,
    Tool,
    ToolParameter,
    ToolValidationError,
    tool,
)

try:
    from selectools.models import OpenAI
    from selectools.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
    print(f"Using OpenAI provider ({OpenAI.GPT_4O_MINI.id})")
except Exception:
    from selectools.providers.stubs import LocalProvider

    provider = LocalProvider()
    print("Using LocalProvider (no API calls)")


# =============================================================================
# Feature 1: Tool Validation at Registration
# =============================================================================


def demo_tool_validation() -> None:
    """Demonstrate tool validation catching errors at registration time."""
    print("\n" + "=" * 70)
    print("FEATURE 1: TOOL VALIDATION AT REGISTRATION")
    print("=" * 70)

    print("\n✅ Example 1: Valid tool definition")
    try:

        @tool(description="Calculate the sum of two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        print(f"   Tool '{add.name}' registered successfully!")
        print(f"   Parameters: {[p.name for p in add.parameters]}")
    except ToolValidationError as e:
        print(f"   Error: {e}")

    print("\n❌ Example 2: Empty tool name (caught at registration)")
    try:
        Tool(
            name="",  # Empty name!
            description="A tool",
            parameters=[],
            function=lambda: "result",
        )
        print("   Tool registered (this shouldn't happen!)")
    except ToolValidationError as e:
        print("   ✓ Validation caught the error:")
        print(f"     {str(e).split(chr(10))[1:4]}")  # Show first few lines

    print("\n❌ Example 3: Duplicate parameter names (caught at registration)")
    try:
        Tool(
            name="bad_tool",
            description="A tool with duplicate params",
            parameters=[
                ToolParameter(name="value", param_type=int, description="First value"),
                ToolParameter(
                    name="value", param_type=int, description="Second value"
                ),  # Duplicate!
            ],
            function=lambda value: str(value),
        )
        print("   Tool registered (this shouldn't happen!)")
    except ToolValidationError as e:
        print("   ✓ Validation caught the error:")
        print(f"     Issue: Duplicate parameter name(s): 'value'")

    print("\n❌ Example 4: Parameter not in function signature (caught at registration)")
    try:
        Tool(
            name="mismatched_tool",
            description="A tool with mismatched params",
            parameters=[
                ToolParameter(
                    name="nonexistent_param", param_type=str, description="Doesn't exist"
                ),
            ],
            function=lambda: "result",  # No parameters!
        )
        print("   Tool registered (this shouldn't happen!)")
    except ToolValidationError as e:
        print("   ✓ Validation caught the error:")
        print(f"     Issue: Parameter 'nonexistent_param' not found in function signature")

    print("\n💡 Benefits:")
    print("   - Errors caught during development, not production")
    print("   - Clear error messages with suggestions")
    print("   - No runtime surprises when agent tries to use the tool")


# =============================================================================
# Feature 2: Observability Hooks
# =============================================================================


def demo_observability_hooks() -> None:
    """Demonstrate observability hooks for monitoring and debugging."""
    print("\n" + "=" * 70)
    print("FEATURE 2: OBSERVABILITY HOOKS")
    print("=" * 70)

    # Create some demo tools

    @tool(description="Search for information")
    def search(query: str) -> str:
        time.sleep(0.1)  # Simulate work
        return f"Found results for: {query}"

    @tool(description="Calculate a mathematical expression")
    def calculate(expression: str) -> str:
        time.sleep(0.05)  # Simulate work
        return f"Result: {eval(expression)}"  # Don't do this in production!

    # Set up hooks for monitoring
    metrics: Dict[str, Any] = {
        "agent_starts": 0,
        "iterations": 0,
        "tool_calls": [],
        "llm_calls": 0,
        "total_tokens": 0,
    }

    def on_agent_start(messages: Any) -> None:
        metrics["agent_starts"] += 1
        print(f"\n🚀 Agent started with {len(messages)} message(s)")

    def on_iteration_start(iteration: Any, messages: Any) -> None:
        metrics["iterations"] = iteration
        print(f"\n🔄 Iteration {iteration} starting...")

    def on_tool_start(tool_name: str, tool_args: Any) -> None:
        print(f"   🔧 Calling tool: {tool_name}")
        print(f"      Args: {tool_args}")

    def on_tool_end(tool_name: str, result: str, duration: float) -> None:
        metrics["tool_calls"].append({"name": tool_name, "duration": duration})
        print(f"   ✅ Tool completed: {tool_name}")
        print(f"      Duration: {duration:.3f}s")
        print(f"      Result preview: {result[:50]}...")

    def on_llm_start(messages: Any, model: str) -> None:
        metrics["llm_calls"] += 1
        print(f"   🤖 LLM call #{metrics['llm_calls']} to {model}")

    def on_llm_end(response: Any, usage: Any) -> None:
        if usage:
            metrics["total_tokens"] += usage.total_tokens
            print(f"   📊 Tokens: {usage.total_tokens} (${usage.cost_usd:.6f})")

    def on_agent_end(response: Any, usage: Any) -> None:
        print(f"\n✨ Agent finished!")
        print(f"   Final response length: {len(response.content)} characters")
        print(f"\n📊 Session Metrics:")
        print(f"   - Total iterations: {metrics['iterations']}")
        print(f"   - Total LLM calls: {metrics['llm_calls']}")
        print(f"   - Total tool calls: {len(metrics['tool_calls'])}")
        print(f"   - Total tokens used: {metrics['total_tokens']}")
        print(f"   - Total cost: ${usage.total_cost_usd:.6f}")
        if metrics["tool_calls"]:
            print(f"\n   Tool breakdown:")
            for call in metrics["tool_calls"]:
                print(f"     - {call['name']}: {call['duration']:.3f}s")

    # Create agent with hooks
    agent = Agent(
        tools=[search, calculate],
        provider=provider,
        config=AgentConfig(
            max_iterations=5,
            hooks={
                "on_agent_start": on_agent_start,
                "on_agent_end": on_agent_end,
                "on_iteration_start": on_iteration_start,
                "on_tool_start": on_tool_start,
                "on_tool_end": on_tool_end,
                "on_llm_start": on_llm_start,
                "on_llm_end": on_llm_end,
            },
        ),
    )

    # Run agent
    print("\n📝 User query: 'What is 2+2 and search for Python tutorials'")
    response = agent.run(
        [Message(role=Role.USER, content="What is 2+2 and search for Python tutorials")]
    )

    print("\n💡 Benefits:")
    print("   - Real-time monitoring of agent behavior")
    print("   - Performance tracking (tool execution times)")
    print("   - Cost tracking per session")
    print("   - Easy integration with logging/monitoring systems")
    print("   - Debug production issues without changing code")


# =============================================================================
# Combined Example: Production-Ready Agent
# =============================================================================


def demo_production_ready() -> None:
    """Demonstrate using both features for a production-ready agent."""
    print("\n" + "=" * 70)
    print("COMBINED: PRODUCTION-READY AGENT")
    print("=" * 70)

    print("\n✅ Defining tools with validation...")

    @tool(description="Process customer feedback")
    def process_feedback(feedback: str, sentiment: str) -> str:
        """
        Process customer feedback with sentiment analysis.
        Note: All parameters are validated at registration time!
        """
        return f"Processed feedback (sentiment: {sentiment}): {feedback[:50]}..."

    print(f"   ✓ Tool '{process_feedback.name}' validated and registered")

    print("\n✅ Setting up observability hooks...")
    logs = []

    def log_hook(event: str, *args: Any) -> None:
        logs.append({"event": event, "timestamp": time.time(), "args": args})

    print("   ✓ Hooks configured for: tool calls, errors, and completion")

    agent = Agent(
        tools=[process_feedback],
        provider=provider,
        config=AgentConfig(
            max_iterations=3,
            hooks={
                "on_tool_start": lambda name, args: log_hook("tool_start", name, args),
                "on_tool_end": lambda name, result, dur: log_hook("tool_end", name, dur),
                "on_tool_error": lambda name, error, args: log_hook("tool_error", name, str(error)),
                "on_agent_end": lambda resp, usage: log_hook("agent_end", usage.total_cost_usd),
            },
        ),
    )

    print(f"\n✨ Agent ready! {len(logs)} events logged so far")
    print("\n💡 This agent is production-ready:")
    print("   ✓ Tools validated at startup (fail fast)")
    print("   ✓ All actions monitored and logged")
    print("   ✓ Errors tracked with context")
    print("   ✓ Cost and performance metrics collected")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print(" Selectools v0.5.2 - Tool Validation & Observability Hooks Demo")
    print("=" * 70)

    # Demo 1: Tool Validation
    demo_tool_validation()

    # Demo 2: Observability Hooks
    demo_observability_hooks()

    # Demo 3: Combined (Production-Ready)
    demo_production_ready()

    print("\n" + "=" * 70)
    print("Demo complete! Both features work together to create production-ready agents.")
    print("=" * 70)
