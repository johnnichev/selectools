#!/usr/bin/env python3
"""
Routing Mode Demo: AgentResult, Custom System Prompt, and agent.reset().

Demonstrates routing_only mode where the agent selects a tool WITHOUT executing it.
Useful for intent classification, tool routing, and structured decision-making.
Shows AgentResult fields (tool_name, tool_args, iterations) and agent.reset()
for handling multiple independent requests.

Requirements:
    pip install selectools

Run:
    python examples/routing_mode_demo.py
"""

from typing import Any, List, Optional, Tuple

from selectools import Agent, AgentConfig, Message, Role
from selectools.tools import tool
from selectools.types import AgentResult, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Fake provider that returns tool calls for routing demonstrations
# ---------------------------------------------------------------------------


class RoutingMockProvider:
    """Provider that returns predetermined tool selections (no LLM call)."""

    name = "routing-mock"
    supports_streaming = False
    supports_async = True

    def __init__(
        self,
        tool_to_call: str,
        args: dict,
    ) -> None:
        self.tool_to_call = tool_to_call
        self.args = args

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        return (
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name=self.tool_to_call,
                        parameters=self.args,
                        id="call_1",
                    )
                ],
            ),
            UsageStats(0, 0, 0, 0.0, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class RoutingMockProviderMultiple:
    """Provider that returns multiple tool calls in one response."""

    name = "routing-mock-multi"
    supports_streaming = False
    supports_async = True

    def __init__(self, tool_calls: List[ToolCall]) -> None:
        self.tool_calls = tool_calls

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        return (
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=self.tool_calls,
            ),
            UsageStats(0, 0, 0, 0.0, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


class TextOnlyProvider:
    """Provider that returns plain text (no tool call)."""

    name = "text-only"
    supports_streaming = False
    supports_async = True

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(
        self,
        *,
        model: str = "",
        system_prompt: str = "",
        messages: Optional[List[Message]] = None,
        tools: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: Any = None,
    ) -> Tuple[Message, UsageStats]:
        return (
            Message(role=Role.ASSISTANT, content=self.text),
            UsageStats(0, 0, 0, 0.0, "mock", "mock"),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Send an email to a recipient")
def send_email(to: str, subject: str, body: str) -> str:
    """Send email (not executed in routing mode)."""
    return f"Email sent to {to}"


@tool(description="Schedule a meeting with attendees")
def schedule_meeting(
    title: str,
    attendees: str,
    datetime: str,
) -> str:
    """Schedule meeting (not executed in routing mode)."""
    return f"Meeting '{title}' scheduled"


@tool(description="Search knowledge base for information")
def search_knowledge(query: str) -> str:
    """Search knowledge base (not executed in routing mode)."""
    return f"Results for: {query}"


@tool(description="Create a support ticket")
def create_ticket(
    title: str,
    description: str,
    priority: str = "medium",
) -> str:
    """Create support ticket (not executed in routing mode)."""
    return f"Ticket created: {title}"


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the routing mode demo."""
    print("\n" + "#" * 70)
    print("# Routing Mode Demo")
    print("#" * 70)

    tools = [send_email, schedule_meeting, search_knowledge, create_ticket]

    # --- Step 1: Define tools ---
    print(
        "\n📌 Step 1: Define tools (send_email, schedule_meeting, search_knowledge, create_ticket)"
    )
    for t in tools:
        print(f"   - {t.name}: {t.description[:50]}...")
    print("   ✅ Tools defined\n")

    # --- Step 2: Create agent with routing_only=True and custom system_prompt ---
    print("📌 Step 2: Create agent with routing_only=True and custom system_prompt")
    custom_prompt = (
        "You are an intent router. Given user input, select the most appropriate "
        "tool. Available: send_email, schedule_meeting, search_knowledge, create_ticket."
    )
    config = AgentConfig(
        routing_only=True,
        system_prompt=custom_prompt,
        max_iterations=1,
    )
    print("   config = AgentConfig(routing_only=True, system_prompt=custom_prompt)")
    print("   ✅ Agent configured for routing (no tool execution)\n")

    # --- Step 3: Send different intents - agent selects tool WITHOUT executing ---
    print("📌 Step 3: Different user intents → agent selects tool WITHOUT executing")

    # Intent: send email
    provider_email = RoutingMockProvider(
        "send_email",
        {"to": "alice@example.com", "subject": "Hi", "body": "Hello!"},
    )
    agent = Agent(tools=tools, provider=provider_email, config=config)
    result = agent.run([Message(role=Role.USER, content="Send an email to alice@example.com")])

    print("\n   Intent: 'Send an email to alice@example.com'")
    print(f"   → Selected tool: {result.tool_name}")
    print(f"   → Tool args: {result.tool_args}")
    print(f"   → Tool was NOT executed (routing_only=True)")
    print("   ✅ Routing works: tool selected, not run\n")

    # --- Step 4: Inspect AgentResult fields ---
    print("📌 Step 4: Inspect AgentResult fields")
    assert isinstance(result, AgentResult)
    print(f"   result.message: {type(result.message).__name__}")
    print(f"   result.tool_name: {result.tool_name}")
    print(f"   result.tool_args: {result.tool_args}")
    print(f"   result.iterations: {result.iterations}")
    print(f"   result.tool_calls: {[tc.tool_name for tc in result.tool_calls]}")
    print("   ✅ AgentResult provides structured metadata\n")

    # --- Step 5: Use routing for intent classification ---
    print("📌 Step 5: Use routing for intent classification")

    intents = [
        ("I need to schedule a team sync for tomorrow 2pm", "schedule_meeting"),
        ("Search for Python tutorials", "search_knowledge"),
        ("Create a bug ticket for login failure", "create_ticket"),
    ]

    for user_msg, expected_tool in intents:
        provider = RoutingMockProvider(expected_tool, {"query": user_msg})
        agent = Agent(tools=tools, provider=provider, config=config)
        result = agent.run([Message(role=Role.USER, content=user_msg)])
        print(f"   '{user_msg[:40]}...'")
        print(f"   → Classified as: {result.tool_name}")
        print()

    print("   ✅ Routing enables intent classification without execution\n")

    # --- Step 6: agent.reset() for multiple independent requests ---
    print("📌 Step 6: agent.reset() pattern for multiple independent requests")

    provider = RoutingMockProvider("search_knowledge", {"query": "docs"})
    agent = Agent(tools=tools, provider=provider, config=config)

    result1 = agent.run([Message(role=Role.USER, content="Request 1: search docs")])
    print(f"   Request 1 → tool: {result1.tool_name}")

    agent.reset()
    result2 = agent.run([Message(role=Role.USER, content="Request 2: search docs")])
    print(f"   Request 2 (after reset) → tool: {result2.tool_name}")

    print("   agent.reset() clears conversation history between requests")
    print("   ✅ Use reset() for stateless request handling\n")

    # --- Step 7: AgentResult with normal (non-routing) mode for comparison ---
    print("📌 Step 7: AgentResult with normal (non-routing) mode")

    # Provider that returns tool call first, then final text (for 2-iteration run)
    class TwoPhaseProvider(RoutingMockProvider):
        def __init__(self) -> None:
            super().__init__("search_knowledge", {"query": "Python"})
            self._call_count = 0

        def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
            self._call_count += 1
            if self._call_count == 1:
                return super().complete(**kwargs)
            return (
                Message(role=Role.ASSISTANT, content="Here are the Python docs."),
                UsageStats(0, 0, 0, 0.0, "mock", "mock"),
            )

    config_normal = AgentConfig(routing_only=False, max_iterations=2)
    provider_normal = TwoPhaseProvider()
    agent_normal = Agent(tools=tools, provider=provider_normal, config=config_normal)

    result_normal = agent_normal.run([Message(role=Role.USER, content="Search for Python")])
    print(f"   routing_only=False: tools execute")
    print(f"   result.tool_name: {result_normal.tool_name}")
    print(f"   result.tool_calls: {len(result_normal.tool_calls)} call(s)")
    print("   ✅ In normal mode, tools run and result includes execution metadata\n")

    # Text-only response (no tool call)
    print("   Text-only response (no tool selected):")
    provider_text = TextOnlyProvider("Just chatting, no tool needed.")
    agent_text = Agent(tools=tools, provider=provider_text, config=config)
    result_text = agent_text.run([Message(role=Role.USER, content="Hello")])
    print(f"   result.tool_name: {result_text.tool_name}")
    print(f"   result.content: {result_text.content[:50]}...")
    print("   ✅ When no tool is selected, tool_name is None\n")

    print("#" * 70)
    print("# Demo complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
