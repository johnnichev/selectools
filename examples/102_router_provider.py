"""
RouterProvider — cost-optimized model routing across provider tiers.

A RouterProvider wraps multiple providers organized cheapest -> priciest and
routes each request to a tier based on a deterministic, rule-based
complexity classification (input size, tool count, code blocks, reasoning
keywords, multi-part questions, structured-output requests).

Strategies:
- cost_optimized: simple -> cheapest, moderate -> middle, complex -> top
- quality_first: always start at the top tier, degrade down on failure
- balanced: never below the middle tier; complex goes to the top

This example is fully offline: it uses stub providers so no API keys are
needed. Swap the stubs for OpenAIProvider/AnthropicProvider in real usage.

Run: python examples/102_router_provider.py
"""

from typing import Any, List, Tuple

from selectools.providers import RouterConfig, RouterProvider
from selectools.providers.router import classify_complexity
from selectools.types import Message, Role
from selectools.usage import UsageStats


class StubProvider:
    """Offline stand-in for a real provider tier."""

    supports_streaming = True
    supports_async = False

    def __init__(self, name: str, default_model: str) -> None:
        self.name = name
        self.default_model = default_model

    def complete(self, *, model: str, **kwargs: Any) -> Tuple[Message, UsageStats]:
        msg = Message(role=Role.ASSISTANT, content=f"[{self.name} answered with {model}]")
        usage = UsageStats(0, 0, 0, 0.0, model, self.name)
        return msg, usage

    def stream(self, *, model: str, **kwargs: Any) -> Any:
        yield f"[{self.name} streaming with {model}]"


def main() -> None:
    # Tiers use real registry model ids so RouterProvider can verify the
    # cheapest-first ordering against selectools' pricing tables.
    router = RouterProvider(
        providers={
            "fast": StubProvider("fast-tier", "gpt-5.4-nano"),  # $0.10/1M input
            "smart": StubProvider("smart-tier", "claude-sonnet-4-6"),  # $3/1M input
            "power": StubProvider("power-tier", "gpt-5.4-pro"),  # $30/1M input
        },
        strategy="cost_optimized",
        config=RouterConfig(),  # thresholds are configurable
        on_route=lambda complexity, tier: print(f"  routed: {complexity} -> {tier}"),
    )
    print(f"Resolved tier order (cheapest first): {router.tier_order}")

    prompts: List[str] = [
        "What is the capital of France?",
        "What changed in v2? And why did latency regress? Output as JSON.",
        "Analyze this function step by step and explain why it deadlocks:\n"
        "```python\nlock.acquire(); lock.acquire()\n```\n"
        "1. Identify the bug\n2. Refactor it",
    ]

    print("\nClassification preview:")
    for p in prompts:
        cls = classify_complexity(p, total_tokens=len(p) // 4, tool_count=0)
        print(f"  {cls:8s} <- {p.splitlines()[0][:60]}")

    print("\nRouting requests:")
    for p in prompts:
        msg, usage = router.complete(
            model="agent-default",  # overridden by each tier's own model
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content=p)],
        )
        print(f"  tier_used={router.tier_used!r} -> {msg.content}")

    # quality_first: everything starts at the top tier
    quality = RouterProvider(
        providers={
            "fast": StubProvider("fast-tier", "gpt-5.4-nano"),
            "power": StubProvider("power-tier", "gpt-5.4-pro"),
        },
        strategy="quality_first",
    )
    msg, _ = quality.complete(
        model="agent-default",
        system_prompt="",
        messages=[Message(role=Role.USER, content="What is 2+2?")],
    )
    print(f"\nquality_first: tier_used={quality.tier_used!r} -> {msg.content}")


if __name__ == "__main__":
    main()
