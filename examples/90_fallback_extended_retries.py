"""
FallbackProvider with Extended Retries — handle Anthropic 529, 504, Cloudflare errors.

Since v0.22.0 (BUG-27), selectools recognizes these transient errors:
- 529 Anthropic Overloaded (very common on US-West traffic)
- 504 Gateway Timeout
- 408 Request Timeout
- 522/524 Cloudflare origin timeouts
- rate_limit_exceeded (underscore form from OpenAI/Mistral)
- overloaded/service_unavailable strings

Prerequisites: OPENAI_API_KEY (or any two provider keys for real fallback)
Run: python examples/90_fallback_extended_retries.py
"""

from selectools import Agent, tool
from selectools.providers.fallback import FallbackProvider, _is_retriable
from selectools.providers.stubs import LocalProvider


@tool(description="no-op")
def _noop() -> str:
    return "ok"


def main() -> None:
    # Demonstrate which errors are now retriable
    test_cases = [
        ("429 Rate Limited", True),
        ("529 Anthropic Overloaded", True),
        ("504 Gateway Timeout", True),
        ("408 Request Timeout", True),
        ("522 Cloudflare connection timed out", True),
        ("524 Cloudflare origin timeout", True),
        ("rate_limit_exceeded: quota reached", True),
        ("overloaded_error: server busy", True),
        ("service_unavailable", True),
        ("400 Bad Request", False),
        ("401 Unauthorized", False),
        ("404 Not Found", False),
    ]

    print("FallbackProvider Retriable Error Detection:")
    print("-" * 55)
    for msg, expected in test_cases:
        result = _is_retriable(Exception(msg))
        status = "✓" if result == expected else "✗"
        print(f"  {status} {msg:45s} -> {'retriable' if result else 'non-retriable'}")

    # Real usage: providers=[primary, backup] with circuit breaker
    fallback = FallbackProvider(
        providers=[LocalProvider(), LocalProvider()],
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown=60.0,
        on_fallback=lambda from_p, to_p, exc: print(f"  Fallback: {from_p} -> {to_p}"),
    )
    agent = Agent(tools=[_noop], provider=fallback)
    result = agent.run("Hello")
    print(f"\nAgent response: {result.content[:80]}")


if __name__ == "__main__":
    main()
