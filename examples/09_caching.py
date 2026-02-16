#!/usr/bin/env python3
"""
Response Caching — InMemoryCache (LRU+TTL) and RedisCache for avoiding redundant LLM calls.

Prerequisites: OPENAI_API_KEY (examples 01-05)
    pip install selectools[cache]  # For RedisCache
Run: python examples/09_caching.py
"""

from typing import Any, List, Optional, Tuple

from selectools import Agent, AgentConfig, InMemoryCache, Message, Role
from selectools.cache import CacheKeyBuilder
from selectools.tools import tool
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Fake provider for offline demo (tracks call count for cache verification)
# ---------------------------------------------------------------------------


class FakeCachingProvider:
    """Provider stub that tracks how many times complete() is called."""

    name = "fake"
    supports_streaming = False
    supports_async = True

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self._responses = responses or ["Hello from the LLM!"]
        self._idx = 0
        self.call_count = 0

    def _next_response(self) -> str:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text

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
        self.call_count += 1
        content = self._next_response()
        return (
            Message(role=Role.ASSISTANT, content=content),
            UsageStats(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost_usd=0.001,
                model=model or "fake",
                provider="fake",
            ),
        )

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(description="Get current weather for a location")
def get_weather(location: str) -> str:
    """Simulated weather lookup."""
    return f"Weather in {location}: Sunny, 72°F"


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the caching demo."""
    print("\n" + "#" * 70)
    print("# Response Caching Demo")
    print("#" * 70)

    # --- Step 1: Create a simple tool ---
    print("\n📌 Step 1: Define a weather lookup tool")
    print("   Tool: get_weather(location: str) -> str")
    print("   ✅ Tool defined\n")

    # --- Step 2: Set up InMemoryCache ---
    print("📌 Step 2: Set up InMemoryCache with max_size and TTL")
    cache = InMemoryCache(max_size=100, default_ttl=300)
    print(f"   cache = InMemoryCache(max_size=100, default_ttl=300)")
    print("   ✅ InMemoryCache created\n")

    # --- Step 3: Create agent with cache in AgentConfig ---
    print("📌 Step 3: Create agent with cache in AgentConfig")
    provider = FakeCachingProvider(responses=["The weather in NYC is sunny, 72°F."])
    config = AgentConfig(max_iterations=1, cache=cache)
    agent = Agent(tools=[get_weather], provider=provider, config=config)
    print("   config = AgentConfig(max_iterations=1, cache=cache)")
    print("   agent = Agent(tools=[get_weather], provider=provider, config=config)")
    print("   ✅ Agent created with caching enabled\n")

    # --- Step 4: Run same query twice - cache miss then cache hit ---
    print("📌 Step 4: Run the same query twice - observe cache miss then cache hit")
    query = "What's the weather in NYC?"

    print(f"\n   First call (query: '{query}'):")
    result1 = agent.run([Message(role=Role.USER, content=query)])
    print(f"   → Response: {result1.content[:60]}...")
    print(f"   → Provider calls so far: {provider.call_count}")
    print(f"   → Cache stats: {cache.stats}")
    print("   ✅ Cache MISS (provider was called)\n")

    # --- Step 5: agent.reset() + same query → cache hit ---
    print("📌 Step 5: agent.reset() + same query → cache HIT (cache survives reset)")
    agent.reset()
    print("   agent.reset()  # Clears conversation history, cache is unchanged")

    print(f"\n   Second call (same query: '{query}'):")
    result2 = agent.run([Message(role=Role.USER, content=query)])
    print(f"   → Response: {result2.content[:60]}...")
    print(f"   → Provider calls so far: {provider.call_count}  (unchanged!)")
    print(f"   → Cache stats: {cache.stats}")
    print("   ✅ Cache HIT (provider was NOT called)\n")

    # --- Step 6: Show cache stats ---
    print("📌 Step 6: Cache statistics")
    stats = cache.stats
    print(f"   hits: {stats.hits}")
    print(f"   misses: {stats.misses}")
    print(f"   hit_rate: {stats.hit_rate:.1%}")
    print(f"   evictions: {stats.evictions}")
    print("   ✅ Stats reflect cache behaviour\n")

    # --- Step 7: cache.clear() and demonstrate miss after clear ---
    print("📌 Step 7: cache.clear() and demonstrate miss after clear")
    cache.clear()
    agent.reset()
    print("   cache.clear()")
    print("   agent.reset()")

    print(f"\n   Third call (same query after clear):")
    result3 = agent.run([Message(role=Role.USER, content=query)])
    print(f"   → Response: {result3.content[:60]}...")
    print(f"   → Provider calls so far: {provider.call_count}  (incremented!)")
    print(f"   → Cache stats: {cache.stats}")
    print("   ✅ Cache MISS (cache was cleared)\n")

    # --- Step 8: RedisCache setup (code example) ---
    print("📌 Step 8: RedisCache setup (code example)")
    print(
        """
   # Optional: Use Redis for distributed caching across processes/servers
   # Requires: pip install selectools[cache]

   from selectools.cache_redis import RedisCache

   redis_cache = RedisCache(
       url="redis://localhost:6379/0",
       prefix="selectools:",
       default_ttl=900,
   )
   config = AgentConfig(cache=redis_cache)
   agent = Agent(tools=[...], provider=provider, config=config)
"""
    )
    print("   ✅ RedisCache is optional; use for multi-process deployments\n")

    # --- Step 9: Verbose mode output for cache hits ---
    print("📌 Step 9: Verbose mode shows cache hit messages")
    cache2 = InMemoryCache(max_size=10, default_ttl=60)
    provider2 = FakeCachingProvider(responses=["Cached response."])
    config_verbose = AgentConfig(max_iterations=1, cache=cache2, verbose=True)
    agent_verbose = Agent(tools=[get_weather], provider=provider2, config=config_verbose)

    print("   First run:")
    agent_verbose.run([Message(role=Role.USER, content="Hi")])
    agent_verbose.reset()
    print(
        "   Second run (with verbose=True, expect '[agent] cache hit -- skipping provider call'):"
    )
    agent_verbose.run([Message(role=Role.USER, content="Hi")])
    print("   ✅ Verbose mode prints cache hit when applicable\n")

    # --- CacheKeyBuilder (bonus) ---
    print("📌 Bonus: CacheKeyBuilder")
    print(
        "   Cache keys are SHA-256 hashes of (model, system_prompt, messages, tools, temperature)"
    )
    msgs = [Message(role=Role.USER, content="Hello")]
    key = CacheKeyBuilder.build(
        model="gpt-4o",
        system_prompt="You are helpful.",
        messages=msgs,
        tools=None,
        temperature=0.0,
    )
    print(f"   Example key: {key[:40]}...")
    print("   ✅ Identical requests produce identical keys → cache hit\n")

    print("#" * 70)
    print("# Demo complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
