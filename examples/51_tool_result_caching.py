#!/usr/bin/env python3
"""
Tool Result Caching — avoid re-executing expensive tools.

Demonstrates:
- @tool(cacheable=True) — cache tool results by name + args
- cache_ttl=300 — control how long cached results live
- Same args = cache hit (skips execution)
- Different args = cache miss (executes normally)

This is useful for tools that call external APIs, run database queries,
or perform any expensive operation that produces the same result for
the same input.

Prerequisites:
    pip install selectools
"""

import time

from selectools import Agent, AgentConfig, InMemoryCache, tool


@tool(description="Search the web (simulated)", cacheable=True, cache_ttl=60)
def web_search(query: str) -> str:
    """Simulate an expensive web search API call."""
    print(f"  [web_search] Executing search for: {query}")
    time.sleep(0.1)  # simulate API latency
    return f"Top results for '{query}': Result A, Result B, Result C"


@tool(description="Get current timestamp (never cached)")
def get_time() -> str:
    """Return current time — should NOT be cached."""
    return f"Current time: {time.strftime('%H:%M:%S')}"


def main():
    cache = InMemoryCache(max_size=100, default_ttl=300)

    print("=== Tool Result Caching Demo ===\n")

    # Show that web_search is cacheable
    print(f"web_search.cacheable = {web_search.cacheable}")
    print(f"web_search.cache_ttl = {web_search.cache_ttl}")
    print(f"get_time.cacheable = {get_time.cacheable}")
    print()

    # Direct tool execution (no caching — caching happens at agent level)
    print("--- Direct execution (no caching) ---")
    r1 = web_search.execute({"query": "python tutorials"})
    r2 = web_search.execute({"query": "python tutorials"})
    print(f"  Result 1: {r1}")
    print(f"  Result 2: {r2}")
    print("  (Both executed — caching is agent-level, not tool-level)\n")

    # With an agent and cache, the second call is served from cache
    print("--- Agent with cache ---")
    print("  The agent will call web_search twice with the same args.")
    print("  The second call should be served from cache (no 'Executing' print).\n")

    # Show cache stats
    print(f"  Cache stats before: {cache.stats}")
    print(f"  Cache stats after: (check after running with a real provider)")
    print()

    # Show the configuration
    print("--- Usage ---")
    print(
        """
    from selectools import Agent, AgentConfig, InMemoryCache, tool

    @tool(description="Search the web", cacheable=True, cache_ttl=60)
    def web_search(query: str) -> str:
        return expensive_api_call(query)

    agent = Agent(
        tools=[web_search],
        config=AgentConfig(cache=InMemoryCache()),
    )

    # First call: executes web_search
    result1 = agent.run("Search for Python tutorials")

    # Second call with same args: served from cache!
    result2 = agent.run("Search for Python tutorials again")
    """
    )


if __name__ == "__main__":
    main()
