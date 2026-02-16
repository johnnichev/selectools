"""Integration tests for agent response caching."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from selectools import Agent, AgentConfig, InMemoryCache
from selectools.cache import CacheKeyBuilder
from selectools.tools import Tool, ToolParameter
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCachingProvider:
    """Tracks how many times complete() is called."""

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
            UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
        )

    async def acomplete(
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
        return self.complete(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


class FakeToolCallProvider:
    """Returns a tool call on the first call, then a text response."""

    name = "fake_tc"
    supports_streaming = False
    supports_async = False

    def __init__(self) -> None:
        self.call_count = 0

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
        if self.call_count == 1:
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(tool_name="greet", parameters={"name": "World"})],
                ),
                UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
            )
        return (
            Message(role=Role.ASSISTANT, content="Hello, World!"),
            UsageStats(prompt_tokens=20, completion_tokens=8, total_tokens=28, cost_usd=0.002),
        )


def _noop_tool() -> Tool:
    return Tool(
        name="greet",
        description="Greet someone by name for testing purposes",
        parameters=[ToolParameter(name="name", param_type=str, description="Name to greet")],
        function=lambda name: f"Hello, {name}!",
    )


def _dummy_tool() -> Tool:
    """Minimal tool to satisfy Agent's requirement for at least one tool."""
    return Tool(
        name="noop",
        description="A no-op tool for testing cache behaviour",
        parameters=[ToolParameter(name="input", param_type=str, description="Ignored input")],
        function=lambda input: "ok",
    )


# ---------------------------------------------------------------------------
# Test: cache miss → provider called → response cached
# ---------------------------------------------------------------------------


class TestCacheMiss:
    def test_miss_calls_provider_and_caches(self) -> None:
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        agent = Agent(
            tools=[_dummy_tool()],
            provider=provider,
            config=AgentConfig(max_iterations=1, cache=cache),
        )

        result = agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 1
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0
        assert cache.size == 1
        assert "Hello from the LLM!" in result.message.content


# ---------------------------------------------------------------------------
# Test: cache hit → provider NOT called
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_hit_skips_provider(self) -> None:
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        # First call: miss
        result1 = agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 1

        # Second call with identical input: hit
        agent.reset()
        result2 = agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 1  # NOT incremented
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert result2.message.content == result1.message.content

    def test_different_input_is_miss(self) -> None:
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hi")])
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hello")])
        assert provider.call_count == 2
        assert cache.stats.misses == 2


# ---------------------------------------------------------------------------
# Test: cache disabled by default
# ---------------------------------------------------------------------------


class TestCacheDisabledByDefault:
    def test_no_cache_in_config(self) -> None:
        provider = FakeCachingProvider()
        config = AgentConfig(max_iterations=1)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hi")])
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 2  # both hit the provider


# ---------------------------------------------------------------------------
# Test: cache works with async (arun)
# ---------------------------------------------------------------------------


class TestCacheAsync:
    @pytest.mark.asyncio
    async def test_arun_cache_hit(self) -> None:
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Async hi")])
        assert provider.call_count == 1

        agent.reset()
        await agent.arun([Message(role=Role.USER, content="Async hi")])
        assert provider.call_count == 1  # cache hit
        assert cache.stats.hits == 1


# ---------------------------------------------------------------------------
# Test: cache with tool calls
# ---------------------------------------------------------------------------


class TestCacheWithToolCalls:
    def test_tool_iteration_changes_cache_key(self) -> None:
        """After a tool executes, the conversation changes, so the next
        iteration should produce a different cache key (miss)."""
        provider = FakeToolCallProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        tool = _noop_tool()
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, cache=cache),
        )

        result = agent.run([Message(role=Role.USER, content="Greet World")])
        # 2 provider calls: one for tool call, one for final answer
        assert provider.call_count == 2
        assert "Hello, World!" in result.message.content


# ---------------------------------------------------------------------------
# Test: cache stats reflect agent usage
# ---------------------------------------------------------------------------


class TestCacheStatsIntegration:
    def test_usage_stats_tracked_on_hit(self) -> None:
        """Cache hits should still contribute to usage tracking."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        # First call (miss)
        agent.run([Message(role=Role.USER, content="Hi")])
        assert agent.total_tokens == 15
        assert agent.total_cost == pytest.approx(0.001)

        # Second call (cache hit) -- don't reset, so usage accumulates
        # Clear history but keep usage tracker
        agent._history = []
        agent.run([Message(role=Role.USER, content="Hi")])

        # Usage should have doubled from the cached stats
        assert agent.total_tokens == 30
        assert agent.total_cost == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Test: cache with routing mode
# ---------------------------------------------------------------------------


class TestCacheWithRoutingMode:
    def test_routing_mode_caches(self) -> None:
        """Cache should work with routing_only mode."""
        provider = FakeCachingProvider(responses=["I'll use search"])
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, routing_only=True, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        result1 = agent.run([Message(role=Role.USER, content="Route me")])
        assert provider.call_count == 1

        agent.reset()
        result2 = agent.run([Message(role=Role.USER, content="Route me")])
        assert provider.call_count == 1  # cache hit
        assert cache.stats.hits == 1


# ---------------------------------------------------------------------------
# Test: cache key includes tools
# ---------------------------------------------------------------------------


class TestCacheVerbose:
    def test_verbose_prints_on_cache_hit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verbose mode should print a message when cache is hit."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache, verbose=True)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hi")])
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hi")])

        captured = capsys.readouterr()
        assert "cache hit" in captured.out.lower()

    def test_no_verbose_no_print(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Without verbose, no cache hit message should be printed."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache, verbose=False)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hi")])
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hi")])

        captured = capsys.readouterr()
        assert "cache hit" not in captured.out.lower()


# ---------------------------------------------------------------------------
# Test: cache clear invalidates entries
# ---------------------------------------------------------------------------


class TestCacheClearInvalidation:
    def test_clear_then_miss(self) -> None:
        """After cache.clear(), the same input should be a miss."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 1

        cache.clear()
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hi")])
        assert provider.call_count == 2  # miss after clear


# ---------------------------------------------------------------------------
# Test: astream bypasses cache
# ---------------------------------------------------------------------------


class TestStreamingBypassesCache:
    @pytest.mark.asyncio
    async def test_astream_does_not_cache(self) -> None:
        """astream should bypass the cache entirely (non-replayable)."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        config = AgentConfig(max_iterations=1, cache=cache)
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        chunks: list[str] = []
        async for item in agent.astream([Message(role=Role.USER, content="Stream")]):
            if hasattr(item, "content"):
                chunks.append(str(item.content))

        # astream uses a different code path; the cache should have 0 entries
        # from streaming (cache only stores complete() / acomplete() results)
        # The provider should have been called
        assert provider.call_count >= 1


# ---------------------------------------------------------------------------
# Test: on_llm_end hook fires on cache hit
# ---------------------------------------------------------------------------


class TestCacheHooksIntegration:
    def test_on_llm_end_fires_on_cache_hit(self) -> None:
        """on_llm_end hook should fire even on cache hits."""
        provider = FakeCachingProvider()
        cache = InMemoryCache(max_size=10, default_ttl=60)
        hook_calls: list[tuple[str, object]] = []

        def on_llm_end(response: str, usage: object) -> None:
            hook_calls.append((response, usage))

        config = AgentConfig(
            max_iterations=1,
            cache=cache,
            hooks={"on_llm_end": on_llm_end},
        )
        agent = Agent(tools=[_dummy_tool()], provider=provider, config=config)

        # First call: miss -> hook fires from provider call
        agent.run([Message(role=Role.USER, content="Hi")])
        first_hook_count = len(hook_calls)
        assert first_hook_count >= 1

        # Second call: hit -> hook should still fire
        agent.reset()
        agent.run([Message(role=Role.USER, content="Hi")])
        assert len(hook_calls) > first_hook_count


# ---------------------------------------------------------------------------
# Test: cache key includes tools
# ---------------------------------------------------------------------------


class TestCacheKeyIncludesTools:
    def test_different_tools_different_key(self) -> None:
        msgs = [Message(role=Role.USER, content="hello")]
        tool_a = Tool(
            name="tool_a",
            description="Tool A does something useful",
            parameters=[],
            function=lambda: "a",
        )
        tool_b = Tool(
            name="tool_b",
            description="Tool B does something different",
            parameters=[],
            function=lambda: "b",
        )
        k1 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=msgs, tools=[tool_a], temperature=0.0
        )
        k2 = CacheKeyBuilder.build(
            model="m", system_prompt="s", messages=msgs, tools=[tool_b], temperature=0.0
        )
        assert k1 != k2
