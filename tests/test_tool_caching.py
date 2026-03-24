"""Tests for tool result caching (@tool(cacheable=True)) — v0.17.6."""

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.cache import InMemoryCache
from selectools.tools.base import Tool, ToolParameter
from selectools.tools.decorators import tool
from selectools.trace import StepType
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats


def _usage():
    return UsageStats(
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="test",
        provider="test",
    )


def _resp(text):
    return (Message(role=Role.ASSISTANT, content=text), _usage())


def _tool_resp(tool_name, params=None):
    return (
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(tool_name=tool_name, parameters=params or {})],
        ),
        _usage(),
    )


# ── Tool definition tests ────────────────────────────────────────────


class TestToolCacheableFlag:
    def test_default_not_cacheable(self):
        t = Tool(name="t", description="t", parameters=[], function=lambda: "ok")
        assert t.cacheable is False
        assert t.cache_ttl == 300

    def test_cacheable_via_constructor(self):
        t = Tool(
            name="t",
            description="t",
            parameters=[],
            function=lambda: "ok",
            cacheable=True,
            cache_ttl=60,
        )
        assert t.cacheable is True
        assert t.cache_ttl == 60

    def test_cacheable_via_decorator(self):
        @tool(description="test", cacheable=True, cache_ttl=120)
        def my_tool(x: str) -> str:
            return x

        assert my_tool.cacheable is True
        assert my_tool.cache_ttl == 120

    def test_decorator_default_not_cacheable(self):
        @tool(description="test")
        def my_tool(x: str) -> str:
            return x

        assert my_tool.cacheable is False


# ── Cache key tests ──────────────────────────────────────────────────


class TestToolCacheKey:
    def test_same_params_same_key(self):
        from selectools.agent._tool_executor import _ToolExecutorMixin

        key1 = _ToolExecutorMixin._build_tool_cache_key("search", {"q": "hello", "limit": 10})
        key2 = _ToolExecutorMixin._build_tool_cache_key("search", {"limit": 10, "q": "hello"})
        assert key1 == key2

    def test_different_params_different_key(self):
        from selectools.agent._tool_executor import _ToolExecutorMixin

        key1 = _ToolExecutorMixin._build_tool_cache_key("search", {"q": "hello"})
        key2 = _ToolExecutorMixin._build_tool_cache_key("search", {"q": "world"})
        assert key1 != key2

    def test_different_tools_different_key(self):
        from selectools.agent._tool_executor import _ToolExecutorMixin

        key1 = _ToolExecutorMixin._build_tool_cache_key("search", {"q": "hello"})
        key2 = _ToolExecutorMixin._build_tool_cache_key("lookup", {"q": "hello"})
        assert key1 != key2

    def test_key_starts_with_tool_result_prefix(self):
        from selectools.agent._tool_executor import _ToolExecutorMixin

        key = _ToolExecutorMixin._build_tool_cache_key("search", {"q": "hi"})
        assert key.startswith("tool_result:search:")


# ── Agent integration tests ──────────────────────────────────────────


class TestToolCachingIntegration:
    def test_cacheable_tool_cached_on_second_call(self, fake_provider):
        """Same tool + same args on second iteration returns cached result."""
        call_count = {"n": 0}

        def counting_fn() -> str:
            call_count["n"] += 1
            return f"result-{call_count['n']}"

        cacheable_tool = Tool(
            name="counter",
            description="count",
            parameters=[],
            function=counting_fn,
            cacheable=True,
            cache_ttl=300,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("counter"),
                _tool_resp("counter"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        result = agent.run("go")
        assert result.content == "done"
        assert call_count["n"] == 1  # only executed once, second was cached

    def test_non_cacheable_tool_always_executes(self, fake_provider):
        """Non-cacheable tool executes every time."""
        call_count = {"n": 0}

        def counting_fn() -> str:
            call_count["n"] += 1
            return f"result-{call_count['n']}"

        regular_tool = Tool(
            name="counter",
            description="count",
            parameters=[],
            function=counting_fn,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("counter"),
                _tool_resp("counter"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[regular_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        agent.run("go")
        assert call_count["n"] == 2

    def test_different_args_cache_miss(self, fake_provider):
        """Same cacheable tool with different args = cache miss."""
        call_count = {"n": 0}

        def search_fn(q: str) -> str:
            call_count["n"] += 1
            return f"results for {q}"

        search_tool = Tool(
            name="search",
            description="search",
            parameters=[ToolParameter(name="q", param_type=str, description="query")],
            function=search_fn,
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("search", {"q": "hello"}),
                _tool_resp("search", {"q": "world"}),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[search_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        agent.run("go")
        assert call_count["n"] == 2  # different args = two executions

    def test_no_cache_configured_always_executes(self, fake_provider):
        """Cacheable tool without a cache on the agent always executes."""
        call_count = {"n": 0}

        def counting_fn() -> str:
            call_count["n"] += 1
            return "ok"

        cacheable_tool = Tool(
            name="counter",
            description="count",
            parameters=[],
            function=counting_fn,
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("counter"),
                _tool_resp("counter"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(max_iterations=5),  # no cache
        )
        agent.run("go")
        assert call_count["n"] == 2

    def test_cache_hit_records_trace_step(self, fake_provider):
        """Cache hit produces a CACHE_HIT trace step."""
        cacheable_tool = Tool(
            name="noop",
            description="noop",
            parameters=[],
            function=lambda: "ok",
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("noop"),
                _tool_resp("noop"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        result = agent.run("go")

        cache_steps = [s for s in result.trace.steps if s.type == StepType.CACHE_HIT]
        assert len(cache_steps) >= 1
        assert cache_steps[0].tool_name == "noop"
        assert "cached" in cache_steps[0].summary

    def test_first_call_records_tool_execution_step(self, fake_provider):
        """First call (cache miss) records a TOOL_EXECUTION trace step."""
        cacheable_tool = Tool(
            name="noop",
            description="noop",
            parameters=[],
            function=lambda: "ok",
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("noop"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        result = agent.run("go")

        exec_steps = [s for s in result.trace.steps if s.type == StepType.TOOL_EXECUTION]
        assert len(exec_steps) == 1
        assert exec_steps[0].tool_name == "noop"

    def test_cached_result_content_matches(self, fake_provider):
        """Cached result returns the exact same string."""
        cacheable_tool = Tool(
            name="noop",
            description="noop",
            parameters=[],
            function=lambda: "exact-value-42",
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("noop"),
                _tool_resp("noop"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        agent.run("go")

        # Verify the tool result message in history has the cached value
        tool_msgs = [m for m in agent._history if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "exact-value-42"
        assert tool_msgs[1].content == "exact-value-42"


class TestToolCachingAsync:
    @pytest.mark.asyncio
    async def test_async_cacheable_tool_cached(self, fake_provider):
        """Async path also caches tool results."""
        call_count = {"n": 0}

        def counting_fn() -> str:
            call_count["n"] += 1
            return f"result-{call_count['n']}"

        cacheable_tool = Tool(
            name="counter",
            description="count",
            parameters=[],
            function=counting_fn,
            cacheable=True,
        )

        provider = fake_provider(
            responses=[
                _tool_resp("counter"),
                _tool_resp("counter"),
                _resp("done"),
            ]
        )
        agent = Agent(
            tools=[cacheable_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=5),
        )
        result = await agent.arun("go")
        assert result.content == "done"
        assert call_count["n"] == 1


class TestToolCachingBackwardCompat:
    def test_existing_tests_unaffected(self, fake_provider):
        """Tool without cacheable flag works exactly as before."""
        regular_tool = Tool(name="t", description="t", parameters=[], function=lambda: "ok")
        provider = fake_provider(responses=[_tool_resp("t"), _resp("done")])
        agent = Agent(
            tools=[regular_tool],
            provider=provider,
            config=AgentConfig(cache=InMemoryCache(), max_iterations=3),
        )
        result = agent.run("go")
        assert result.content == "done"
