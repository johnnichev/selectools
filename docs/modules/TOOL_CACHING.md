# Tool Result Caching

**Added in:** v0.17.6
**Files:** `src/selectools/tools/base.py`, `src/selectools/tools/decorators.py`, `src/selectools/agent/_tool_executor.py`

## Overview

Tool result caching avoids re-executing expensive tools (API calls, database queries) when the same tool is called with identical arguments. Mark a tool as `cacheable=True` and the agent's cache handles the rest.

## Quick Start

```python
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

# Second call with same args: served from cache (no execution)
result2 = agent.run("Search for Python tutorials again")
```

## How It Works

1. When a cacheable tool is called, the executor builds a cache key from `tool_name + sha256(sorted(params))`
2. If the key exists in the agent's cache and hasn't expired, the cached result is returned immediately
3. If it's a miss, the tool executes normally and the result is stored with the tool's `cache_ttl`
4. Cache hits record a `StepType.CACHE_HIT` trace step (no `on_tool_start`/`on_tool_end` events fire)

## API Reference

### @tool() decorator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cacheable` | `bool` | `False` | Enable result caching for this tool |
| `cache_ttl` | `int` | `300` | Cache TTL in seconds (5 minutes default) |

### Tool constructor

```python
tool = Tool(
    name="search",
    description="Search the web",
    parameters=[...],
    function=search_fn,
    cacheable=True,
    cache_ttl=120,
)
```

## Requirements

Tool result caching requires a cache on the agent:

```python
config = AgentConfig(cache=InMemoryCache())
# or
config = AgentConfig(cache=RedisCache(url="redis://localhost"))
```

Without a cache configured, `cacheable=True` tools execute normally every time.

## Trace Visibility

Cache hits appear in `result.trace`:

```python
for step in result.trace.steps:
    if step.type == StepType.CACHE_HIT:
        print(f"Cached: {step.tool_name} → {step.tool_result}")
```

!!! warning
    Streaming tools (`streaming=True`) are not cached — their results are generated progressively and cannot be replayed from cache.

## See Also

- [Tools](TOOLS.md) — tool definition and decorator
- [Agent](AGENT.md) — cache configuration
