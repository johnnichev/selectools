# MCP Integration

**Added in:** v0.17.1

Connect to any MCP-compatible tool server and expose selectools tools as MCP servers. Requires `pip install selectools[mcp]`.

---

## Quick Start — Use MCP Tools

```python
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider
from selectools.mcp import mcp_tools, MCPServerConfig

# Connect to an MCP server and get tools
with mcp_tools(MCPServerConfig(command="python", args=["server.py"])) as tools:
    agent = Agent(
        provider=OpenAIProvider(),
        config=AgentConfig(model="gpt-4.1-mini"),
        tools=tools,
    )
    result = agent.run("Search for Python tutorials")
```

MCP tools are regular selectools `Tool` objects. All existing features work automatically: traces, observers, guardrails, policies, evals, cost tracking.

---

## MCPServerConfig

```python
from selectools.mcp import MCPServerConfig

# stdio transport (local subprocess)
config = MCPServerConfig(
    command="python",
    args=["my_server.py"],
    name="search",
)

# Streamable HTTP transport (remote)
config = MCPServerConfig(
    url="http://api.example.com/mcp",
    transport="streamable-http",
    headers={"Authorization": "Bearer token"},
    name="api",
)
```

| Field | Default | Description |
|---|---|---|
| `name` | Auto-generated | Human-readable server name |
| `transport` | `"stdio"` | `"stdio"` or `"streamable-http"` |
| `command` | | Command for stdio (e.g., `"python"`) |
| `args` | `[]` | Command arguments |
| `url` | | URL for HTTP transport |
| `headers` | `None` | HTTP headers (auth, etc.) |
| `timeout` | `30.0` | Connection/call timeout (seconds) |
| `max_retries` | `2` | Retries on transport failure |
| `auto_reconnect` | `True` | Auto-reconnect on failure |
| `circuit_breaker_threshold` | `3` | Failures before circuit opens |
| `circuit_breaker_cooldown` | `60.0` | Seconds before retry after circuit opens |
| `screen_output` | `True` | Screen outputs for prompt injection |
| `cache_tools` | `True` | Cache tool list after first fetch |

---

## MCPClient

Direct client for advanced use cases.

```python
from selectools.mcp import MCPClient, MCPServerConfig

# Async context manager (preferred)
async with MCPClient(config) as client:
    tools = await client.list_tools()
    result = await client._call_tool("search", {"query": "python"})

# Sync context manager
with MCPClient(config) as client:
    tools = client.list_tools_sync()
```

### Circuit Breaker

If an MCP server fails repeatedly, the circuit breaker opens and tool calls fail immediately instead of waiting for timeouts:

```python
config = MCPServerConfig(
    command="python",
    args=["unreliable_server.py"],
    circuit_breaker_threshold=3,     # Open after 3 failures
    circuit_breaker_cooldown=60.0,   # Retry after 60 seconds
)
```

### Retry with Backoff

```python
config = MCPServerConfig(
    command="python",
    args=["server.py"],
    max_retries=3,          # Retry up to 3 times
    retry_backoff=1.0,      # 1s, 2s, 4s exponential backoff
)
```

---

## MultiMCPClient

Connect to multiple MCP servers simultaneously.

```python
from selectools.mcp import MultiMCPClient, MCPServerConfig

async with MultiMCPClient([
    MCPServerConfig(command="python", args=["search.py"], name="search"),
    MCPServerConfig(url="http://api.example.com/mcp",
                    transport="streamable-http", name="api"),
]) as client:
    tools = await client.list_all_tools()
    # Tools are prefixed: search_web_search, api_query, etc.
    agent = Agent(provider=p, tools=tools, config=c)
```

### Graceful Degradation

If one server fails to connect, the others still work:

```python
async with MultiMCPClient(configs) as client:
    print(f"Active: {client.active_servers}")   # ["search"]
    print(f"Failed: {client.failed_servers}")   # ["api"]
    tools = await client.list_all_tools()       # Only search tools
```

### Name Prefixing

Tool names are prefixed with the server name to avoid collisions:

```python
MultiMCPClient(configs, prefix_tools=True)   # search_query, api_fetch
MultiMCPClient(configs, prefix_tools=False)  # Raises ValueError on collision
```

---

## MCPServer — Expose Tools

Turn any selectools `@tool` function into an MCP server:

```python
from selectools import tool
from selectools.mcp import MCPServer

@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"72°F in {city}"

@tool(description="Search documents")
def search(query: str) -> str:
    return f"Results for: {query}"

server = MCPServer(tools=[get_weather, search])
server.serve(transport="stdio")
# or: server.serve(transport="streamable-http", port=8080)
```

This server can be used by Claude Desktop, Cursor, VS Code, or other selectools agents.

---

## With Agent — Full Example

```python
import asyncio
from selectools import Agent, AgentConfig, tool
from selectools.providers import AnthropicProvider
from selectools.mcp import MCPClient, MCPServerConfig

@tool(description="Local calculator")
def multiply(a: int, b: int) -> str:
    return str(a * b)

async def main():
    config = MCPServerConfig(command="python", args=["math_server.py"])

    async with MCPClient(config) as client:
        mcp_tools = await client.list_tools()

        # Mix local + MCP tools
        agent = Agent(
            provider=AnthropicProvider(),
            config=AgentConfig(model="claude-haiku-4-5"),
            tools=[multiply] + mcp_tools,
        )

        # Agent automatically selects the right tool
        result = await agent.arun("Add 5 and 3")      # Uses MCP 'add' tool
        result2 = await agent.arun("Multiply 6 by 7")  # Uses local 'multiply'

asyncio.run(main())
```

---

## With Eval Framework

Evaluate MCP-powered agents like any other:

```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(
    agent=agent,  # Agent with MCP tools
    cases=[
        TestCase(input="Add 10 and 20", expect_tool="add"),
        TestCase(input="Search for Python", expect_tool="search"),
    ],
)
report = suite.run()
print(report.accuracy)
```

---

## API Reference

| Symbol | Description |
|---|---|
| `MCPServerConfig(...)` | Server connection configuration |
| `MCPClient(config)` | Single-server client |
| `MultiMCPClient(configs)` | Multi-server client |
| `MCPServer(tools)` | Expose tools as MCP server |
| `mcp_tools(config)` | Context manager shortcut |
| `MCPError` | Base MCP exception |
| `MCPConnectionError` | Connection failure |
| `MCPToolError` | Tool call failure |
