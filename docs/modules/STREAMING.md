# Streaming and Performance Module

**Directory:** `src/selectools/agent/`
**Key Types:** `StreamChunk`, `AgentResult` (from `selectools.types`)

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [E2E Streaming (v0.11.0)](#e2e-streaming-v0110)
4. [Parallel Tool Execution (v0.11.0)](#parallel-tool-execution-v0110)
5. [Native Function Calling (v0.10.0)](#native-function-calling-v0100)
6. [Routing Mode (v0.10.0)](#routing-mode-v0100)
7. [Context Propagation (v0.10.0)](#context-propagation-v0100)
8. [AgentResult (v0.9.0)](#agentresult-v090)
9. [Custom System Prompt (v0.9.0)](#custom-system-prompt-v090)
10. [Agent.reset() (v0.9.0)](#agentreset-v090)
11. [Performance Comparison](#performance-comparison)
12. [Practical Examples](#practical-examples)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)
15. [Further Reading](#further-reading)

---

## Overview

The selectools library provides a rich set of streaming and performance features that enable real-time token delivery, concurrent tool execution, and programmatic inspection of agent behavior. These capabilities span from token-level streaming (`astream`) to routing without execution (`routing_only`), from native function calling to context-preserving tool execution.

### Feature Summary

| Feature | Version | Purpose |
|---------|---------|---------|
| **E2E Streaming** | v0.11.0 | Token-by-token output with native tool call support |
| **Parallel Tool Execution** | v0.11.0 | Run multiple tools concurrently in a single iteration |
| **Native Function Calling** | v0.10.0 | Provider-native tool APIs, no regex parsing |
| **Routing Mode** | v0.10.0 | Select a tool without executing it (classification, intent routing) |
| **Context Propagation** | v0.10.0 | Preserve tracing and auth when running sync tools in executors |
| **AgentResult** | v0.9.0 | Structured return with message, tool metadata, iterations |
| **Custom System Prompt** | v0.9.0 | Inject domain instructions via `AgentConfig` |
| **Agent.reset()** | v0.9.0 | Clear state for clean reuse across requests |

### Import Paths

```python
from selectools import Agent, AgentConfig, Message, Role
from selectools.types import StreamChunk, AgentResult
```

---

## Quick Start

### Streaming with astream()

```python
import asyncio
from selectools import Agent, AgentConfig, Message, Role, OpenAIProvider
from selectools.types import StreamChunk, AgentResult

agent = Agent(
    tools=[search_tool],
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=3),
)


async def main():
    async for item in agent.astream([Message(role=Role.USER, content="Search for Python tutorials")]):
        if isinstance(item, StreamChunk):
            print(item.content, end="", flush=True)
        elif isinstance(item, AgentResult):
            print(f"\n\nDone in {item.iterations} iterations")
            if item.tool_calls:
                print(f"Tools used: {[tc.tool_name for tc in item.tool_calls]}")


asyncio.run(main())
```

---

## E2E Streaming (v0.11.0)

### Agent.astream()

`Agent.astream(messages)` returns an `AsyncGenerator` yielding `Union[StreamChunk, AgentResult]`:

- **StreamChunk** — Intermediate content chunks (text and/or tool calls)
- **AgentResult** — Final result, yielded once when the agent completes

### StreamChunk

```python
@dataclass
class StreamChunk:
    content: str = ""                              # Text delta
    role: Role = Role.ASSISTANT
    tool_calls: Optional[List[ToolCall]] = None    # Optional; present when chunk contains tool invocations
```

- `content`: The text portion of this chunk
- `tool_calls`: Optional list of `ToolCall` objects when the LLM emits tool invocations during streaming

### AgentResult as Final Item

The last item yielded by `astream()` is always an `AgentResult`. It carries:

- `message` — Final assistant response
- `tool_name` — Last tool called (or `None`)
- `tool_args` — Args for last tool
- `iterations` — Number of loop iterations
- `tool_calls` — All `ToolCall` objects from the run

### Provider Protocol

Providers implement `astream()` yielding `Union[str, ToolCall]`:

- **Text deltas** — Raw `str` chunks (token-by-token)
- **Tool calls** — Complete `ToolCall` objects when ready (native function calling)

```
Provider.astream()
    │
    ├──► yield "Hello"       (str)
    ├──► yield " "           (str)
    ├──► yield "world"       (str)
    ├──► yield ToolCall(...) (when tool invocation complete)
    └──► yield "!"           (str)
```

### Fallback Chain

When a provider does not support streaming:

```
astream() requested
    │
    ├──► Provider has astream()?  ──► Use it
    │
    ├──► Provider has acomplete()? ──► Call it, yield full response as single StreamChunk
    │
    └──► Otherwise ──► Run complete() in ThreadPoolExecutor (sync in async wrapper)
```

### Tool Call Accumulation and Multi-Iteration

1. **Accumulation**: Tool calls are accumulated as they stream in from the provider.
2. **Execution**: When all tool calls in a response are ready, they are executed (in parallel if `parallel_tool_execution=True`).
3. **Continue**: Results are appended to history; streaming continues with the next LLM call.
4. **Final result**: When the LLM produces a final text response with no tool calls, `AgentResult` is yielded.

```
┌─────────────────────────────────────────────────────────────────┐
│                     astream() Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Iteration 1:                                                    │
│    StreamChunk("Searching...")                                   │
│    StreamChunk(tool_calls=[ToolCall(search, {"query": "..."})])  │
│    [Tools executed]                                              │
│                                                                  │
│  Iteration 2 (streaming continues):                              │
│    StreamChunk("Here are the results:")                          │
│    StreamChunk("  - Result 1")                                   │
│    ...                                                           │
│    AgentResult(message=..., iterations=2, tool_calls=[...])      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallel Tool Execution (v0.11.0)

### Overview

When the LLM requests multiple tool calls in a single response (common with native function calling), the agent executes them concurrently instead of sequentially.

### Configuration

```python
config = AgentConfig(
    parallel_tool_execution=True  # Default: enabled
)
```

Set to `False` for strictly sequential execution.

### Async Execution

Uses `asyncio.gather()` for concurrent tool runs:

```python
results = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])
```

### Sync Execution

Uses `ThreadPoolExecutor` with one worker per tool:

```python
with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
    futures = [pool.submit(run_tool, tc) for tc in tool_calls]
    results = [f.result() for f in futures]
```

### Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Result ordering** | Results appended to history in original request order |
| **Error isolation** | One tool failure does not block others |
| **Hook invocation** | `on_tool_start`, `on_tool_end`, `on_tool_error` fire for every tool |
| **Single-tool optimization** | Only one tool called → sequential path, no executor overhead |

---

## Native Function Calling (v0.10.0)

### Overview

selectools uses provider-native tool APIs instead of regex parsing:

- **OpenAI** — `functions` / `tool_use` in chat completions
- **Anthropic** — `tool_use` blocks
- **Gemini** — `function_calling` in responses

### Message.tool_calls

Responses carry structured `ToolCall` objects on `Message.tool_calls`:

```python
response = provider.complete(...)
msg = response[0]

if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Tool: {tc.tool_name}, Args: {tc.parameters}")
```

### No Regex Parsing

- Providers return `ToolCall` objects directly.
- No text-based patterns such as `TOOL_CALL {...}`.

### Fallback

When a provider returns plain text only (no native tool format), the agent falls back to `ToolCallParser` regex parsing.

---

## Routing Mode (v0.10.0)

### Overview

`AgentConfig(routing_only=True)` makes the agent choose a tool but not run it. Useful for classification, intent routing, and tool selection.

### Configuration

```python
config = AgentConfig(routing_only=True)
agent = Agent(tools=[...], provider=provider, config=config)
```

### Return Value

Returns `AgentResult` with:

- `tool_name` — Selected tool
- `tool_args` — Parsed arguments
- `message` — Assistant message containing the selection

No tool execution; only one LLM call.

### Use Cases

| Use Case | Example |
|----------|---------|
| **Classification** | Route to sales vs support vs billing |
| **Intent detection** | Choose between search, calculator, or Q&A |
| **Tool preselection** | Decide which tools to enable before full execution |

### Example

```python
from selectools import Agent, AgentConfig, Message, Role, OpenAIProvider
from selectools.types import AgentResult

config = AgentConfig(routing_only=True)
agent = Agent(
    tools=[search_tool, calculator_tool, support_tool],
    provider=OpenAIProvider(),
    config=config,
)

result = agent.run([Message(role=Role.USER, content="I need help with my bill")])

# Inspect routing decision without executing
assert result.tool_name == "support_tool"
assert "billing" in str(result.tool_args).lower() or "bill" in str(result.tool_args).lower()
```

---

## Context Propagation (v0.10.0)

### Overview

When sync tools run inside a `ThreadPoolExecutor` (e.g. async agent calling sync tools), `contextvars.copy_context()` is used so request-scoped state (tracing, auth, etc.) is preserved.

### How It Works

```python
# In tools/base.py - sync tool execution from async context
context = contextvars.copy_context()
func_with_args = functools.partial(self.function, **call_args)
result = await loop.run_in_executor(executor, context.run, func_with_args)
```

### Preserved State

- OpenTelemetry tracing spans
- Auth tokens
- Request IDs
- Other `contextvars` values

### Async Tools

Async tools run in the same event loop as the agent; no executor, so context is already intact.

---

## AgentResult (v0.9.0)

### Overview

`agent.run()` and `agent.arun()` return `AgentResult` instead of `Message`, enabling programmatic inspection of tool usage and iterations.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `message` | `Message` | Final assistant response |
| `tool_name` | `Optional[str]` | Last tool called, or `None` |
| `tool_args` | `Dict[str, Any]` | Args for last tool call |
| `iterations` | `int` | Number of agent loop iterations |
| `tool_calls` | `List[ToolCall]` | All tool calls in order |

### Backward Compatibility

- `result.content` → `result.message.content`
- `result.role` → `result.message.role`

### Example

```python
result = agent.run([Message(role=Role.USER, content="What's the weather in Tokyo?")])

print(result.content)           # Final text
print(result.tool_name)         # e.g. "get_weather"
print(result.tool_args)         # e.g. {"location": "Tokyo"}
print(result.iterations)        # e.g. 2
print(len(result.tool_calls))   # Number of tools invoked
```

---

## Custom System Prompt (v0.9.0)

### Overview

`AgentConfig(system_prompt="...")` injects domain instructions before tool schemas. They persist across iterations.

### Configuration

```python
config = AgentConfig(
    system_prompt="You are a medical assistant. Only provide information you are confident about."
)
agent = Agent(tools=[...], provider=provider, config=config)
```

### When to Use

- Domain constraints (medical, legal, etc.)
- Tone and persona
- Guardrails and safety
- Language or formatting rules

### Example

```python
config = AgentConfig(
    system_prompt="""You are a financial advisor.
    - Never guarantee returns.
    - Always recommend consulting a licensed professional.
    - Use clear, non-technical language."""
)
agent = Agent(tools=[lookup_stock, get_news], provider=provider, config=config)
```

---

## Agent.reset() (v0.9.0)

### Overview

`Agent.reset()` clears history, usage stats, analytics, and memory so the same agent instance can be reused across requests.

### What It Clears

- `_history` — Message history
- `usage` — Token/cost stats
- `analytics` — If enabled
- `memory` — If `ConversationMemory` is set, calls `memory.clear()`

### Pattern

```python
agent = Agent(tools=[...], provider=provider, memory=ConversationMemory())

# Create once, reset between requests
for user_request in requests:
    agent.reset()
    result = agent.run([Message(role=Role.USER, content=user_request)])
```

---

## Performance Comparison

### Sequential vs Parallel Tool Execution

| Scenario | Sequential | Parallel | Speedup |
|----------|------------|----------|---------|
| 3 tools × 0.15s each | ~0.45s | ~0.15s | ~3× |
| 5 tools × 0.2s each | ~1.0s | ~0.2s | ~5× |
| 1 tool | 0.15s | 0.15s | 1× (no overhead) |

### Benchmark Example

```python
import time
from selectools import Agent, AgentConfig, Message, Role, tool

@tool(description="Simulate slow API")
def slow_api(delay: float) -> str:
    time.sleep(delay)
    return f"Done after {delay}s"

agent_parallel = Agent(
    tools=[slow_api],
    provider=provider,
    config=AgentConfig(parallel_tool_execution=True, max_iterations=2),
)
agent_sequential = Agent(
    tools=[slow_api],
    provider=provider,
    config=AgentConfig(parallel_tool_execution=False, max_iterations=2),
)

# With a prompt that triggers 3 tool calls:
# parallel: ~0.15s
# sequential: ~0.45s
```

---

## Practical Examples

### Routing Mode for Intent Classification

```python
config = AgentConfig(routing_only=True)
agent = Agent(
    tools=[sales_tool, support_tool, billing_tool],
    provider=provider,
    config=config,
)

intent = agent.run([Message(role=Role.USER, content=user_message)])
if intent.tool_name == "sales_tool":
    route_to_sales_team(intent.tool_args)
elif intent.tool_name == "support_tool":
    create_support_ticket(intent.tool_args)
else:
    forward_to_billing(intent.tool_args)
```

### AgentResult Inspection for Analytics

```python
result = agent.run(messages)

if result.tool_calls:
    for tc in result.tool_calls:
        log_tool_usage(tc.tool_name, tc.parameters)

if result.iterations > 3:
    alert_complex_conversation()
```

### System Prompt for Domain Experts

```python
config = AgentConfig(
    system_prompt="You are a Python expert. Prefer type hints and modern syntax. Suggest tests when relevant.",
    max_iterations=5,
)
agent = Agent(tools=[search_docs, run_code], provider=provider, config=config)
```

---

## Best Practices

### 1. Use astream() for Responsive UX

```python
async for item in agent.astream(messages):
    if isinstance(item, StreamChunk):
        await websocket.send_json({"type": "chunk", "content": item.content})
    elif isinstance(item, AgentResult):
        await websocket.send_json({"type": "done", "iterations": item.iterations})
```

### 2. Keep parallel_tool_execution Enabled

Default is `True`; disable only when tool ordering or side effects require sequential execution.

### 3. Prefer routing_only for Classification

Use routing mode for cheap classification instead of a full agent run.

### 4. Reuse Agents with reset()

```python
agent = Agent(...)
for req in queue:
    agent.reset()
    result = agent.run(req)
```

### 5. Use AgentResult for Observability

Use `result.tool_calls` and `result.iterations` for logging and monitoring.

---

## Troubleshooting

### Streaming Yields Nothing Until Complete

**Cause**: Provider lacks `astream()`; agent falls back to `acomplete()` and yields a single chunk.

**Fix**: Use a provider that implements `astream()` (e.g. OpenAI, Anthropic, Gemini).

### Parallel Tools Seem Sequential

**Cause**: `parallel_tool_execution=False` or only one tool per response.

**Fix**: Set `AgentConfig(parallel_tool_execution=True)` and use prompts that trigger multiple tools.

### Context Lost in Sync Tools

**Cause**: Older selectools versions or custom executor usage without context propagation.

**Fix**: Upgrade to v0.10.0+; sync tools from async agent should receive proper context propagation.

### routing_only Still Executes Tools

**Cause**: Misconfiguration or different code path.

**Fix**: Ensure `AgentConfig(routing_only=True)` is passed to `Agent`, not just `AgentConfig()`.

---

## Further Reading

- [Agent Module](AGENT.md) - Agent lifecycle, hooks, configuration
- [Tools Module](TOOLS.md) - Tool definition and validation
- [Providers Module](PROVIDERS.md) - Provider implementations and streaming
- [Memory Module](MEMORY.md) - Conversation memory and `reset()`

---

**Next Steps:** Enable streaming with `agent.astream()` and optimize tool-heavy workflows with `parallel_tool_execution=True`.
