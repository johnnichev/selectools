# Agent Module

**File:** `src/selectools/agent.py`
**Classes:** `Agent`, `AgentConfig`

## Table of Contents

1. [Overview](#overview)
2. [Agent Loop Lifecycle](#agent-loop-lifecycle)
3. [Tool Selection Process](#tool-selection-process)
4. [Configuration](#configuration)
5. [Retry and Error Handling](#retry-and-error-handling)
6. [Sync vs Async Execution](#sync-vs-async-execution)
7. [Hook System](#hook-system)
8. [Memory Integration](#memory-integration)
9. [Streaming](#streaming)
10. [Parallel Tool Execution](#parallel-tool-execution)
11. [Response Caching](#response-caching)
12. [Implementation Details](#implementation-details)

---

## Overview

The **Agent** class is the central orchestrator of the selectools framework. It manages the iterative loop of sending messages to an LLM, parsing responses for tool calls, executing those tools, and feeding results back until the task is complete.

### Key Responsibilities

1. **Conversation Management**: Maintain message history with optional memory
2. **Provider Communication**: Call LLM APIs through provider abstraction
3. **Tool Orchestration**: Detect, validate, and execute tool calls
4. **Error Recovery**: Handle failures with retries and backoff
5. **Observability**: Invoke lifecycle hooks for monitoring
6. **Cost Tracking**: Monitor token usage and costs
7. **Analytics**: Track tool usage patterns (optional)
8. **Parallel Execution**: Execute independent tool calls concurrently
9. **Streaming**: Token-level streaming with native tool support
10. **Response Caching**: Avoid redundant LLM calls via pluggable cache layer

### Core Dependencies

```python
from .types import Message, Role
from .tools import Tool
from .prompt import PromptBuilder
from .parser import ToolCallParser
from .providers.base import Provider
from .memory import ConversationMemory  # Optional
from .usage import AgentUsage
from .analytics import AgentAnalytics  # Optional
```

---

## Agent Loop Lifecycle

### State Machine Diagram

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Load Message History  │
              │  (from memory if set)  │
              └────────────┬───────────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │  ITERATION LOOP                 │
         │  (max_iterations times)         │
         └─────────────┬───────────────────┘
                       │
      ┌────────────────┴────────────────┐
      │                                 │
      ▼                                 ▼
┌──────────────┐              ┌──────────────────┐
│ Build Prompt │              │  Call Hook:      │
│ with Tools   │              │  on_iteration_   │
└──────┬───────┘              │  start           │
       │                      └──────────────────┘
       ▼
┌──────────────────┐
│ Call LLM Provider│
│ (with retries)   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Parse Response  │
│  (ToolCallParser)│
└──────┬───────────┘
       │
       ├─────────┐ No tool call?
       │         │
       │         ▼
       │    ┌────────────────┐
       │    │ Return Final   │
       │    │ Response to    │
       │    │ User           │
       │    └────────────────┘
       │
       │ Yes, tool call
       ▼
┌──────────────────┐
│ Validate Tool    │
│ Name & Params    │
└──────┬───────────┘
       │
       ├─────────┐ Invalid?
       │         │
       │         ▼
       │    ┌────────────────┐
       │    │ Append Error   │
       │    │ Message        │
       │    └────┬───────────┘
       │         │
       │         └──────┐
       │                │
       │ Valid          │
       ▼                │
┌──────────────────┐   │
│ Execute Tool     │   │
│ (with timeout)   │   │
└──────┬───────────┘   │
       │                │
       ├─────────┐      │
       │         │      │
       │ Success │ Fail │
       ▼         ▼      │
┌──────────────────┐   │
│ Append Result    │   │
│ to History       │   │
└──────┬───────────┘   │
       │                │
       └────────────────┘
       │
       ▼
   Loop continues
   (next iteration)
```

### Execution Flow

#### 1. Initialization

```python
agent = Agent(
    tools=[search_tool, calculator_tool],
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=6),
    memory=ConversationMemory(max_messages=20)
)
```

The agent initializes with:

- Tool registry (`_tools_by_name` dict for O(1) lookup)
- System prompt (built from tool schemas)
- Empty history
- Usage tracker
- Optional analytics tracker

#### 2. Run Method Entry

```python
response = agent.run([
    Message(role=Role.USER, content="Search for Python and calculate 2+2")
])
```

**Steps:**

1. Call `on_agent_start` hook
2. Load history from memory (if available)
3. Append new messages to history
4. Enter iteration loop

#### 3. Iteration Loop

```python
iteration = 0
while iteration < self.config.max_iterations:
    iteration += 1

    # 1. Call hook
    self._call_hook("on_iteration_start", iteration, self._history)

    # 2. Call provider
    response_text = self._call_provider()

    # 3. Parse response
    parse_result = self.parser.parse(response_text)

    # 4. Check for tool call
    if not parse_result.tool_call:
        # No tool call - we're done!
        return Message(role=Role.ASSISTANT, content=response_text)

    # 5. Execute tool
    tool = self._tools_by_name.get(tool_name)
    result = self._execute_tool_with_timeout(tool, parameters)

    # 6. Append to history
    self._append_assistant_and_tool(response_text, result, tool_name)

    # 7. Loop continues...
```

#### 4. Tool Execution

```python
def _execute_tool_with_timeout(self, tool, parameters, chunk_callback):
    if not self.config.tool_timeout_seconds:
        return tool.execute(parameters, chunk_callback)

    # Execute with timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool.execute, parameters, chunk_callback)
        try:
            return future.result(timeout=self.config.tool_timeout_seconds)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"Tool '{tool.name}' timed out")
```

**Features:**

- Optional timeout enforcement
- Chunk callback for streaming tools
- Exception wrapping for better errors
- Analytics tracking (if enabled)

#### 5. Loop Termination

The loop exits when:

1. **No tool call detected** → Return LLM response as final answer
2. **Max iterations reached** → Return timeout message
3. **Exception raised** → Propagate to caller

---

## Tool Selection Process

### How the Agent Decides Which Tool to Use

The agent doesn't directly "decide" - it relies on the LLM to make the decision based on the system prompt and conversation context.

```
┌──────────────────────────────────────────────────────────────┐
│ System Prompt (built by PromptBuilder)                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ You are an assistant that can call tools when helpful.      │
│                                                              │
│ Tool call contract:                                          │
│ - Emit TOOL_CALL with JSON: {"tool_name": "<name>",         │
│                              "parameters": {...}}            │
│ - Include every required parameter                           │
│ - Wait for tool results before giving a final answer        │
│                                                              │
│ Available tools (JSON schema):                               │
│                                                              │
│ {                                                            │
│   "name": "search",                                          │
│   "description": "Search the web for information",           │
│   "parameters": {                                            │
│     "type": "object",                                        │
│     "properties": {                                          │
│       "query": {"type": "string", "description": "..."}      │
│     },                                                       │
│     "required": ["query"]                                    │
│   }                                                          │
│ }                                                            │
│                                                              │
│ [... more tools ...]                                         │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Conversation History                                         │
├──────────────────────────────────────────────────────────────┤
│ USER: Search for Python tutorials                            │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ LLM (GPT-4o, Claude, Gemini, etc.)                          │
│ - Reads system prompt (knows about available tools)         │
│ - Analyzes user request                                      │
│ - Decides if a tool is needed                                │
│ - Generates TOOL_CALL if appropriate                         │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ LLM Response                                                 │
├──────────────────────────────────────────────────────────────┤
│ TOOL_CALL                                                    │
│ {                                                            │
│   "tool_name": "search",                                     │
│   "parameters": {"query": "Python tutorials"}               │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
```

### Validation Flow

```python
# 1. Parse the tool call
parse_result = self.parser.parse(response_text)

if parse_result.tool_call:
    tool_name = parse_result.tool_call.tool_name
    parameters = parse_result.tool_call.parameters

    # 2. Check if tool exists
    tool = self._tools_by_name.get(tool_name)
    if not tool:
        error_msg = f"Unknown tool '{tool_name}'. Available: {list(self._tools_by_name.keys())}"
        # Append error and continue loop
        self._append_assistant_and_tool(response_text, error_msg, tool_name)
        continue

    # 3. Validate parameters
    try:
        tool.validate(parameters)
    except ToolValidationError as e:
        # Append validation error and continue
        self._append_assistant_and_tool(response_text, str(e), tool_name)
        continue

    # 4. Execute tool
    result = tool.execute(parameters)
```

**Error Handling Strategy:**

The agent doesn't fail on invalid tool calls. Instead:

1. Append error message to conversation
2. Let LLM see the error
3. LLM can retry with corrections or choose a different approach

This creates a **self-correcting loop**.

---

## Configuration

### AgentConfig Dataclass

```python
@dataclass
class AgentConfig:
    # Model selection
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1000

    # Loop control
    max_iterations: int = 6

    # Reliability
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    rate_limit_cooldown_seconds: float = 5.0
    request_timeout: Optional[float] = 30.0
    tool_timeout_seconds: Optional[float] = None

    # Cost management
    cost_warning_threshold: Optional[float] = None

    # Observability
    verbose: bool = False
    enable_analytics: bool = False
    hooks: Optional[Hooks] = None

    # Execution mode
    routing_only: bool = False
    parallel_tool_execution: bool = True

    # Streaming
    stream: bool = False

    # Caching
    cache: Optional[Cache] = None  # InMemoryCache, RedisCache, or custom
```

### Configuration Patterns

#### Production Config

```python
config = AgentConfig(
    model="gpt-4o-mini",  # Cost-effective
    temperature=0.0,      # Deterministic
    max_tokens=2000,
    max_iterations=10,
    max_retries=3,
    retry_backoff_seconds=2.0,
    rate_limit_cooldown_seconds=10.0,
    request_timeout=60.0,
    tool_timeout_seconds=30.0,
    cost_warning_threshold=0.50,  # Alert at $0.50
    verbose=False,
    enable_analytics=True
)
```

#### Production Config with Caching

```python
from selectools import InMemoryCache

cache = InMemoryCache(max_size=2000, default_ttl=600)
config = AgentConfig(
    model="gpt-4o-mini",
    temperature=0.0,
    cache=cache,               # Enable response caching
    max_retries=3,
    cost_warning_threshold=0.50,
)
```

#### Development Config

```python
config = AgentConfig(
    model="gpt-4o",
    verbose=True,         # See what's happening
    max_iterations=3,     # Fast feedback
    stream=True,          # See responses live
)
```

#### Budget-Conscious Config

```python
config = AgentConfig(
    model="gpt-4o-mini",
    max_tokens=500,
    max_iterations=3,
    cost_warning_threshold=0.01,
)
```

---

## Retry and Error Handling

### Retry Logic Flow

```
Provider Call
    │
    ▼
┌─────────────┐
│  Attempt 1  │
└─────┬───────┘
      │
      ├─────── Success? ──→ Return
      │
      │ Failure (ProviderError)
      ▼
  Is rate limit?
      │
      ├── Yes ──→ Sleep(rate_limit_cooldown * attempt)
      │
      └── No
      │
      ▼
  Sleep(retry_backoff * attempt)
      │
      ▼
┌─────────────┐
│  Attempt 2  │
└─────┬───────┘
      │
      ├─────── Success? ──→ Return
      │
      │ Failure
      ▼
  [Repeat up to max_retries]
      │
      ▼
  Final Failure
      │
      ▼
  Return error message
```

### Implementation

```python
def _call_provider(self, stream_handler=None):
    attempts = 0
    last_error = None

    while attempts <= self.config.max_retries:
        attempts += 1

        try:
            # Call provider
            response_text, usage_stats = self.provider.complete(
                model=self.config.model,
                system_prompt=self._system_prompt,
                messages=self._history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )

            # Track usage
            self.usage.add_usage(usage_stats)

            return response_text

        except ProviderError as exc:
            last_error = str(exc)

            if attempts > self.config.max_retries:
                break

            # Rate limit handling
            if self._is_rate_limit_error(last_error):
                time.sleep(self.config.rate_limit_cooldown_seconds * attempts)

            # Standard backoff
            if self.config.retry_backoff_seconds:
                time.sleep(self.config.retry_backoff_seconds * attempts)

    return f"Provider error: {last_error or 'unknown error'}"
```

### Rate Limit Detection

```python
def _is_rate_limit_error(self, message: str) -> bool:
    lowered = message.lower()
    return "rate limit" in lowered or "429" in lowered
```

### Tool Execution Errors

Tool errors don't cause the entire agent to fail:

```python
try:
    result = self._execute_tool_with_timeout(tool, parameters)
    self._call_hook("on_tool_end", tool.name, result, duration)

except Exception as exc:
    self._call_hook("on_tool_error", tool.name, exc, parameters)

    error_message = f"Error executing tool '{tool.name}': {exc}"
    self._append_assistant_and_tool(response_text, error_message, tool.name)

    # Continue to next iteration - let LLM handle the error
    continue
```

---

## Sync vs Async Execution

### Sync Execution (`run()`)

```python
response = agent.run([Message(role=Role.USER, content="Hello")])
```

**When to use:**

- Simple scripts
- Jupyter notebooks
- Single-threaded applications
- Blocking I/O is acceptable

### Async Execution (`arun()`)

```python
response = await agent.arun([Message(role=Role.USER, content="Hello")])
```

**When to use:**

- Web frameworks (FastAPI, aiohttp)
- Concurrent operations
- High-performance applications
- Multiple agents in parallel

### Implementation Differences

#### Sync Path

```python
def run(self, messages, stream_handler=None):
    # Provider call (blocking)
    response_text, usage_stats = self.provider.complete(...)

    # Tool execution (blocking)
    result = tool.execute(parameters)
```

#### Async Path

```python
async def arun(self, messages, stream_handler=None):
    # Provider call (non-blocking)
    if hasattr(self.provider, "acomplete"):
        response_text, usage_stats = await self.provider.acomplete(...)
    else:
        # Fallback: run sync in executor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response_text, usage_stats = await loop.run_in_executor(
                executor, lambda: self.provider.complete(...)
            )

    # Tool execution (non-blocking)
    result = await tool.aexecute(parameters)
```

### Async Tool Support

Tools can be async:

```python
@tool(description="Fetch data from API")
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

The agent automatically detects and handles async tools via `tool.is_async` flag.

---

## Hook System

### Available Hooks

```python
hooks = {
    # Agent lifecycle
    "on_agent_start": lambda messages: ...,
    "on_agent_end": lambda response, usage: ...,

    # Iteration lifecycle
    "on_iteration_start": lambda iteration, messages: ...,
    "on_iteration_end": lambda iteration, response: ...,

    # Tool lifecycle
    "on_tool_start": lambda tool_name, args: ...,
    "on_tool_chunk": lambda tool_name, chunk: ...,  # Streaming tools
    "on_tool_end": lambda tool_name, result, duration: ...,
    "on_tool_error": lambda tool_name, error, args: ...,

    # LLM lifecycle
    "on_llm_start": lambda messages, model: ...,
    "on_llm_end": lambda response, usage: ...,

    # Error handling
    "on_error": lambda error, context: ...,
}

config = AgentConfig(hooks=hooks)
```

### Hook Invocation

```python
def _call_hook(self, hook_name: str, *args, **kwargs):
    if not self.config.hooks or hook_name not in self.config.hooks:
        return

    try:
        self.config.hooks[hook_name](*args, **kwargs)
    except Exception:
        # Silently ignore hook errors to prevent breaking agent
        pass
```

**Design Decision:** Hook errors never break the agent. They're for observability, not control flow.

### Use Cases

#### Logging

```python
def log_tool(tool_name, args):
    logger.info(f"Calling {tool_name} with {args}")

config = AgentConfig(hooks={"on_tool_start": log_tool})
```

#### Metrics

```python
def track_duration(tool_name, result, duration):
    metrics.histogram("tool_duration", duration, tags={"tool": tool_name})

config = AgentConfig(hooks={"on_tool_end": track_duration})
```

#### Debugging

```python
def debug_iteration(iteration, messages):
    print(f"=== Iteration {iteration} ===")
    for msg in messages:
        print(f"{msg.role}: {msg.content[:100]}")

config = AgentConfig(hooks={"on_iteration_start": debug_iteration})
```

---

## Memory Integration

### With Memory

```python
memory = ConversationMemory(max_messages=20)
agent = Agent(tools=[...], provider=provider, memory=memory)

# First turn
response1 = agent.run([Message(role=Role.USER, content="My name is Alice")])

# Second turn - history is preserved
response2 = agent.run([Message(role=Role.USER, content="What's my name?")])
# LLM can reference "Alice" from previous turn
```

**Flow:**

```
run() called
    │
    ├─→ memory.get_history()  # Load previous messages
    ├─→ Append new messages
    ├─→ memory.add_many(new_messages)
    │
    ├─→ Execute loop
    │
    ├─→ memory.add(final_response)  # Save response
    │
    └─→ Return
```

### Without Memory

```python
agent = Agent(tools=[...], provider=provider)  # No memory

# Each call is independent
response = agent.run([Message(role=Role.USER, content="Hello")])
```

History is local to each `run()` call.

### Memory Limits

```python
memory = ConversationMemory(
    max_messages=20,      # Keep last 20 messages
    max_tokens=4000       # Or limit by token count
)
```

When limits are exceeded, oldest messages are dropped (sliding window).

---

## Streaming

### Agent.astream()

The `astream()` method provides token-by-token streaming with native tool call support:

```python
async for item in agent.astream([Message(role=Role.USER, content="Search for Python")]):
    if isinstance(item, StreamChunk):
        print(item.content, end="", flush=True)
    elif isinstance(item, AgentResult):
        print(f"\nDone in {item.iterations} iterations")
```

### How It Works

1. Provider streams text deltas and tool call deltas via `astream()`
2. Text chunks are yielded as `StreamChunk` objects
3. Tool calls are accumulated until complete, then executed
4. Tool results are appended to history and the loop continues
5. Final `AgentResult` is yielded when no more tool calls

### Provider Protocol

All providers implement `astream()` returning `Union[str, ToolCall]`:

- **Text deltas**: Yielded as raw `str` chunks
- **Tool calls**: Yielded as complete `ToolCall` objects when ready

### Fallback Behavior

If a provider doesn't support `astream()`, the agent falls back to:

1. `acomplete()` (async non-streaming)
2. `complete()` via executor (sync in async wrapper)

The response is still yielded as a single `StreamChunk` for API consistency.

---

## Parallel Tool Execution

### Overview

When an LLM requests multiple tool calls in a single response (common with native function calling), the agent executes them concurrently instead of sequentially.

### Configuration

```python
config = AgentConfig(
    parallel_tool_execution=True  # Default: enabled
)
```

Set to `False` to force sequential execution.

### How It Works

#### Async (`arun`, `astream`)

Uses `asyncio.gather()` to run all tool calls concurrently:

```python
results = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])
```

#### Sync (`run`)

Uses `ThreadPoolExecutor` with one worker per tool call:

```python
with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
    futures = [pool.submit(run_tool, tc) for tc in tool_calls]
    results = [f.result() for f in futures]
```

### Guarantees

1. **Result ordering**: Tool results are appended to history in the same order as the original tool calls, regardless of completion order
2. **Error isolation**: If one tool fails, others still complete successfully
3. **Hook invocation**: `on_tool_start`, `on_tool_end`, and `on_tool_error` fire for every tool
4. **Single tool optimization**: When only one tool is called, the sequential path is used (no overhead)

### Example

Three tools each taking 0.15s:
- **Sequential**: ~0.45s total
- **Parallel**: ~0.15s total (3x speedup)

```python
# Automatic - no code changes needed
agent = Agent(
    tools=[weather_tool, stock_tool, news_tool],
    provider=OpenAIProvider(),
    config=AgentConfig(parallel_tool_execution=True)
)

# LLM requests all 3 tools → executed concurrently
result = await agent.arun([Message(role=Role.USER, content="...")])
```

---

## Response Caching

### Overview

The agent supports **pluggable response caching** to avoid redundant LLM calls. When `AgentConfig(cache=...)` is set, the agent checks the cache before every `provider.complete()` / `provider.acomplete()` call. On a cache hit, the stored `(Message, UsageStats)` is returned immediately without calling the LLM.

### Architecture

```
Agent._call_provider()
    │
    ├─→ CacheKeyBuilder.build(model, prompt, messages, tools, temperature)
    │     → SHA-256 hex digest (deterministic)
    │
    ├─→ cache.get(key)
    │     ├── HIT  → return cached (Message, UsageStats), fire on_llm_end hook
    │     └── MISS → continue to provider call
    │
    ├─→ provider.complete(...)
    │
    └─→ cache.set(key, (response_msg, usage_stats))
```

### Cache Protocol

Any object satisfying the `Cache` protocol can be used:

```python
@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> Optional[Tuple[Any, Any]]: ...
    def set(self, key: str, value: Tuple[Any, Any], ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...

    @property
    def stats(self) -> CacheStats: ...
```

### Built-in Backends

#### InMemoryCache

Thread-safe LRU + TTL cache with zero external dependencies:

```python
from selectools import InMemoryCache

cache = InMemoryCache(
    max_size=1000,    # Max entries (LRU eviction)
    default_ttl=300,  # 5 minutes
)
```

**Features:**
- `OrderedDict`-based O(1) LRU operations
- Per-entry TTL with monotonic timestamp expiry
- Thread-safe via `threading.Lock`
- `CacheStats` tracking (hits, misses, evictions, hit_rate)

#### RedisCache

Distributed TTL cache for multi-process deployments:

```python
from selectools.cache_redis import RedisCache

cache = RedisCache(
    url="redis://localhost:6379/0",
    prefix="selectools:",
    default_ttl=300,
)
```

**Features:**
- Server-side TTL management
- Pickle-serialized `(Message, UsageStats)` entries
- Key prefix namespacing
- Requires optional dependency: `pip install selectools[cache]`

### Cache Key Generation

`CacheKeyBuilder` creates deterministic SHA-256 keys from request parameters:

```python
from selectools import CacheKeyBuilder

key = CacheKeyBuilder.build(
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    messages=[Message(role=Role.USER, content="Hello")],
    tools=[my_tool],
    temperature=0.0,
)
# → "selectools:a3f2b8c1d4e5..."
```

**Inputs hashed:** model, system_prompt, messages (role + content + tool_calls), tools (name + description + parameters), temperature.

**Guarantees:**
- Same inputs always produce the same key
- Different inputs produce different keys
- Tool ordering is preserved in the hash

### What Gets Cached

| Call Type | Cached? | Reason |
| --- | --- | --- |
| `provider.complete()` | Yes | Deterministic request/response |
| `provider.acomplete()` | Yes | Deterministic request/response |
| `provider.astream()` | No | Non-replayable generator |
| Tool execution results | No | Side effects possible |

### Usage Examples

#### Basic In-Memory Caching

```python
from selectools import Agent, AgentConfig, InMemoryCache

cache = InMemoryCache(max_size=500, default_ttl=600)
config = AgentConfig(model="gpt-4o-mini", cache=cache)
agent = Agent(tools=[my_tool], provider=provider, config=config)

# First call → cache miss → LLM called
response1 = agent.run([Message(role=Role.USER, content="What is Python?")])

# Reset history, same question → cache hit → instant response
agent.reset()
response2 = agent.run([Message(role=Role.USER, content="What is Python?")])

print(cache.stats)
# CacheStats(hits=1, misses=1, evictions=0, hit_rate=50.00%)
```

#### Distributed Redis Caching

```python
from selectools.cache_redis import RedisCache

cache = RedisCache(url="redis://my-redis:6379/0", default_ttl=900)
config = AgentConfig(cache=cache)

# Cache is shared across processes/servers
agent = Agent(tools=[...], provider=provider, config=config)
```

#### Monitoring Cache Performance

```python
stats = cache.stats
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hits: {stats.hits}, Misses: {stats.misses}")
print(f"Evictions: {stats.evictions}")
```

### Verbose Mode

When `verbose=True`, cache hits are logged:

```
[agent] cache hit -- skipping provider call
```

### Integration with Usage Tracking

Cache hits still contribute to `AgentUsage`. The stored `UsageStats` is replayed via `agent.usage.add_usage()`, so cost tracking remains accurate even when responses come from cache.

---

## Implementation Details

### Key Attributes

```python
class Agent:
    def __init__(self, tools, provider, config, memory):
        self.tools = tools                      # List of Tool objects
        self._tools_by_name = {...}             # Dict for O(1) lookup
        self.provider = provider                # Provider instance
        self.prompt_builder = PromptBuilder()   # Generates system prompts
        self.parser = ToolCallParser()          # Parses tool calls
        self.config = config                    # AgentConfig
        self.memory = memory                    # Optional ConversationMemory
        self.usage = AgentUsage()               # Tracks tokens/cost
        self.analytics = AgentAnalytics()       # Optional analytics

        # Pre-build system prompt (constant per agent instance)
        self._system_prompt = self.prompt_builder.build(self.tools)

        # Local conversation history (reset per run if no memory)
        self._history: List[Message] = []
```

### History Management

```python
def _append_assistant_and_tool(self, assistant_content, tool_content, tool_name, tool_result=None):
    assistant_msg = Message(role=Role.ASSISTANT, content=assistant_content)
    tool_msg = Message(
        role=Role.TOOL,
        content=tool_content,
        tool_name=tool_name,
        tool_result=tool_result,
    )

    # Append to local history
    self._history.append(assistant_msg)
    self._history.append(tool_msg)

    # Also save to memory if available
    if self.memory:
        self.memory.add_many([assistant_msg, tool_msg])
```

### Usage Tracking Convenience Methods

```python
@property
def total_cost(self) -> float:
    return self.usage.total_cost_usd

@property
def total_tokens(self) -> int:
    return self.usage.total_tokens

def get_usage_summary(self) -> str:
    return str(self.usage)  # Pretty-printed summary

def reset_usage(self) -> None:
    self.usage = AgentUsage()
```

### Analytics Access

```python
def get_analytics(self) -> AgentAnalytics | None:
    return self.analytics  # None if not enabled
```

---

## Best Practices

### 1. Choose Appropriate Iteration Limits

```python
# Quick interactions
config = AgentConfig(max_iterations=3)

# Complex multi-step tasks
config = AgentConfig(max_iterations=10)

# Simple single-shot (no tools expected)
config = AgentConfig(max_iterations=1)
```

### 2. Set Tool Timeouts

```python
config = AgentConfig(
    tool_timeout_seconds=30.0  # Prevent runaway tools
)
```

### 3. Use Verbose Mode for Debugging

```python
config = AgentConfig(verbose=True)
# Prints token counts, costs, tool calls
```

### 4. Enable Cost Warnings

```python
config = AgentConfig(
    cost_warning_threshold=0.10  # Warn at $0.10
)
```

### 5. Reset Usage Between Sessions

```python
agent.reset_usage()  # Clear token/cost counters
```

### 6. Use Memory for Conversations

```python
# For chatbots, Q&A systems, assistants
memory = ConversationMemory(max_messages=20)
agent = Agent(..., memory=memory)
```

### 7. Enable Analytics for Optimization

```python
config = AgentConfig(enable_analytics=True)
agent = Agent(..., config=config)

# Later: analyze which tools are used most
analytics = agent.get_analytics()
print(analytics.summary())
```

---

## Performance Optimization

### 1. Reuse Agent Instances

```python
# Good: Create once, use many times
agent = Agent(tools=[...], provider=provider)
for query in queries:
    response = agent.run([Message(role=Role.USER, content=query)])
```

### 2. Use Async for Concurrency

```python
# Process multiple queries concurrently
async def process_queries(queries):
    agent = Agent(...)
    tasks = [agent.arun([Message(role=Role.USER, content=q)]) for q in queries]
    return await asyncio.gather(*tasks)
```

### 3. Limit max_tokens

```python
# Reduce output tokens to save cost
config = AgentConfig(max_tokens=500)
```

### 4. Choose Efficient Models

```python
# Use mini models when appropriate
config = AgentConfig(model="gpt-4o-mini")  # 15x cheaper than gpt-4o
```

---

## Testing

### Unit Testing with Local Provider

```python
from selectools.providers.stubs import LocalProvider

agent = Agent(
    tools=[my_tool],
    provider=LocalProvider(),  # No API calls
    config=AgentConfig(max_iterations=2, model="local")
)

response = agent.run([Message(role=Role.USER, content="test")])
```

### Mocking Hooks

```python
def test_agent_with_hooks():
    called = []

    def track_calls(tool_name, args):
        called.append((tool_name, args))

    config = AgentConfig(hooks={"on_tool_start": track_calls})
    agent = Agent(tools=[...], provider=provider, config=config)

    agent.run([...])

    assert len(called) > 0
    assert called[0][0] == "expected_tool"
```

---

## Common Pitfalls

### 1. Forgetting to Set API Keys

```python
# ❌ This will raise ProviderConfigurationError
provider = OpenAIProvider()  # OPENAI_API_KEY not set

# ✅ Set via env var
export OPENAI_API_KEY="sk-..."

# ✅ Or pass directly
provider = OpenAIProvider(api_key="sk-...")
```

### 2. Infinite Loops

```python
# ❌ If LLM keeps calling tools that fail
config = AgentConfig(max_iterations=1000)  # Dangerous!

# ✅ Use reasonable limits
config = AgentConfig(max_iterations=6)  # Default is safe
```

### 3. Not Handling Tool Errors

```python
# Agent handles tool errors gracefully by default
# But tools should still validate inputs and provide helpful errors

@tool(description="Divide two numbers")
def divide(a: float, b: float) -> str:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return str(a / b)
```

---

## Further Reading

- [Tools Module](TOOLS.md) - Tool definition and validation
- [Parser Module](PARSER.md) - Tool call parsing details
- [Providers Module](PROVIDERS.md) - Provider implementations
- [Memory Module](MEMORY.md) - Conversation memory
- [Usage Module](USAGE.md) - Cost tracking

---

**Next Steps:** Understand how tools are defined and validated in the [Tools Module](TOOLS.md).
