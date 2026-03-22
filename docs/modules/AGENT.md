# Agent Module

**File:** `src/selectools/agent/core.py`
**Classes:** `Agent`, `AgentConfig`

## Table of Contents

1. [Overview](#overview)
2. [Agent Loop Lifecycle](#agent-loop-lifecycle)
3. [Tool Selection Process](#tool-selection-process)
4. [Configuration](#configuration)
5. [Retry and Error Handling](#retry-and-error-handling)
6. [Sync vs Async Execution](#sync-vs-async-execution)
7. [Hook System](#hook-system)
8. [AgentObserver Protocol](#agentobserver-protocol)
9. [Memory Integration](#memory-integration)
10. [Streaming](#streaming)
11. [Parallel Tool Execution](#parallel-tool-execution)
12. [Response Caching](#response-caching)
13. [Structured Output](#structured-output)
14. [Execution Traces](#execution-traces)
15. [Reasoning Visibility](#reasoning-visibility)
16. [Provider Fallback](#provider-fallback)
17. [Batch Processing](#batch-processing)
18. [Tool Policy & Human-in-the-Loop](#tool-policy-human-in-the-loop)
19. [Terminal Actions](#terminal-actions)
20. [Implementation Details](#implementation-details)

---

## Overview

The **Agent** class is the central orchestrator of the selectools framework. It manages the iterative loop of sending messages to an LLM, parsing responses for tool calls, executing those tools, and feeding results back until the task is complete.

### Key Responsibilities

1. **Conversation Management**: Maintain message history with optional memory
2. **Provider Communication**: Call LLM APIs through provider abstraction (with fallback)
3. **Tool Orchestration**: Detect, validate, enforce policies, and execute tool calls
4. **Structured Output**: Validate LLM responses against Pydantic/JSON Schema with auto-retry
5. **Execution Traces**: Record structured timeline of every step (`AgentTrace`)
6. **Reasoning Visibility**: Extract and surface *why* the agent chose a tool
7. **Error Recovery**: Handle failures with retries and backoff
8. **Observability**: Invoke lifecycle hooks for monitoring
9. **Cost Tracking**: Monitor token usage and costs
10. **Analytics**: Track tool usage patterns (optional)
11. **Parallel Execution**: Execute independent tool calls concurrently
12. **Batch Processing**: Process multiple prompts concurrently
13. **Streaming**: Token-level streaming with native tool support
14. **Response Caching**: Avoid redundant LLM calls via pluggable cache layer
15. **Tool Policy & HITL**: Declarative allow/review/deny rules with human approval

### Properties & Convenience Methods

| Property / Method | Description |
|---|---|
| `agent.name` | Returns `config.name` (default: `"agent"`). Useful for multi-agent identification. |
| `agent(messages, **kw)` | Shorthand for `agent.run(messages, **kw)` via `__call__`. |
| `agent.ask(prompt)` | Shorthand for `run()` with a single string prompt. |
| `agent.aask(prompt)` | Async version of `ask()`. |

```python
# Named agents for multi-agent systems
researcher = Agent(tools=[search], config=AgentConfig(name="researcher"))
print(researcher.name)  # "researcher"

# Call the agent directly
result = researcher("Find info about Python")  # same as researcher.run(...)
```

### Core Dependencies

```python
from .types import Message, Role
from .tools import Tool
from .prompt import PromptBuilder
from .parser import ToolCallParser
from .structured import parse_and_validate, build_schema_instruction
from .trace import AgentTrace, TraceStep
from .policy import ToolPolicy, PolicyDecision
from .providers.base import Provider
from .providers.fallback import FallbackProvider
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

    # Tool Safety
    tool_policy: Optional[ToolPolicy] = None  # allow/review/deny rules
    confirm_action: Optional[ConfirmAction] = None  # Human-in-the-loop callback
    approval_timeout: float = 60.0  # Seconds before auto-deny

    # Sessions & Persistence (v0.16.0)
    session_store: Optional[SessionStore] = None  # Auto-save/load conversation state
    session_id: Optional[str] = None  # Unique session identifier

    # Summarize-on-Trim (v0.16.0)
    summarize_on_trim: bool = False  # Summarize trimmed messages before dropping
    summarize_provider: Optional[Provider] = None  # Provider for summarization (defaults to agent's)
    summarize_model: Optional[str] = None  # Model for summarization (use a cheap model)
    summarize_max_tokens: int = 150  # Max tokens for the summary response

    # Advanced Memory (v0.16.0)
    entity_memory: Optional[EntityMemory] = None  # LLM-based entity extraction
    knowledge_graph: Optional[KnowledgeGraphMemory] = None  # Relationship triple extraction
    knowledge_memory: Optional[KnowledgeMemory] = None  # Cross-session durable memory
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

All three execution methods share the same parameters and feature set (as of v0.16.3):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `str \| List[Message]` | required | User prompt or message list |
| `stream_handler` | `Callable[[str], None]` | `None` | Callback for streaming chunks (run/arun only) |
| `response_format` | `ResponseFormat` | `None` | Pydantic model or JSON Schema for structured output |
| `parent_run_id` | `str` | `None` | Links trace to a parent agent's run for nested orchestration |

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

> **Deprecated in v0.16.5** — `AgentConfig.hooks` emits a `DeprecationWarning` and is
> internally wrapped via `_HooksAdapter` into the observer pipeline. Use `AgentObserver`
> or `AsyncAgentObserver` instead. Hooks continue to function but will be removed in a
> future release.

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

## AgentObserver Protocol

**File:** `src/selectools/observer.py`
**Classes:** `AgentObserver`, `LoggingObserver`

The **AgentObserver** protocol is a class-based alternative to the hooks dict, designed for structured observability integrations (Langfuse, OpenTelemetry, Datadog). Every callback receives a **`run_id`** for cross-request correlation, and tool callbacks also receive a **`call_id`** for matching parallel tool start/end pairs.

### Quick Start

```python
from selectools import Agent, AgentConfig, AgentObserver, LoggingObserver

class MyObserver(AgentObserver):
    def on_llm_start(self, run_id, messages, model, system_prompt):
        print(f"[{run_id}] LLM call to {model}")

    def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
        print(f"[{run_id}] {tool_name} finished in {duration_ms:.1f}ms")

agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(observers=[MyObserver(), LoggingObserver()]),
)
```

### All 25 Lifecycle Events

| Event | Scope | Parameters (after `run_id`) | When |
|---|---|---|---|
| `on_run_start` | Run | `messages`, `system_prompt` | Start of `run()`/`arun()`/`astream()` |
| `on_run_end` | Run | `result` (AgentResult) | Agent produces final result |
| `on_error` | Run | `error`, `context` | Unrecoverable error |
| `on_llm_start` | LLM | `messages`, `model`, `system_prompt` | Before each provider call |
| `on_llm_end` | LLM | `response`, `usage` | After each provider call |
| `on_cache_hit` | LLM | `model`, `response` | Response served from cache |
| `on_usage` | LLM | `usage` (UsageStats) | Per-call token/cost stats |
| `on_llm_retry` | LLM | `attempt`, `max_retries`, `error`, `backoff_seconds` | LLM call about to be retried |
| `on_tool_start` | Tool | `call_id`, `tool_name`, `tool_args` | Before tool execution |
| `on_tool_end` | Tool | `call_id`, `tool_name`, `result`, `duration_ms` | After successful tool execution |
| `on_tool_error` | Tool | `call_id`, `tool_name`, `error`, `tool_args`, `duration_ms` | Tool raised an exception |
| `on_tool_chunk` | Tool | `call_id`, `tool_name`, `chunk` | Streaming tool emits a chunk |
| `on_iteration_start` | Iteration | `iteration`, `messages` | Start of agent loop iteration |
| `on_iteration_end` | Iteration | `iteration`, `response` | End of agent loop iteration |
| `on_batch_start` | Batch | `batch_id`*, `prompts_count` | Before `batch()`/`abatch()` |
| `on_batch_end` | Batch | `batch_id`*, `results_count`, `errors_count`, `total_duration_ms` | After all batch items complete |
| `on_policy_decision` | Policy | `tool_name`, `decision`, `reason`, `tool_args` | After tool policy evaluation |
| `on_structured_validate` | Structured | `success`, `attempt`, `error` | After structured output validation |
| `on_provider_fallback` | Fallback | `failed_provider`, `next_provider`, `error` | FallbackProvider switches provider |
| `on_memory_trim` | Memory | `messages_removed`, `messages_remaining`, `reason` | Memory enforces limits |
| `on_session_load` | Session | `session_id`, `message_count` | Session loaded from store (v0.16.0) |
| `on_session_save` | Session | `session_id`, `message_count` | Session saved to store (v0.16.0) |
| `on_memory_summarize` | Memory | `summary`, `messages_summarized` | Trimmed messages summarized (v0.16.0) |
| `on_entity_extraction` | Memory | `entities`, `turn_count` | Entities extracted from turn (v0.16.0) |

*`on_batch_start`/`on_batch_end` use `batch_id` instead of `run_id`.

### Built-in LoggingObserver

Emits structured JSON events to Python's `logging` module:

```python
import logging
logging.basicConfig(level=logging.INFO)

agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(observers=[LoggingObserver()]),
)
```

Output:

```json
{"event": "run_start", "run_id": "a3f2...", "model": "gpt-4o-mini", "timestamp": 1708099200.0}
{"event": "llm_end", "run_id": "a3f2...", "tokens": 150, "duration_ms": 312.5}
{"event": "tool_end", "run_id": "a3f2...", "tool": "search", "duration_ms": 45.2}
```

### Observer vs Hooks

| Aspect | Hooks (`dict`) | AgentObserver |
|---|---|---|
| **Correlation** | Manual (closures, thread-local) | Built-in `run_id` + `call_id` |
| **Multiple consumers** | One callback per event | Multiple observers |
| **Event coverage** | 8 events | 28 events (including batch, fallback, retry, memory) |
| **Type safety** | Dict keys are strings | Protocol methods with signatures |
| **Use case** | Quick debugging, simple logging | Production observability (Langfuse, OTel, Datadog) |

Both systems work together — hooks and observers fire independently for the same events.

### AsyncAgentObserver

For async-native applications (FastAPI, aiohttp, async SQLAlchemy), `AsyncAgentObserver`
provides 25 async `a_on_*` methods that mirror the sync observer:

```python
from selectools import AsyncAgentObserver

class DBObserver(AsyncAgentObserver):
    blocking = True  # await inline — must complete before next tool

    async def a_on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
        await db.execute("INSERT INTO events ...")

class WebhookObserver(AsyncAgentObserver):
    blocking = False  # fire-and-forget via asyncio.ensure_future

    async def a_on_run_end(self, run_id, result):
        await httpx.post("https://hooks.example.com/...", json={...})

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(observers=[DBObserver(), WebhookObserver()]),
)
```

- **`blocking=True`**: Awaited inline — the agent loop waits for completion. Use for DB writes, rate limiting, result enrichment.
- **`blocking=False`** (default): Dispatched via `asyncio.ensure_future()`. Use for webhooks, logging, audit trails.

Async observers are called in `arun()` and `astream()` after each sync observer notification.
In sync `run()`, only sync observers fire.

### Trace Metadata & Nested Agents

```python
config = AgentConfig(
    parent_run_id="outer-agent-run-id",
    trace_metadata={"user_id": "u123", "environment": "production"},
    observers=[MyObserver()],
)
result = agent.run("classify this")
print(result.trace.parent_run_id)   # "outer-agent-run-id"
print(result.trace.metadata)         # {"user_id": "u123", "environment": "production"}

# Export as OpenTelemetry spans
spans = result.trace.to_otel_spans()
```

---

## Memory Integration

### Basic Memory

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

### Persistent Sessions

Auto-save and auto-load conversation state across process restarts using `session_store` and `session_id`:

```python
from selectools.sessions import JsonFileSessionStore

store = JsonFileSessionStore(directory="./sessions")
agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(session_store=store, session_id="user-123"),
)

# First run — auto-loads existing session (if any), auto-saves after
result = agent.run([Message(role=Role.USER, content="My name is Alice")])

# Later (even after restart) — session is restored automatically
result = agent.run([Message(role=Role.USER, content="What's my name?")])
# Agent knows: "Alice"
```

Three backends are available: `JsonFileSessionStore`, `SQLiteSessionStore`, `RedisSessionStore`. All support TTL-based expiry.

See [Sessions Module](SESSIONS.md) for backend details and TTL configuration.

### Summarize-on-Trim

When messages are trimmed by the sliding window, optionally generate a summary of the dropped messages and inject it as system context:

```python
agent = Agent(
    tools=[...], provider=provider,
    memory=ConversationMemory(max_messages=30),
    config=AgentConfig(
        summarize_on_trim=True,
        summarize_provider=provider,       # Provider for summarization
        summarize_model="gpt-4o-mini",     # Use a cheap/fast model
        summarize_max_tokens=150,          # Max tokens for the summary
    ),
)
```

**Flow:** When `_enforce_limits()` trims messages → the trimmed messages are sent to the summarize provider → a 2-3 sentence summary is generated → stored in `memory.summary` → injected as a system-level context message on subsequent turns.

See [Memory Module](MEMORY.md#summarize-on-trim) for implementation details.

### Entity Memory

Automatically extract named entities (people, organizations, projects, etc.) from each turn and inject them as context:

```python
from selectools import EntityMemory

entity_memory = EntityMemory(provider=provider)
agent = Agent(
    tools=[...], provider=provider, memory=memory,
    config=AgentConfig(entity_memory=entity_memory),
)

agent.run([Message(role=Role.USER, content="I'm working with Alice from Acme Corp")])
# Extracts: Alice (person, Acme Corp), Acme Corp (organization)
# Injected as [Known Entities] in system prompt on next turn
```

See [Entity Memory Module](ENTITY_MEMORY.md) for entity types, deduplication, and LRU pruning.

### Knowledge Graph Memory

Extract (subject, relation, object) triples from conversation and query them for context injection:

```python
from selectools import KnowledgeGraphMemory

kg = KnowledgeGraphMemory(provider=provider, storage="sqlite")
agent = Agent(
    tools=[...], provider=provider, memory=memory,
    config=AgentConfig(knowledge_graph=kg),
)

agent.run([Message(role=Role.USER, content="Alice manages Project Alpha")])
# Extracts: (Alice, manages, Project Alpha)
# Injected as [Known Relationships] in system prompt on next turn
```

See [Knowledge Graph Module](KNOWLEDGE_GRAPH.md) for storage backends and querying.

### Cross-Session Knowledge Memory

Persistent knowledge that survives across sessions — daily logs plus a long-term fact store:

```python
from selectools import KnowledgeMemory

knowledge = KnowledgeMemory(directory="./workspace", recent_days=2)
agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(knowledge_memory=knowledge),
)
# Auto-registers a `remember` tool — the agent can save facts explicitly
# [Long-term Memory] and [Recent Memory] injected into system prompt
```

See [Knowledge Memory Module](KNOWLEDGE.md) for daily logs, fact storage, and retention configuration.

### Context Injection Order

When multiple memory features are active, context is injected into the system prompt in this order:

```
1. [Conversation Summary]     ← summarize_on_trim
2. [Known Entities]           ← entity_memory
3. [Known Relationships]      ← knowledge_graph
4. [Long-term Memory]         ← knowledge_memory (persistent facts)
5. [Recent Memory]            ← knowledge_memory (daily logs)
```

Each section is only present when the corresponding feature is configured and has data.

---

## Streaming

### Agent.astream()

The `astream()` method provides token-by-token streaming with **full feature parity** with `run()` and `arun()` (as of v0.16.3). It supports `response_format`, `parent_run_id`, input/output guardrails, coherence checks, knowledge context injection, entity/KG extraction, session save, structured output validation, analytics, and verbose output.

```python
async for item in agent.astream([Message(role=Role.USER, content="Search for Python")]):
    if isinstance(item, StreamChunk):
        print(item.content, end="", flush=True)
    elif isinstance(item, AgentResult):
        print(f"\nDone in {item.iterations} iterations")
```

**Signature:**

```python
async def astream(
    messages: Union[str, List[Message]],
    response_format: Optional[ResponseFormat] = None,  # Structured output
    parent_run_id: Optional[str] = None,               # Trace linking
) -> AsyncGenerator[Union[StreamChunk, AgentResult], None]:
```

### How It Works

1. Shared `_prepare_run()` sets up trace, guardrails, memory, knowledge context (identical to run/arun)
2. Provider streams text deltas and tool call deltas via `astream()`
3. Text chunks are yielded as `StreamChunk` objects
4. Shared `_process_response()` applies output guardrails, parses tool calls, extracts reasoning
5. Tool calls are executed with coherence checks, output screening, analytics, and usage tracking
6. Shared `_finalize_run()` saves session, extracts entities/KG, builds full `AgentResult`
7. Final `AgentResult` is yielded (includes `parsed`, `reasoning`, `reasoning_history`, `provider_used`)

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

## Structured Output

### Overview

Pass a Pydantic `BaseModel` or dict JSON Schema as `response_format` to get typed, validated results from the LLM. The agent injects schema instructions into the system prompt, extracts JSON from the response, validates it, and retries on failure.

### Usage

```python
from pydantic import BaseModel
from typing import Literal

class Classification(BaseModel):
    intent: Literal["billing", "support", "sales", "cancel"]
    confidence: float
    priority: Literal["low", "medium", "high"]

result = agent.ask("I want to cancel my account", response_format=Classification)
print(result.parsed)  # Classification(intent="cancel", confidence=0.95, priority="high")
print(result.content)  # Raw JSON string
```

### How It Works

1. `build_schema_instruction(schema)` generates a prompt fragment describing the expected JSON shape
2. Schema instruction is appended to the system prompt for the duration of the run
3. LLM response is passed through `extract_json()` to isolate the JSON block
4. `parse_and_validate()` validates against the Pydantic model or JSON Schema
5. On validation failure, the error is fed back to the LLM for a retry
6. `result.parsed` contains the typed object; `result.content` has the raw string

### Supported Formats

- **Pydantic v2 `BaseModel`**: Full schema generation with type coercion
- **`dict` JSON Schema**: Raw JSON Schema for non-Pydantic users

### ResponseFormat Type

`ResponseFormat` is a type alias for what `response_format` accepts:

```python
from selectools import ResponseFormat  # Union[Type[Any], Dict[str, Any]]
```

It accepts either a Pydantic `BaseModel` subclass or a raw JSON Schema dict.

### Standalone Helpers

These utilities can be used independently for custom validation pipelines:

```python
from selectools.structured import (
    extract_json,
    schema_from_response_format,
    parse_and_validate,
    build_schema_instruction,
    validation_retry_message,
)
```

| Function | Description |
|---|---|
| `extract_json(text)` | Extract the first JSON object/array from text (handles code blocks, brace-balanced extraction). Returns `None` if no JSON found. |
| `schema_from_response_format(fmt)` | Convert a Pydantic model or dict to a JSON Schema dict. |
| `parse_and_validate(text, fmt)` | Extract JSON from text, validate against schema, return typed object. Raises `ValueError` on failure. |
| `build_schema_instruction(schema)` | Generate the system prompt fragment that instructs the LLM to produce JSON matching the schema. |
| `validation_retry_message(error)` | Generate the retry message sent to the LLM when validation fails. |

**Example — custom extraction pipeline:**

```python
from selectools.structured import extract_json, parse_and_validate
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    score: float

raw_text = 'Here is the analysis: ```json\n{"label": "positive", "score": 0.95}\n```'

json_str = extract_json(raw_text)     # '{"label": "positive", "score": 0.95}'
result = parse_and_validate(raw_text, Sentiment)  # Sentiment(label="positive", score=0.95)
```

### TraceStep Types for Structured Output

When structured validation fails, a `structured_retry` step appears in the trace:

```python
for step in result.trace:
    if step.type == "structured_retry":
        print(f"Validation failed: {step.error}")
```

---

## Execution Traces

### Overview

Every `run()` / `arun()` automatically produces an `AgentTrace` — a structured timeline of the entire execution. Access it via `result.trace`.

### Usage

```python
result = agent.run("Classify this ticket")

for step in result.trace:
    print(f"{step.type} | {step.duration_ms:.0f}ms | {step.summary}")

result.trace.to_json("trace.json")
print(result.trace.timeline())

llm_steps = result.trace.filter(type="llm_call")
total_llm_ms = sum(s.duration_ms for s in llm_steps)
```

### TraceStep Types

| Type | Description |
|---|---|
| `llm_call` | Provider API call with model, tokens, duration |
| `tool_selection` | LLM chose a tool (name, args, reasoning) |
| `tool_execution` | Tool was executed (name, result summary, duration) |
| `cache_hit` | Response served from cache |
| `error` | Error during execution |
| `structured_retry` | Structured output validation failed, retrying |
| `guardrail` | Input/output guardrail triggered (v0.15.0) |
| `coherence_check` | Coherence check blocked a tool call (v0.15.0) |
| `output_screening` | Tool output screening detected injection (v0.15.0) |
| `session_load` | Session loaded from store (v0.16.0) |
| `session_save` | Session saved to store (v0.16.0) |
| `memory_summarize` | Trimmed messages summarized (v0.16.0) |
| `entity_extraction` | Entities extracted from conversation (v0.16.0) |
| `kg_extraction` | Knowledge graph triples extracted (v0.16.0) |

### AgentTrace Methods

- `trace.to_dict()` — Serialize to dict
- `trace.to_json(filepath)` — Write JSON to file
- `trace.timeline()` — Human-readable timeline string
- `trace.filter(type=...)` — Filter steps by type
- `trace.total_duration_ms` — Total execution time

---

## Reasoning Visibility

### Overview

LLMs often return explanatory text alongside tool calls. This reasoning is now captured and surfaced on `AgentResult`.

### Usage

```python
result = agent.run("Route this customer request")

print(result.reasoning)
# "The customer is asking about billing charges, routing to billing_support"

for i, reasoning in enumerate(result.reasoning_history):
    print(f"Iteration {i}: {reasoning}")
```

### How It Works

The agent extracts text content from LLM responses that precede or accompany tool call decisions. No extra LLM calls are needed — it purely captures what providers already return but previously discarded.

- `result.reasoning` — reasoning text from the final tool selection
- `result.reasoning_history` — list of reasoning strings, one per iteration
- `step.reasoning` on `tool_selection` trace steps

---

## Provider Fallback

### Overview

`FallbackProvider` wraps multiple providers in priority order. If one fails, the next is tried automatically with circuit breaker protection.

### Usage

```python
from selectools import FallbackProvider, OpenAIProvider, AnthropicProvider

provider = FallbackProvider([
    OpenAIProvider(default_model="gpt-4o-mini"),
    AnthropicProvider(default_model="claude-haiku"),
])
agent = Agent(tools=[...], provider=provider)
```

### Circuit Breaker

After `max_failures` consecutive failures, a provider is skipped for `cooldown_seconds`:

```python
provider = FallbackProvider(
    providers=[openai, anthropic, local],
    max_failures=3,
    cooldown_seconds=60,
    on_fallback=lambda name, error: print(f"Skipping {name}: {error}"),
)
```

### Supported Methods

`FallbackProvider` implements the full `Provider` protocol: `complete()`, `acomplete()`, `stream()`, `astream()`.

---

## Batch Processing

### Overview

Process multiple prompts concurrently with configurable parallelism.

### Usage

```python
# Sync
results = agent.batch(
    ["Cancel my sub", "How do I upgrade?", "Payment failed"],
    max_concurrency=5,
)

# Async
results = await agent.abatch(
    ["Cancel my sub", "How do I upgrade?", "Payment failed"],
    max_concurrency=10,
)
```

### Guarantees

- Returns `list[AgentResult]` in same order as input
- Per-request error isolation (one failure doesn't cancel the batch)
- Respects `response_format` if provided
- `on_progress(completed, total)` callback for monitoring

---

## Tool Policy & Human-in-the-Loop

### Overview

Declarative allow/review/deny rules evaluated before every tool execution, with optional human approval for flagged tools.

### Tool Policy

```python
from selectools import ToolPolicy

policy = ToolPolicy(
    allow=["search_*", "read_*", "get_*"],
    review=["send_*", "create_*", "update_*"],
    deny=["delete_*", "drop_*"],
    deny_when=[{"tool": "send_email", "arg": "to", "pattern": "*@external.com"}],
)
config = AgentConfig(tool_policy=policy)
```

**Evaluation order**: `deny` → `review` → `allow` → unknown defaults to `review`.

### Human-in-the-Loop

```python
async def confirm(tool_name: str, tool_args: dict, reason: str) -> bool:
    return await get_user_approval(tool_name, tool_args)

config = AgentConfig(
    tool_policy=policy,
    confirm_action=confirm,
    approval_timeout=60,
)
```

**Agent loop behaviour:**

| Policy Decision | Behaviour |
|---|---|
| `allow` | Execute immediately |
| `review` + `confirm_action` | Call callback; execute if approved, deny if rejected |
| `review` + no callback | Deny with error message to LLM |
| `deny` | Return error to LLM, never execute |

---

## Terminal Actions

Some tools are "terminal" — the agent loop should stop after they execute, without making another LLM call.

**Static declaration** — tool author marks it at definition time:

```python
@tool(terminal=True)
def present_question(question_id: int) -> str:
    """Present a question card to the student."""
    return json.dumps({"action": "present_question", "id": question_id})
```

**Dynamic condition** — stop decision depends on the result content:

```python
config = AgentConfig(
    stop_condition=lambda tool_name, result: "present_question" in result,
)
```

After tool execution, the agent checks:
`tool.terminal or (config.stop_condition and config.stop_condition(tool_name, result))`

If true, the tool result becomes `AgentResult.content` and the loop exits immediately.
Works in `run()`, `arun()`, `astream()`, and parallel tool execution.

---

## Implementation Details

### Internal Architecture — Mixin Decomposition

The Agent class is composed from 4 internal mixins for maintainability:

| Mixin | File | Responsibility |
|-------|------|---------------|
| `_ToolExecutorMixin` | `agent/_tool_executor.py` | Tool execution pipeline, policy, coherence, parallel execution |
| `_ProviderCallerMixin` | `agent/_provider_caller.py` | LLM provider calls, caching, retry, streaming |
| `_LifecycleMixin` | `agent/_lifecycle.py` | Observer notification, fallback provider wiring |
| `_MemoryManagerMixin` | `agent/_memory_manager.py` | Memory operations, session persistence, entity/KG extraction |

All public methods remain on the `Agent` class — the mixins are internal implementation details.

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
- [Dynamic Tools Module](DYNAMIC_TOOLS.md) - Dynamic tool loading and runtime management
- [Parser Module](PARSER.md) - Tool call parsing details
- [Providers Module](PROVIDERS.md) - Provider implementations and FallbackProvider
- [Memory Module](MEMORY.md) - Conversation memory and tool-pair-aware trimming
- [Sessions Module](SESSIONS.md) - Persistent session storage with 3 backends
- [Entity Memory Module](ENTITY_MEMORY.md) - Named entity extraction and tracking
- [Knowledge Graph Module](KNOWLEDGE_GRAPH.md) - Relationship triple extraction
- [Knowledge Memory Module](KNOWLEDGE.md) - Cross-session durable memory
- [Usage Module](USAGE.md) - Cost tracking
- [Architecture](../ARCHITECTURE.md) - System-level overview including new modules

---

**Next Steps:** Understand how tools are defined and validated in the [Tools Module](TOOLS.md).
