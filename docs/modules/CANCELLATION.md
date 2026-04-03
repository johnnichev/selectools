---
description: "Thread-safe cooperative cancellation with CancellationToken"
tags:
  - runtime
  - cancellation
---

# Agent Cancellation

**Added in:** v0.17.3
**File:** `src/selectools/cancellation.py`
**Classes:** `CancellationToken`

## Overview

The cancellation system provides cooperative stopping for running agents. A `CancellationToken` can be shared with the agent and cancelled from any thread — the agent stops at the next iteration boundary and returns a partial result.

## Quick Start

```python
import asyncio
from selectools import Agent, AgentConfig, CancellationToken

token = CancellationToken()

async def run_with_timeout():
    agent = Agent(tools=[...], provider=provider, config=AgentConfig(
        cancellation_token=token,
        max_iterations=20,
    ))

    # Cancel after 10 seconds from another task
    async def timeout():
        await asyncio.sleep(10)
        token.cancel()

    asyncio.create_task(timeout())
    result = await agent.arun("Long research task")

    if "cancelled" in result.content.lower():
        print("Agent was cancelled — partial results available")
    return result
```

## CancellationToken API

```python
token = CancellationToken()

token.is_cancelled   # False
token.cancel()       # Signal cancellation (thread-safe)
token.is_cancelled   # True
token.reset()        # Clear signal for reuse
token.is_cancelled   # False

token.raise_if_cancelled()  # Raises CancellationError if cancelled
```

The token uses `threading.Event` internally — safe to call `.cancel()` from any thread (UI handlers, supervisor agents, timeout managers).

## Check Points

The agent checks `is_cancelled` at two points per iteration:

1. **Start of iteration** — before the LLM call
2. **After tool execution** — before continuing to the next iteration

This means cancellation latency is bounded by the duration of one LLM call + one tool execution.

## Observer Event

```python
from selectools import AgentObserver

class MyObserver(AgentObserver):
    def on_cancelled(self, run_id: str, iteration: int, reason: str):
        print(f"Run {run_id} cancelled at iteration {iteration}")
```

## Trace Step

Cancellations are recorded as `StepType.CANCELLED`:

```python
cancelled_steps = [s for s in result.trace.steps if s.type == "cancelled"]
```

## Token Reuse

A single token can be reused across multiple runs by calling `reset()`:

```python
token = CancellationToken()

result1 = agent.run("task 1")  # completes normally
token.cancel()
result2 = agent.run("task 2")  # cancelled immediately
token.reset()
result3 = agent.run("task 3")  # completes normally
```

## See Also

- [Token Budget](BUDGET.md) — cost-based stopping
- [Agent](AGENT.md) — `AgentConfig` reference
