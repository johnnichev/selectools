---
description: "Token and cost budgets with automatic agent stopping"
tags:
  - runtime
  - budget
---

# Token Budget & Cost Limits

**Added in:** v0.17.3
**File:** `src/selectools/agent/config.py`, `src/selectools/agent/core.py`

## Overview

The budget system prevents runaway costs by enforcing hard limits on token usage and dollar spend per agent run. When a limit is hit, the agent stops gracefully and returns a partial result.

## Quick Start

```python
from selectools import Agent, AgentConfig

config = AgentConfig(
    max_total_tokens=50000,  # stop after 50k cumulative tokens
    max_cost_usd=0.20,       # stop after $0.20 cumulative cost
    max_iterations=12,       # existing iteration limit still applies
)

agent = Agent(tools=[...], provider=provider, config=config)
result = agent.run("Analyze this dataset")

# Check if budget was the reason for stopping
if "budget exceeded" in result.content.lower():
    print(f"Stopped early — used {result.usage.total_tokens} tokens, ${result.usage.total_cost_usd:.4f}")
```

## How It Works

The budget check runs at the **start of each iteration**, after the previous iteration's tokens have been counted. If cumulative usage exceeds either limit:

1. A `BUDGET_EXCEEDED` trace step is recorded
2. The `on_budget_exceeded` observer event fires
3. The agent returns an `AgentResult` with partial content from completed iterations

```
Iteration 1: 15,000 tokens → total: 15,000 (under 50,000) ✓
Iteration 2: 20,000 tokens → total: 35,000 (under 50,000) ✓
Iteration 3: budget check → 35,000 < 50,000 → continue
             18,000 tokens → total: 53,000
Iteration 4: budget check → 53,000 ≥ 50,000 → STOP
```

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_total_tokens` | `Optional[int]` | `None` | Cumulative token limit. `None` = no limit. |
| `max_cost_usd` | `Optional[float]` | `None` | Cumulative cost limit in USD. `None` = no limit. |

Both fields are `None` by default, preserving backward compatibility.

## Observer Event

```python
from selectools import AgentObserver

class MyObserver(AgentObserver):
    def on_budget_exceeded(self, run_id: str, reason: str, tokens_used: int, cost_used: float):
        log.warning(f"Budget exceeded: {reason} (tokens={tokens_used}, cost=${cost_used:.4f})")
```

## Trace Step

Budget stops are recorded as `StepType.BUDGET_EXCEEDED` in the execution trace:

```python
for step in result.trace.steps:
    if step.type == "budget_exceeded":
        print(step.summary)  # "Token budget exceeded: 53000/50000 tokens"
```

## Interaction with Other Limits

If both `max_iterations` and `max_total_tokens` are set, whichever limit is hit first wins. Budget is checked before the LLM call, so no tokens are wasted on a call that would exceed the budget.

## See Also

- [Usage & Cost Tracking](USAGE.md) — per-call and per-run token/cost tracking
- [Agent](AGENT.md) — `AgentConfig` reference
