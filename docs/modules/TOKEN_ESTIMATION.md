---
description: "Pre-run token and cost estimation without calling the LLM"
tags:
  - runtime
  - estimation
---

# Token Estimation

**Added in:** v0.17.4
**File:** `src/selectools/token_estimation.py`
**Functions:** `estimate_tokens`, `estimate_run_tokens`
**Classes:** `TokenEstimate`

## Overview

The token estimation module provides approximate token counts before calling the LLM. Use it for budget pre-checks, memory trimming decisions, and model selection.

## Quick Start

```python
from selectools import estimate_tokens, estimate_run_tokens
from selectools.tools import Tool

# Single string estimation
tokens = estimate_tokens("Hello, how can I help you today?")
print(f"~{tokens} tokens")

# Full run estimation with breakdown
tools = [my_search_tool, my_calculator_tool]
messages = [Message(role=Role.USER, content="Find the revenue for Q4")]

estimate = estimate_run_tokens(
    messages=messages,
    tools=tools,
    system_prompt="You are a financial analyst.",
    model="gpt-4o",
)

print(f"System prompt: {estimate.system_tokens} tokens")
print(f"Messages:      {estimate.message_tokens} tokens")
print(f"Tool schemas:  {estimate.tool_schema_tokens} tokens")
print(f"Total:         {estimate.total_tokens} tokens")
print(f"Context window: {estimate.context_window}")
print(f"Remaining:     {estimate.remaining_tokens} tokens")
print(f"Method:        {estimate.method}")  # "tiktoken" or "heuristic"
```

## Estimation Methods

| Method | When Used | Accuracy |
|--------|-----------|----------|
| **tiktoken** | OpenAI models when `tiktoken` is installed | Exact |
| **heuristic** | All other cases (`len(text) // 4`) | ~85% for English |

Install tiktoken for exact OpenAI counts: `pip install tiktoken`

## TokenEstimate Fields

| Field | Type | Description |
|-------|------|-------------|
| `system_tokens` | `int` | Tokens in the system prompt |
| `message_tokens` | `int` | Tokens in conversation messages (+ ~4 per message overhead) |
| `tool_schema_tokens` | `int` | Tokens in tool JSON schemas |
| `total_tokens` | `int` | Sum of the above |
| `context_window` | `int` | Model's context window (from model registry, 0 if unknown) |
| `remaining_tokens` | `int` | `context_window - total_tokens` |
| `model` | `str` | Model used for the estimate |
| `method` | `str` | `"tiktoken"` or `"heuristic"` |

## Budget Pre-Check Pattern

```python
from selectools import estimate_run_tokens, AgentConfig

estimate = estimate_run_tokens(messages, tools, system_prompt, model="gpt-4o")

if estimate.total_tokens > budget_remaining:
    raise Exception(f"Would use ~{estimate.total_tokens} tokens, budget is {budget_remaining}")

# Safe to proceed
config = AgentConfig(max_total_tokens=budget_remaining)
result = agent.run(message)
```

## See Also

- [Token Budget](BUDGET.md) — enforce limits during execution
- [Usage & Cost Tracking](USAGE.md) — actual token counts after execution
- [Models & Pricing](MODELS.md) — model registry with context windows
