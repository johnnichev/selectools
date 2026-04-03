---
description: "Runtime model switching based on task complexity"
tags:
  - runtime
  - models
---

# Model Switching Per Iteration

**Added in:** v0.17.4
**File:** `src/selectools/agent/config.py`, `src/selectools/agent/core.py`

## Overview

Model switching allows different LLM models on different iterations of the agent loop. Use cheap models for tool selection and expensive models for final synthesis — cutting costs without sacrificing output quality.

## Quick Start

```python
from selectools import Agent, AgentConfig

config = AgentConfig(
    model="claude-haiku-4-5",  # default (cheap)
    model_selector=lambda iteration, tool_calls, usage: (
        "claude-sonnet-4-6" if iteration > 3 else "claude-haiku-4-5"
    ),
)

agent = Agent(tools=[...], provider=provider, config=config)
result = agent.run("Research and write a report on AI agents")
# Iterations 1-3: Haiku (tool calls, data gathering)
# Iterations 4+: Sonnet (synthesis, writing)
```

## model_selector Callback

```python
def my_selector(
    iteration: int,           # current iteration (1-based)
    tool_calls: List[ToolCall],  # all tool calls so far
    usage: AgentUsage,        # cumulative token/cost usage
) -> str:
    """Return the model name to use for this iteration."""
    return "gpt-4o" if usage.total_cost_usd > 0.05 else "gpt-4o-mini"
```

The callback receives full context about the run so far, enabling strategies like:

- **Iteration-based**: cheap model early, expensive model late
- **Cost-based**: switch to cheap model when budget is getting low
- **Tool-based**: use a specific model after certain tools are called

## Observer Event

```python
from selectools import AgentObserver

class MyObserver(AgentObserver):
    def on_model_switch(self, run_id: str, iteration: int, old_model: str, new_model: str):
        print(f"Iteration {iteration}: {old_model} → {new_model}")
```

The event only fires when the model actually changes — returning the same model doesn't trigger it.

## Trace Verification

Each `LLM_CALL` trace step records the effective model:

```python
for step in result.trace.steps:
    if step.type == "llm_call":
        print(f"{step.model}: {step.prompt_tokens} prompt, ${step.cost_usd:.4f}")
```

## Reset Between Runs

The effective model resets to `config.model` at the start of each run via `_prepare_run()`. The selector is re-evaluated fresh for every run.

## See Also

- [Agent](AGENT.md) — `AgentConfig` reference
- [Models & Pricing](MODELS.md) — available models
- [Usage & Cost Tracking](USAGE.md) — per-iteration cost breakdown
