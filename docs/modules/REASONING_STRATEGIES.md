---
description: "ReAct, Chain-of-Thought, and Plan-Then-Act reasoning modes"
tags:
  - runtime
  - reasoning
---

# Reasoning Strategies

**Added in:** v0.17.6
**File:** `src/selectools/prompt.py`, `src/selectools/agent/config.py`
**Exports:** `REASONING_STRATEGIES`

## Overview

Reasoning strategies inject structured instructions into the agent's system prompt, guiding the LLM to follow a specific reasoning pattern. Combined with `result.reasoning`, you get full visibility into the agent's thought process.

## Quick Start

```python
from selectools import Agent, AgentConfig

agent = Agent(
    tools=[...],
    config=AgentConfig(reasoning_strategy="react"),
)
result = agent.run("What's the weather in NYC and how should I dress?")
print(result.reasoning)  # Shows Thought → Action → Observation steps
```

## Available Strategies

### ReAct (Reason + Act)

```python
config = AgentConfig(reasoning_strategy="react")
```

Follows the **Thought → Action → Observation** cycle from the [ReAct paper](https://arxiv.org/abs/2210.03629). The agent explains its reasoning before each tool call and analyzes results before proceeding.

Best for: multi-step tool use where you want to see the agent's decision-making process.

### Chain of Thought (CoT)

```python
config = AgentConfig(reasoning_strategy="cot")
```

The agent writes out its complete chain of reasoning step by step before taking action. Breaks complex problems into smaller steps.

Best for: complex reasoning tasks, math, logic puzzles, multi-part questions.

### Plan Then Act

```python
config = AgentConfig(reasoning_strategy="plan_then_act")
```

The agent first creates a numbered plan of all steps needed, then executes each step in order. Revises the plan if a step fails.

Best for: multi-tool workflows where upfront planning improves execution.

## API Reference

### AgentConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_strategy` | `Optional[str]` | `None` | Strategy name: `"react"`, `"cot"`, `"plan_then_act"`, or `None` |

### REASONING_STRATEGIES

```python
from selectools import REASONING_STRATEGIES

print(REASONING_STRATEGIES.keys())
# dict_keys(['react', 'cot', 'plan_then_act'])
```

A dict mapping strategy names to their instruction text. Useful for discovering available strategies or customizing prompts.

### Custom PromptBuilder

You can also set the strategy directly on `PromptBuilder`:

```python
from selectools import PromptBuilder

pb = PromptBuilder(reasoning_strategy="cot")
agent = Agent(tools=[...], prompt_builder=pb)
```

!!! tip
    When a custom `prompt_builder` is passed to the Agent, it takes precedence over `config.reasoning_strategy`.

## See Also

- [Agent](AGENT.md) — main agent configuration
- [Prompt](PROMPT.md) — prompt builder internals
