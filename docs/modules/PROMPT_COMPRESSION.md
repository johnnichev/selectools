---
description: "Automatic prompt compression to fit within context windows"
tags:
  - runtime
  - compression
---

# Prompt Compression

**Added in:** v0.17.7
**Files:** `src/selectools/agent/config.py`, `src/selectools/agent/_memory_manager.py`, `src/selectools/trace.py`, `src/selectools/observer.py`

## Overview

Prompt compression prevents context-window overflow in long conversations by **proactively summarising older messages** before they crowd out new context.

When enabled, selectools estimates the token count before each LLM call.  If the fill-rate exceeds the configured threshold, older messages are summarised into a single `[Compressed context]` system message.  The most recent turns are always kept verbatim.

```python
from selectools import Agent, AgentConfig

agent = Agent(
    tools=[...],
    config=AgentConfig(
        compress_context=True,    # enable compression
        compress_threshold=0.75,  # trigger at 75 % context fill
        compress_keep_recent=4,   # always keep last 4 turns verbatim
    ),
)
```

## Quick Start

```python
from selectools import Agent, AgentConfig, Message, Role
from selectools.memory import ConversationMemory

memory = ConversationMemory(max_messages=100)
agent = Agent(
    tools=[...],
    provider=provider,
    memory=memory,
    config=AgentConfig(
        model="gpt-4o-mini",
        compress_context=True,
        compress_threshold=0.75,
    ),
)

# Run many turns — compression fires automatically when context fills up
for i in range(50):
    result = agent.run(f"Tell me about topic {i}.")

compressed_steps = [s for s in result.trace.steps if s.type.value == "prompt_compressed"]
print(f"Compressed {len(compressed_steps)} time(s)")
```

## Configuration

All three fields are on `AgentConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compress_context` | `bool` | `False` | Enable proactive context compression. |
| `compress_threshold` | `float` | `0.75` | Fill-rate trigger. When `total_tokens / context_window ≥ threshold`, compression fires. |
| `compress_keep_recent` | `int` | `4` | Number of most-recent turns to keep verbatim. Each "turn" = 1 user + 1 assistant message. |

## How It Works

Compression runs in `_prepare_run()`, before the LLM call, on every agent iteration:

1. **Estimate tokens** — `estimate_run_tokens(history, tools, system_prompt, model)` returns a `TokenEstimate`.
2. **Check fill-rate** — `fill_rate = total_tokens / context_window`. If below threshold, do nothing.
3. **Split history** — separate `SYSTEM` messages from user/assistant turns; identify "old" turns vs. the `compress_keep_recent` most-recent turns.
4. **Guard** — if fewer than 2 messages are eligible for compression, skip.
5. **Summarise** — call the provider with a summarisation prompt asking for 3–5 sentences.
6. **Replace history** — `self._history = system_msgs + [Message(SYSTEM, "[Compressed context] …")] + recent_turns`
7. **Record** — append `TraceStep(PROMPT_COMPRESSED, …)` and fire `on_prompt_compressed` observer event.

!!! important "Memory is never modified"
    Compression only modifies `self._history` — the per-call working view.  `self.memory` (the persistent store) is **never touched**.  After the run, the full uncompressed conversation is still in memory.

## Trace Step

Every compression event adds a `PROMPT_COMPRESSED` step to `AgentTrace`:

```python
result = agent.run("What have we covered?")
for step in result.trace.steps:
    if step.type.value == "prompt_compressed":
        print(step.summary)
        # "Compressed 8 messages: 95000→3200 tokens"
        print(f"Before: {step.prompt_tokens:,} tokens")
        print(f"After:  {step.completion_tokens:,} tokens")
```

## Observer Event

```python
from selectools.observer import AgentObserver

class MyObserver(AgentObserver):
    def on_prompt_compressed(
        self,
        run_id: str,
        before_tokens: int,
        after_tokens: int,
        messages_compressed: int,
    ) -> None:
        reduction = 1 - after_tokens / before_tokens
        print(
            f"[{run_id[:8]}] Compressed {messages_compressed} messages, "
            f"{reduction:.0%} token reduction "
            f"({before_tokens:,} → {after_tokens:,})"
        )

agent = Agent(
    tools=[...],
    config=AgentConfig(
        compress_context=True,
        observers=[MyObserver()],
    ),
)
```

## Error Handling

If the summarisation LLM call raises an exception, compression is **silently skipped** — the agent continues normally with the full (uncompressed) history.  This prevents a transient LLM failure from crashing a long-running conversation.

## Threshold Guide

| Threshold | When compression fires |
|-----------|----------------------|
| `0.90` | Very late — context almost full before compression |
| `0.75` | Default — reasonable headroom for the reply |
| `0.60` | Aggressive — compresses frequently, keeping tokens low |

## Example

See [`examples/53_prompt_compression.py`](https://github.com/johnnichev/selectools/blob/main/examples/53_prompt_compression.py) for a runnable demo including observer integration and the trace step structure.
