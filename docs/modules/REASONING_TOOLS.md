# Reasoning Tools

Turn reasoning into explicit, bounded, **inspectable tool calls**.

[Reasoning *strategies*](REASONING_STRATEGIES.md) (`PromptBuilder(reasoning_strategy="react")`)
nudge the model to think a certain way, but the thinking stays hidden inside the
model's output. **Reasoning tools** make each reasoning step a `think` / `analyze`
tool call instead — so the chain shows up as structured steps in the trace and is
bounded by explicit `min_steps` / `max_steps`. The two compose: a strategy shapes
*how* the model reasons; the tools make that reasoning *visible and bounded*.

Marked `@beta`. Lives in `selectools.toolbox.reasoning_tools`.

## Quick start

```python
from selectools import Agent
from selectools.toolbox.reasoning_tools import make_reasoning_tools

agent = Agent(
    tools=[*my_tools, *make_reasoning_tools(min_steps=1, max_steps=8)],
    provider=provider,
)
agent.run("Plan and execute the migration.")
```

The agent now has a `think` tool (a scratchpad for one reasoning step) and an
`analyze` tool (evaluate a result and decide the next step). Both are plain tools
that return an acknowledgement — they call no external system; their value is that
the reasoning becomes part of the conversation as discrete, inspectable steps.

## Inspecting the chain

Hold a `ReasoningTools` instance to read the recorded steps after a run:

```python
from selectools.toolbox.reasoning_tools import ReasoningTools

reasoning = ReasoningTools(min_steps=2, max_steps=6)
agent = Agent(tools=[*my_tools, *reasoning.tools], provider=provider)
agent.run("...")

for step in reasoning.steps:        # list[ReasoningStep]
    print(step.index, step.kind, step.content)

reasoning.reset()                   # reuse the instance for another run
```

Both `think` and `analyze` count against the **same** budget, so
`reasoning.count` is the total number of reasoning steps taken.

## Bounds

| Bound | Behavior |
|---|---|
| `max_steps` (default 10) | **Enforced.** Once reached, further `think`/`analyze` calls are not recorded and return a message telling the agent to stop reasoning and answer — a real guard against reasoning loops. Pass `None` for unbounded. |
| `min_steps` (default 1) | **Guidance.** Advertised in the tool descriptions; each call reports how many more steps are expected. A model cannot be forced to call a tool, so the floor is a nudge, not a hard gate. |

```python
make_reasoning_tools(min_steps=0, max_steps=None)  # optional, unbounded
make_reasoning_tools(min_steps=3, max_steps=12)     # think hard, but cap it
```

Invalid bounds (`min_steps < 0`, `max_steps < 1`, `max_steps < min_steps`) raise
`ValueError` at construction.

## API

| Symbol | Description |
|---|---|
| `make_reasoning_tools(min_steps=1, max_steps=10) -> list[Tool]` | Fresh `think` + `analyze` tools backed by a new bounded log. |
| `ReasoningTools(min_steps=1, max_steps=10)` | Holds the log; `.tools`, `.think_tool()`, `.analyze_tool()`, `.steps`, `.count`, `.reset()`. |
| `ReasoningStep` | `index` (1-based), `kind` (`"think"`/`"analyze"`), `content`. |

## When to use which

- **Reasoning strategy** (prompt) — lightweight, zero tool overhead; good default.
- **Reasoning tools** — when you want the chain *recorded* (for evals, debugging,
  audit), or want a hard cap on reasoning effort. Use both together for shaped,
  visible, bounded reasoning.
