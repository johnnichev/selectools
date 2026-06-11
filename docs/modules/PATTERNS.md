---
description: "PlanAndExecute, Reflective, Debate, TeamLead agent coordination patterns"
tags:
  - multi-agent
  - patterns
---

# Advanced Agent Patterns

**Import:** `from selectools.patterns import PlanAndExecuteAgent`

**Stability:** beta

```python title="patterns_quickstart.py"
from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider
from selectools.patterns import ReflectiveAgent

@tool(description="No-op tool")
def noop(x: str) -> str:
    return x

provider = LocalProvider()

# Actor drafts content, critic evaluates and requests revisions
actor = Agent(
    tools=[noop],
    provider=provider,
    config=AgentConfig(max_iterations=1),
    system_prompt="You are a technical writer.",
)
critic = Agent(
    tools=[noop],
    provider=provider,
    config=AgentConfig(max_iterations=1),
    system_prompt="You are an editor. Say 'approved' when satisfied.",
)

agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=2)
result = agent.run("Explain what a vector database is in two sentences")

print(f"Final draft: {result.final_draft[:200]}")
print(f"Approved: {result.approved}")
print(f"Rounds: {result.total_rounds}")
```

!!! tip "See Also"
    - [Orchestration](ORCHESTRATION.md) - AgentGraph routing, parallel execution, and HITL
    - [Supervisor](SUPERVISOR.md) - SupervisorAgent with 4 built-in strategies

---

**Added in:** v0.19.1
**Module:** `src/selectools/patterns/`
**Import:** `from selectools.patterns import ...`  or  `from selectools import ...`

Four production-ready multi-agent coordination patterns built on the v0.18.0 orchestration primitives. Each pattern wires up the AgentGraph topology for you — no graph-wiring required.

## Pattern Overview

| Pattern | When to use | Key concept |
|---------|-------------|-------------|
| `PlanAndExecuteAgent` | Multi-step tasks with distinct specialist roles | Planner generates typed `PlanStep` list; executors run sequentially with context chaining |
| `ReflectiveAgent` | Quality-critical output (writing, code, analysis) | Actor drafts, critic evaluates, actor revises until approved |
| `DebateAgent` | Decisions needing multiple perspectives | N agents argue positions; judge synthesizes conclusion |
| `TeamLeadAgent` | Large tasks that can be decomposed into parallel work | Lead delegates subtasks; team executes sequentially, in parallel, or dynamically |

All patterns support `.run()` (sync) and `.arun()` (async).

---

## PlanAndExecuteAgent

```python
from selectools import Agent, OpenAIProvider
from selectools.patterns import PlanAndExecuteAgent

provider = OpenAIProvider()
planner = Agent(provider=provider, system_prompt="You are a planning agent.")
researcher = Agent(provider=provider, system_prompt="You are a research agent.")
writer = Agent(provider=provider, system_prompt="You are a writing agent.")

agent = PlanAndExecuteAgent(
    planner=planner,
    executors={"researcher": researcher, "writer": writer},
)
result = agent.run("Research LLM safety and write a 500-word blog post")
print(result.content)
```

### How it works

1. The **planner** agent is called once to produce a JSON execution plan:
   ```json
   [
     {"executor": "researcher", "task": "Find 3 key LLM safety concerns"},
     {"executor": "writer",     "task": "Write a blog post using the research"}
   ]
   ```
2. Each step's executor is called in sequence. Each step receives the accumulated output of previous steps as context.
3. The final executor's output becomes `result.content`.

### Replanning on failure

```python
agent = PlanAndExecuteAgent(
    planner=planner,
    executors={"researcher": researcher, "writer": writer},
    replanner=True,          # re-call planner if a step fails
    max_replan_attempts=2,   # limit replanning cycles
)
```

If a step raises an exception and `replanner=True`, the planner is re-called with the failure context to revise the remaining steps.

### Result type

`PlanAndExecuteAgent.run()` returns a `GraphResult`:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Aggregated output from all executor steps |
| `state` | `GraphState` | Final graph state |
| `node_results` | `dict` | Per-step `AgentResult` objects keyed by step name |
| `trace` | `AgentTrace` | Execution trace |

### Constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planner` | `Agent` | required | Agent that generates the execution plan |
| `executors` | `Dict[str, Agent]` | required | Name → Agent mapping (at least one required) |
| `replanner` | `bool` | `False` | Re-call planner on step failure |
| `max_replan_attempts` | `int` | `2` | Max replanning cycles |
| `observers` | `List[AgentObserver]` | `[]` | Observer instances |
| `cancellation_token` | `CancellationToken` | `None` | Cooperative cancellation |
| `max_cost_usd` | `float` | `None` | Cost budget (informational) |

### Planning-as-config (beta)

Any `Agent` can opt into the plan → (approve) → execute → synthesize flow
without wiring `PlanAndExecuteAgent` manually — set
`AgentConfig(planning=PlanningConfig(...))`:

```python
from selectools import Agent, AgentConfig, PlanningConfig

config = AgentConfig(
    planning=PlanningConfig(
        enabled=True,
        provider=None,      # planner provider override (defaults to the agent's)
        model=None,         # planner model override (defaults to the agent's)
        auto_approve=True,  # False requires plan_approval_handler
        reasoning=True,     # surface the plan via result.reasoning
    )
)
agent = Agent(tools, provider=provider, config=config)
result = agent.run("Research X, then write a summary, and finally review it.")
print(result.reasoning)                      # the generated plan
print(result.trace.metadata["planning"])     # plan + steps_executed
```

Internally the agent clones itself into a planner and a single executor and
delegates to `PlanAndExecuteAgent`, then runs one final synthesis call. The
result is a normal `AgentResult` with usage aggregated across the planner,
every step, and the synthesis call.

**Complexity gate.** Simple single-step inputs skip planning entirely. A
cheap local heuristic scores the prompt (sequence connectives like "then" /
"finally", numbered or bulleted lists, 3+ sentences, semicolons, length over
~120 estimated tokens); planning triggers when the score reaches
`min_complexity` (default `2`). Set `always=True` to plan every input.

**Plan approval.** With `auto_approve=False`, `plan_approval_handler` is
required. It receives the structured plan (`List[PlanStep]`) and returns
`True` (approve), `False` (reject — the agent falls back to a standard run
with a one-time warning), or an edited `List[PlanStep]`.

**Interplay.** Streaming runs (`astream()`, or `run()` with a
`stream_handler`) skip planning with a one-time `UserWarning` per agent.
Structured output works: `response_format` is applied to the final synthesis
call. `enabled=False` (or leaving `planning` unset) is a zero-overhead no-op.

**Known gap.** `result.trace` is the synthesis run's trace (the plan is
attached under `trace.metadata["planning"]`); per-step traces are not merged
because `PlanAndExecuteAgent` does not aggregate traces.

| `PlanningConfig` field | Type | Default | Description |
|------------------------|------|---------|-------------|
| `enabled` | `bool` | `False` | Master switch |
| `provider` | `Provider` | `None` | Planner-call provider override |
| `model` | `str` | `None` | Planner-call model override |
| `auto_approve` | `bool` | `True` | Execute plans without approval |
| `plan_approval_handler` | `Callable` | `None` | Required when `auto_approve=False` |
| `reasoning` | `bool` | `True` | Put the plan in `result.reasoning` |
| `always` | `bool` | `False` | Bypass the complexity gate |
| `min_complexity` | `int` | `2` | Heuristic score needed to trigger planning |

See `examples/109_planning_as_config.py` for a fully offline demo.

---

## ReflectiveAgent

```python
from selectools.patterns import ReflectiveAgent

actor = Agent(provider=provider, system_prompt="You are a technical writer.")
critic = Agent(provider=provider, system_prompt="You are an editor. Give feedback. Say 'approved' when satisfied.")

agent = ReflectiveAgent(actor=actor, critic=critic, max_reflections=3)
result = agent.run("Write a concise explanation of transformer attention")

print(result.final_draft)   # final approved draft
print(result.approved)      # True if critic said "approved"
print(result.total_rounds)  # number of actor-critic cycles
```

### How it works

Each round:
1. **Actor** receives the task (round 0) or the task + previous draft + critique (round N).
2. **Critic** evaluates the draft and provides feedback.
3. If the critic's response contains `stop_condition` (default: `"approved"`), the loop ends.

The loop also ends when `max_reflections` is reached regardless of approval.

### Per-round records

```python
for rnd in result.rounds:
    print(f"Round {rnd.round_number}: approved={rnd.approved}")
    print(f"  Draft:   {rnd.draft[:100]}...")
    print(f"  Critique: {rnd.critique[:100]}...")
```

### Result type — `ReflectiveResult`

| Field | Type | Description |
|-------|------|-------------|
| `final_draft` | `str` | Actor's last output |
| `rounds` | `List[ReflectionRound]` | Per-round records |
| `approved` | `bool` | True if critic triggered stop condition |
| `total_rounds` | `int` (property) | `len(rounds)` |

### Constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor` | `Agent` | required | Agent that produces drafts |
| `critic` | `Agent` | required | Agent that evaluates drafts |
| `max_reflections` | `int` | `3` | Maximum actor-critic rounds |
| `stop_condition` | `str` | `"approved"` | Word in critic output that ends the loop (case-insensitive) |
| `observers` | `List[AgentObserver]` | `[]` | Observer instances |
| `cancellation_token` | `CancellationToken` | `None` | Cooperative cancellation |

---

## DebateAgent

```python
from selectools.patterns import DebateAgent

optimist = Agent(provider=provider, system_prompt="You argue in favour of the proposal.")
skeptic  = Agent(provider=provider, system_prompt="You argue against the proposal.")
judge    = Agent(provider=provider, system_prompt="You synthesize debate arguments objectively.")

agent = DebateAgent(
    agents={"optimist": optimist, "skeptic": skeptic},
    judge=judge,
    max_rounds=2,
)
result = agent.run("Should we rewrite our monolith as microservices?")

print(result.conclusion)      # judge's synthesis
print(result.total_rounds)    # 2

for rnd in result.rounds:
    for position, argument in rnd.arguments.items():
        print(f"[{position}] {argument[:200]}")
```

### How it works

1. Each debate round: every agent is called in order. Rounds 2+ include the prior round's transcript so agents can respond to each other.
2. After all rounds, the **judge** receives the full transcript and synthesizes a conclusion.

!!! tip
    Use 2–3 rounds for most decisions. More rounds increase cost without proportional quality improvement.

### Result type — `DebateResult`

| Field | Type | Description |
|-------|------|-------------|
| `conclusion` | `str` | Judge's synthesized conclusion |
| `rounds` | `List[DebateRound]` | Per-round argument records |
| `total_rounds` | `int` (property) | `len(rounds)` |

**`DebateRound`:**

| Field | Type | Description |
|-------|------|-------------|
| `round_number` | `int` | 0-indexed round |
| `arguments` | `Dict[str, str]` | position name → argument text |

### Constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `Dict[str, Agent]` | required | Position name → Agent (minimum 2) |
| `judge` | `Agent` | required | Agent that synthesizes the conclusion |
| `max_rounds` | `int` | `3` | Number of debate rounds |
| `observers` | `List[AgentObserver]` | `[]` | Observer instances |
| `cancellation_token` | `CancellationToken` | `None` | Cooperative cancellation |

---

## TeamLeadAgent

```python
from selectools.patterns import TeamLeadAgent

lead       = Agent(provider=provider, system_prompt="You are a project lead.")
researcher = Agent(provider=provider, system_prompt="You find and summarize information.")
writer     = Agent(provider=provider, system_prompt="You write clear, concise reports.")

# Sequential — subtasks run one after another, each sees prior results
agent = TeamLeadAgent(lead=lead, team={"researcher": researcher, "writer": writer},
                      delegation_strategy="sequential")

# Parallel — subtasks run simultaneously via AgentGraph fan-out
agent = TeamLeadAgent(lead=lead, team={"researcher": researcher, "writer": writer},
                      delegation_strategy="parallel")

# Dynamic (default) — lead reviews after each result and may reassign
agent = TeamLeadAgent(lead=lead, team={"researcher": researcher, "writer": writer},
                      delegation_strategy="dynamic", max_reassignments=2)

result = agent.run("Produce a competitive analysis of the top 3 LLM frameworks")
print(result.content)
print(result.total_assignments)  # total task executions including reassignments
```

### Delegation strategies

| Strategy | Execution | Best for |
|----------|-----------|----------|
| `sequential` | One subtask at a time; each step sees prior outputs as context | Ordered pipelines where step N needs step N-1's output |
| `parallel` | All subtasks run simultaneously via `AgentGraph` fan-out | Independent tasks with no data dependencies |
| `dynamic` | Lead reviews progress after each result; may add/reassign work | Open-ended tasks where the plan may need to adapt |

### How the lead delegates

The lead agent generates a JSON subtask plan:
```json
[
  {"assignee": "researcher", "task": "Find the top 3 LLM frameworks"},
  {"assignee": "writer",     "task": "Write the competitive analysis"}
]
```

In **dynamic** mode, after all pending subtasks complete, the lead reviews the work log and decides whether to synthesize or reassign:
```json
{
  "complete": false,
  "reassignments": [{"assignee": "researcher", "task": "Also compare pricing models"}],
  "synthesis": ""
}
```

### Result type — `TeamLeadResult`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Final synthesized output from the lead |
| `subtasks` | `List[Subtask]` | All subtask records including reassignments |
| `total_assignments` | `int` (property) | Sum of `subtask.attempt` across all subtasks |

**`Subtask`:**

| Field | Type | Description |
|-------|------|-------------|
| `assignee` | `str` | Team member name |
| `task` | `str` | Task description |
| `result` | `Optional[str]` | Execution output |
| `status` | `str` | `"pending"` / `"done"` / `"reassigned"` |
| `attempt` | `int` | How many times this subtask was executed |

### Constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lead` | `Agent` | required | Agent that plans, reviews, and synthesizes |
| `team` | `Dict[str, Agent]` | required | Member name → Agent (at least one required) |
| `delegation_strategy` | `str` | `"dynamic"` | `"sequential"`, `"parallel"`, or `"dynamic"` |
| `max_reassignments` | `int` | `2` | Maximum reassignment cycles (dynamic only) |
| `observers` | `List[AgentObserver]` | `[]` | Observer instances |
| `cancellation_token` | `CancellationToken` | `None` | Cooperative cancellation |
| `max_cost_usd` | `float` | `None` | Cost budget (informational) |

---

## Async Usage

All patterns support `await agent.arun(prompt)`:

```python
import asyncio

async def main():
    result = await agent.arun("Write a technical blog post about vector databases")
    print(result.content)

asyncio.run(main())
```

---

## Choosing a Pattern

```
Need typed step-by-step execution with named specialists?
  → PlanAndExecuteAgent

Need iterative quality improvement with self-critique?
  → ReflectiveAgent

Need to explore a decision from multiple viewpoints?
  → DebateAgent

Need to decompose a large task across a team?
  → TeamLeadAgent (parallel for speed, dynamic for adaptability)
```

---

## See Also

- [Orchestration](ORCHESTRATION.md) — `AgentGraph`, routing, parallel execution, HITL
- [Supervisor](SUPERVISOR.md) — `SupervisorAgent` with 4 built-in strategies
- [Pipeline](PIPELINE.md) — Composable pipelines with `@step` and `|` operator
- **Examples**: `70_plan_and_execute.py`, `71_reflective_agent.py`, `72_debate_agent.py`, `73_team_lead_agent.py`

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 70 | [`70_plan_and_execute.py`](https://github.com/johnnichev/selectools/blob/main/examples/70_plan_and_execute.py) | PlanAndExecuteAgent with planner and specialist executors |
| 71 | [`71_reflective_agent.py`](https://github.com/johnnichev/selectools/blob/main/examples/71_reflective_agent.py) | ReflectiveAgent with actor-critic revision loop |
| 72 | [`72_debate_agent.py`](https://github.com/johnnichev/selectools/blob/main/examples/72_debate_agent.py) | DebateAgent with multiple perspectives and judge synthesis |
| 73 | [`73_team_lead_agent.py`](https://github.com/johnnichev/selectools/blob/main/examples/73_team_lead_agent.py) | TeamLeadAgent with dynamic delegation strategy |
