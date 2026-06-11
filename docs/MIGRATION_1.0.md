# Migrating from 0.x to 1.0

This guide covers everything that changes between the final 0.24.x releases and
selectools **v1.0.0**. It is short on purpose: 1.0 is a stability declaration, not a
rewrite. If your code runs warning-free on 0.24.x with
`python -W error::DeprecationWarning`, you are already 1.0-ready except for the items
below.

For competitor migrations (LangChain, CrewAI, etc.) see [Migration Guides](MIGRATION.md).

---

## What semver means at 1.0

From v1.0.0 onward, selectools follows semantic versioning over its **stable** surface,
as defined in the [Deprecation Policy](DEPRECATION_POLICY.md):

- Every public symbol carries a stability marker: `@stable`, `@beta`, or `@deprecated`.
- **`@stable` APIs are frozen.** Breaking changes to any `@stable` symbol require v2.0.0.
- **`@beta` APIs are excluded from the semver guarantee.** They may change in a minor
  release (1.1, 1.2, ...) without a deprecation cycle. The marker — not the version
  number — is the contract.
- Deprecations continue to follow the two-minor-release window: anything deprecated in
  1.X is not removed before 1.X+2.

Check a symbol's marker at runtime via `obj.__stability__`, or see
[Stability Markers](modules/STABILITY.md).

---

## Breaking change: `AgentConfig.hooks` removed

The dict-based `hooks` parameter — deprecated since the observer system shipped (see
[ADR-002](decisions/002-observer-replaces-hooks.md)) — is **removed in 1.0**.
`AgentConfig(hooks={...})` is no longer accepted. Migrate to `AgentObserver`
(or `AsyncAgentObserver` for async handlers), which receives richer data: a stable
`run_id` for the whole run, a `call_id` per tool invocation, the resolved system prompt,
and structured results.

**Before (0.x):**

```python
from selectools import Agent, AgentConfig

def log_tool(tool_name, tool_args):
    print(f"-> {tool_name}({tool_args})")

agent = Agent(
    [search_tool],
    config=AgentConfig(
        hooks={
            "on_tool_start": log_tool,
            "on_agent_end": lambda response, usage: print(usage),
        }
    ),
)
```

**After (1.0):**

```python
from selectools import Agent, AgentConfig, AgentObserver

class LoggingObserver(AgentObserver):
    def on_tool_start(self, run_id, call_id, tool_name, tool_args):
        print(f"-> {tool_name}({tool_args})")

    def on_run_end(self, run_id, result):
        print(result.usage)

agent = Agent([search_tool], config=AgentConfig(observers=[LoggingObserver()]))
```

### Hook → observer method mapping

| 0.x hook key | 1.0 observer method | Notes |
|--------------|--------------------|-------|
| `on_agent_start` | `on_run_start(run_id, messages, system_prompt)` | gains `run_id` + resolved system prompt |
| `on_agent_end` | `on_run_end(run_id, result)` | `result` is the full `AgentResult` (use `result.usage`) |
| `on_iteration_start` | `on_iteration_start(run_id, iteration, messages)` | gains `run_id`; `iteration` is 1-based |
| `on_iteration_end` | `on_iteration_end(run_id, iteration, response)` | gains `run_id` |
| `on_tool_start` | `on_tool_start(run_id, call_id, tool_name, tool_args)` | `call_id` matches the paired end/error event, even with parallel tools |
| `on_tool_chunk` | `on_tool_chunk(run_id, call_id, tool_name, chunk)` | |
| `on_tool_end` | `on_tool_end(run_id, call_id, tool_name, result, duration_ms)` | duration is now **milliseconds** |
| `on_tool_error` | `on_tool_error(run_id, call_id, tool_name, error, tool_args, duration_ms)` | |
| `on_llm_start` | `on_llm_start(run_id, messages, model, system_prompt)` | |
| `on_llm_end` | `on_llm_end(run_id, response, usage)` | |
| `on_error` | `on_error(run_id, error, context)` | |

Override only the methods you need — all observer methods are optional no-ops on the base
class. Ready-made observers: `LoggingObserver`, `SimpleStepObserver`, plus `OTelObserver`
and `LangfuseObserver` in the `[observe]` extra.

---

## Rename: `Agent._clone_for_isolation()` → `Agent.clone_for_isolation()`

The agent-cloning method used for concurrency isolation (eval suites, A2A task handling,
planning) is promoted to public API at 1.0 and loses its underscore prefix. If you called
the private name directly:

```python
clone = agent._clone_for_isolation()   # 0.x
clone = agent.clone_for_isolation()    # 1.0
```

Behavior is unchanged: the clone shares tools and provider configuration but gets isolated
memory and fresh usage accounting.

---

## `__all__` additions (not breaking)

Symbols that were importable but missing from `selectools.__all__` are now declared.
This matters if you use `from selectools import *` or lint against the public API:

- **Pipeline family:** `Pipeline`, `Step`, `StepResult`, `step`, `branch`, `parallel`,
  `retry`, `cache_step`, `compose`
- **Router provider:** `RouterProvider`, `RouterConfig` (these two were previously
  importable only from `selectools.providers`; the top-level re-export is new)

Existing `from selectools import Pipeline`-style imports keep working as before; they are
simply official now. `from selectools import RouterProvider` becomes available for the
first time.

---

## Python 3.9 support dropped

v1.0.0 raises the floor to **Python 3.10** (`requires-python = ">=3.10"`). Python 3.9
reached upstream end-of-life in October 2025.

- The last series supporting 3.9 is **0.24.x**, which moves to security-fixes-only when
  1.0 ships (see [SECURITY.md](https://github.com/johnnichev/selectools/blob/main/SECURITY.md)).
- If you are stuck on 3.9, pin `selectools<1.0`.

---

## No other breaking changes

!!! note "Pending final confirmation"
    The statement below is finalized by the pre-1.0 stability-marking sweep. Until the
    1.0.0 tag is cut, treat this section as provisional.

Aside from the `AgentConfig.hooks` removal, the `clone_for_isolation` rename, and the
Python 3.9 drop, **no other breaking changes are planned for 1.0**. Code that runs
deprecation-warning-free on the latest 0.24.x release should run unmodified on 1.0
(on Python 3.10+).

---

## Symbols promoted to stable at 1.0

The pre-1.0 marking sweep promotes proven `@beta` APIs (2+ releases, no breaking
changes) to `@stable`:

| Symbol | Module | 0.x marker | 1.0 marker |
|--------|--------|-----------|------------|
| `AgentGraph` | `selectools.orchestration` | `@beta` | `@stable` |
| `GraphState` | `selectools.orchestration` | `@beta` | `@stable` |
| `MergePolicy` | `selectools.orchestration` | `@beta` | `@stable` |
| `ContextMode` | `selectools.orchestration` | `@beta` | `@stable` |
| `InterruptRequest` | `selectools.orchestration` | `@beta` | `@stable` |
| `Scatter` | `selectools.orchestration` | `@beta` | `@stable` |
| `CheckpointStore` | `selectools.orchestration` | `@beta` | `stable` (registry) |
| `InMemoryCheckpointStore` | `selectools.orchestration` | `@beta` | `@stable` |
| `FileCheckpointStore` | `selectools.orchestration` | `@beta` | `@stable` |
| `SQLiteCheckpointStore` | `selectools.orchestration` | `@beta` | `@stable` |
| `SupervisorAgent` | `selectools.orchestration` | `@beta` | `@stable` |
| `PlanAndExecuteAgent` | `selectools.patterns` | `@beta` | `@stable` |
| `ReflectiveAgent` | `selectools.patterns` | `@beta` | `@stable` |
| `DebateAgent` | `selectools.patterns` | `@beta` | `@stable` |
| `TeamLeadAgent` | `selectools.patterns` | `@beta` | `@stable` |
| `trace_to_html` | `selectools.trace` | `@beta` | `@stable` |
| `SimpleStepObserver` | `selectools.observer` | `@beta` | `@stable` |
| `RedisSessionStore` | `selectools.sessions` | `@beta` | `@stable` |
| `AzureOpenAIProvider` | `selectools.providers` | `@beta` | `@stable` |

Companion types of the promoted APIs (result/config/event dataclasses, enums, and
graph helper functions such as `GraphResult`, `GraphEvent`, `GraphNode`, `PlanStep`,
`ReflectiveResult`, `DebateResult`, `TeamLeadResult`, `SupervisorStrategy`,
`CheckpointMetadata`, `goto`, `update`, `merge_states`, `trace_to_json`) were
previously unmarked and are now `@stable` alongside their parent APIs.

`CheckpointStore` is `@runtime_checkable` and therefore carries its marker in
`selectools.stability.STABILITY_REGISTRY` instead of a class attribute (a class
attribute would become a structural protocol member on Python 3.9-3.11 and break
`isinstance()` for third-party stores).

Deliberately **not** promoted (still `@beta` at 1.0): the loop-detection family,
`LocalProvider` (test stub), the `Pipeline`/`compose` family, everything shipped in
v0.24/v0.25-dev (`pending`, `results`, `a2a`, `AgentAPI`, knowledge backends,
unified memory, `LiteLLMProvider`/`RouterProvider`, sanitizers, planning,
compression/HITL types), and the `rag`/`mcp`/`embeddings`/`observe`/`evals`
(evaluators) surfaces.

Anything still `@beta` at 1.0 (for example, the `selectools serve` CLI surface) remains
outside the semver guarantee until promoted in a later 1.x release.
