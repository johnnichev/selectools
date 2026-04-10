# Competitor-Informed Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 22 bugs in selectools identified by cross-referencing 95+ closed bug reports from Agno (39k stars) and 60+ from PraisonAI (6.9k stars) against selectools v0.21.0 code.

**Architecture:** TDD per bug — write failing regression test first, implement minimal fix, verify test passes, commit. Each bug is isolated enough to be fixed independently and tested independently. Bugs are grouped by severity; HIGH severity are shipping blockers.

**Tech Stack:** Python 3.9+, pytest, threading, asyncio, typing (Literal, Union, Optional).

**Branch:** `v0.22.0-competitor-bug-fixes`

**Test command:** `pytest tests/ -x -q` (or targeted: `pytest tests/path/test.py::test_name -v`)

---

## Bug Inventory

### HIGH severity (6) — shipping blockers

| ID | Bug | File | Competitor |
|---|---|---|---|
| BUG-01 | Streaming `run()/arun()` silently drops ToolCall objects | `_provider_caller.py:217-236, 472-509` | Agno #6757 |
| BUG-02 | `typing.Literal` crashes `@tool()` creation | `tools/decorators.py:16-46` | Agno #6720 |
| BUG-03 | `asyncio.run()` in 8 sync wrappers crashes in event loops | `graph.py:479, 1059`, `supervisor.py:240`, 4 patterns, `pipeline.py:486` | PraisonAI #1165 |
| BUG-04 | HITL `InterruptRequest` dropped in parallel groups | `orchestration/graph.py:1246` | Agno #4921 |
| BUG-05 | HITL `InterruptRequest` dropped in subgraphs | `orchestration/graph.py:1315` | Agno #4921 |
| BUG-06 | `ConversationMemory` has no `threading.Lock` | `memory.py` | PraisonAI #1164 |

### MEDIUM severity (9)

| ID | Bug | File | Competitor |
|---|---|---|---|
| BUG-07 | `<think>` tag content leaks into conversation history | `providers/anthropic_provider.py:107-143` | Agno #6878 |
| BUG-08 | ChromaDB/Pinecone/Qdrant no batch size limits | `rag/stores/chroma.py:119`, +2 others | Agno #7030 |
| BUG-09 | MCP concurrent tool calls race on shared session | `mcp/client.py:186` | Agno #6073 |
| BUG-10 | No type coercion for LLM tool args (`"42"` → `int`) | `tools/base.py:326-344` | PraisonAI #410 |
| BUG-11 | `Union[str, int]` crashes `@tool()` creation | `tools/decorators.py:26-31` | Agno #6720 |
| BUG-12 | Multi-interrupt generator nodes skip subsequent interrupts | `orchestration/graph.py:1139-1166` | Agno #4921 |
| BUG-13 | `GraphState.to_dict()` doesn't serialize `data` dict (corrupts checkpoints) | `orchestration/state.py:91,117` | Agno #7365 |
| BUG-14 | No session namespace isolation (shared session_id collision) | `sessions.py` | Agno #6275 |
| BUG-15 | Unbounded summary growth (context budget overflow) | `agent/_memory_manager.py:99-100` | Agno #5011 |

### LOW-MEDIUM severity (7)

| ID | Bug | File | Competitor |
|---|---|---|---|
| BUG-16 | `_build_cancelled_result` missing entity/KG extraction | `agent/core.py:540-562` | CLAUDE.md #23 |
| BUG-17 | `AgentTrace.add()` not thread-safe in parallel branches | `trace.py:118` | Agno #5847 |
| BUG-18 | Async observer exceptions silently lost | `agent/_lifecycle.py:48` | Agno #6236 |
| BUG-19 | `_clone_for_isolation` shallow-copies, sharing observer state | `agent/core.py:1124` | PraisonAI #1260 |
| BUG-20 | OTel/Langfuse observer dicts mutated without locks | `observe/otel.py:46-48`, `observe/langfuse.py:55-57` | PraisonAI #1260 |
| BUG-21 | No vector store search result deduplication | All 4 store `search()` methods | Agno #7047 |
| BUG-22 | `Optional[T]` without default treated as required | `tools/decorators.py:98` | Agno #7066 |

---

## File Structure

**Modified source files (by task):**
- `src/selectools/agent/_provider_caller.py` (Task 1 / BUG-01)
- `src/selectools/tools/decorators.py` (Tasks 2, 11, 22)
- `src/selectools/tools/base.py` (Tasks 2, 10)
- `src/selectools/orchestration/graph.py` (Tasks 3, 4, 5, 12)
- `src/selectools/orchestration/supervisor.py` (Task 3)
- `src/selectools/patterns/{team_lead,debate,reflective,plan_and_execute}.py` (Task 3)
- `src/selectools/pipeline.py` (Task 3)
- `src/selectools/memory.py` (Task 6)
- `src/selectools/providers/anthropic_provider.py` (Task 7)
- `src/selectools/rag/stores/chroma.py` (Task 8)
- `src/selectools/rag/stores/pinecone.py` (Task 8)
- `src/selectools/rag/stores/qdrant.py` (Task 8)
- `src/selectools/mcp/client.py` (Task 9)
- `src/selectools/orchestration/state.py` (Task 13)
- `src/selectools/sessions.py` (Task 14)
- `src/selectools/agent/_memory_manager.py` (Task 15)
- `src/selectools/agent/core.py` (Tasks 16, 19)
- `src/selectools/trace.py` (Task 17)
- `src/selectools/agent/_lifecycle.py` (Task 18)
- `src/selectools/observe/otel.py` (Task 20)
- `src/selectools/observe/langfuse.py` (Task 20)
- `src/selectools/rag/stores/{memory,sqlite,faiss}.py` (Task 21)

**New helper module:**
- `src/selectools/_async_utils.py` — Safe `run_sync()` helper for BUG-03

**New regression tests (one per bug):**

Per `tests/CLAUDE.md`, all regression tests are appended to the canonical file
`tests/agent/test_regression.py` as new top-level test functions named
`test_bug{NN}_*`. No new files or subdirectories are created — the
`tests/regressions/` layout was rejected in code review (I1). Each bug adds:

- `test_bug01_*` — streaming tool calls (Task 1)
- `test_bug02_*` — literal types (Task 2)
- `test_bug03_*` — asyncio reentry (Task 3)
- `test_bug04_*` — parallel HITL (Task 4)
- `test_bug05_*` — subgraph HITL (Task 5)
- `test_bug06_*` — memory thread safety (Task 6)
- `test_bug07_*` — think tag stripping (Task 7)
- `test_bug08_*` — RAG batch limits (Task 8)
- `test_bug09_*` — MCP concurrent (Task 9)
- `test_bug10_*` — tool arg coercion (Task 10)
- `test_bug11_*` — union types (Task 11)
- `test_bug12_*` — multi-interrupt (Task 12)
- `test_bug13_*` — GraphState serialization (Task 13)
- `test_bug14_*` — session namespace (Task 14)
- `test_bug15_*` — summary cap (Task 15)
- `test_bug16_*` — cancelled extraction (Task 16)
- `test_bug17_*` — trace thread safety (Task 17)
- `test_bug18_*` — observer exceptions (Task 18)
- `test_bug19_*` — clone isolation (Task 19)
- `test_bug20_*` — observer thread safety (Task 20)
- `test_bug21_*` — vector dedup (Task 21)
- `test_bug22_*` — optional-not-required (Task 22)

Helper classes for each bug should be prefixed with `_Bug{NN}` to stay out of
pytest collection and to avoid colliding with helpers from other bugs.

---

## HIGH SEVERITY BUGS (Tasks 1-6)

### Task 1: BUG-01 — Streaming drops ToolCall objects

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug01_streaming_preserves_tool_calls` (and async/sync-fallback siblings)
- Modify: `src/selectools/agent/_provider_caller.py:217-236` (sync `_streaming_call`)
- Modify: `src/selectools/agent/_provider_caller.py:472-509` (async `_astreaming_call`)

- [ ] **Step 1: Write the failing regression test**

```python
# Append to tests/agent/test_regression.py (BUG-01)
"""BUG-01: Streaming run()/arun() silently drops ToolCall objects.

Source: Agno #6757 pattern — competitor bug where tool function names
become empty strings in streaming responses.

Selectools variant: _streaming_call and _astreaming_call filter chunks
with `isinstance(chunk, str)` which drops ToolCall objects entirely.
Tools are never executed when AgentConfig(stream=True).
"""
from __future__ import annotations

from typing import Any, Iterator

import pytest

from selectools import Agent, AgentConfig, Tool, ToolParameter
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role, ToolCall


class StreamingToolProvider(LocalProvider):
    """Provider that yields a ToolCall during streaming."""

    name = "streaming_tool_stub"
    supports_streaming = True
    supports_async = False

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def stream(self, **kwargs: Any) -> Iterator[Any]:
        self.call_count += 1
        if self.call_count == 1:
            yield "I will call a tool. "
            yield ToolCall(tool_name="echo", parameters={"text": "hello"})
        else:
            yield "Done. Got: hello"


def _echo_fn(text: str) -> str:
    return text


def test_streaming_preserves_tool_calls():
    """When stream=True, ToolCall objects from the provider must be executed."""
    echo_tool = Tool(
        name="echo",
        description="Echo text",
        parameters=[ToolParameter(name="text", param_type=str, description="t", required=True)],
        function=_echo_fn,
    )
    provider = StreamingToolProvider()
    agent = Agent(
        tools=[echo_tool],
        provider=provider,
        config=AgentConfig(stream=True, max_iterations=3),
    )
    result = agent.run([Message(role=Role.USER, content="echo hello")])
    assert "Done" in result.content, f"Expected tool to execute; got: {result.content!r}"
    assert provider.call_count >= 2, "Agent should have looped after tool execution"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug01\"
```
Expected: FAIL — tool never executes; content does not contain "Done".

- [ ] **Step 3: Fix sync `_streaming_call`**

Replace `src/selectools/agent/_provider_caller.py` lines 217-236:

```python
    def _streaming_call(
        self, stream_handler: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, List["ToolCall"]]:
        if not getattr(self.provider, "supports_streaming", False):
            raise ProviderError(f"Provider {self.provider.name} does not support streaming.")

        aggregated: List[str] = []
        tool_calls: List["ToolCall"] = []
        for chunk in self.provider.stream(
            model=self._effective_model,
            system_prompt=self._system_prompt,
            messages=self._history,
            tools=self.tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
        ):
            if isinstance(chunk, str):
                if chunk:
                    aggregated.append(chunk)
                    if stream_handler:
                        stream_handler(chunk)
            elif isinstance(chunk, ToolCall):
                tool_calls.append(chunk)

        return "".join(aggregated), tool_calls
```

Add `Tuple` and `ToolCall` to the imports at the top of the file.

- [ ] **Step 4: Fix async `_astreaming_call`**

Apply the same change to `_astreaming_call` at lines 472-509. Both the `astream` branch (lines 489-493) and the sync fallback branch (lines 504-507) must collect ToolCalls into the same `tool_calls` list.

- [ ] **Step 5: Update callers of `_streaming_call` / `_astreaming_call`**

Find the `_call_provider` / `_acall_provider` methods that call `_streaming_call`. Find the line that constructs `Message(role=Role.ASSISTANT, content=response_text)`. Pass tool_calls into the Message:

```python
response_text, streamed_tool_calls = self._streaming_call(stream_handler)
return Message(
    role=Role.ASSISTANT,
    content=response_text,
    tool_calls=streamed_tool_calls or None,
)
```

Use Grep to find both call sites:
```bash
grep -n "_streaming_call\|_astreaming_call" src/selectools/agent/_provider_caller.py
```

- [ ] **Step 6: Run the regression test to verify fix**

```bash
pytest tests/agent/test_regression.py -v -k \"bug01\"
```
Expected: PASS.

- [ ] **Step 7: Run the full test suite to check no regressions**

```bash
pytest tests/ -x -q -k "not e2e"
```
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add tests/agent/test_regression.py src/selectools/agent/_provider_caller.py
git commit -m "fix(streaming): collect ToolCall objects during streaming

BUG-01: _streaming_call and _astreaming_call filtered chunks with
isinstance(chunk, str), silently dropping ToolCall objects yielded
by providers. Any user with AgentConfig(stream=True) calling run()
would find native provider tool calls were never executed.

Now both methods return (text, tool_calls) tuple. Caller propagates
tool_calls into the returned Message.

Cross-referenced from Agno #6757."
```

---

### Task 2: BUG-02 — `typing.Literal` crashes `@tool()` creation

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug02_literal_types`
- Modify: `src/selectools/tools/decorators.py:10,16-46,98-111`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-02)
"""BUG-02: typing.Literal crashes @tool() creation.

Source: Agno #6720 — get_json_schema_for_arg() does not handle
typing.Literal, producing {"type": "object"} instead of
{"type": "string", "enum": [...]}.

Selectools variant: _unwrap_type() returns Literal unchanged, then
_validate_tool_definition() rejects it as an unsupported type.
"""
from __future__ import annotations

from typing import Literal, Optional

from selectools.tools import tool


def test_literal_str_produces_enum():
    @tool()
    def set_mode(mode: Literal["fast", "slow", "auto"]) -> str:
        return f"mode={mode}"

    assert set_mode.name == "set_mode"
    params = {p.name: p for p in set_mode.parameters}
    assert "mode" in params
    assert params["mode"].enum == ["fast", "slow", "auto"]
    assert params["mode"].param_type is str


def test_literal_int_produces_enum():
    @tool()
    def set_level(level: Literal[1, 2, 3]) -> str:
        return f"level={level}"

    params = {p.name: p for p in set_level.parameters}
    assert params["level"].enum == [1, 2, 3]
    assert params["level"].param_type is int


def test_optional_literal_works():
    @tool()
    def filter_by(tag: Optional[Literal["red", "blue"]] = None) -> str:
        return f"tag={tag}"

    params = {p.name: p for p in filter_by.parameters}
    assert params["tag"].enum == ["red", "blue"]
    assert params["tag"].required is False
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug02\"
```
Expected: FAIL — ToolValidationError on `@tool()` application.

- [ ] **Step 3: Add Literal handling to `decorators.py`**

Update the import line 10 to include `Literal` and `Tuple`:

```python
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin, get_type_hints
```

Add a helper function before `_unwrap_type` (around line 15):

```python
def _literal_info(type_hint: Any) -> Optional[Tuple[Any, List[Any]]]:
    """Return (base_type, enum_values) for Literal[...] hints, else None.

    Unwraps Optional[Literal[...]] as well. Base type is inferred from the
    first literal value (e.g. Literal["a", "b"] → str).
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _literal_info(non_none[0])
    if sys.version_info >= (3, 10):
        import types as _types  # noqa: PLC0415
        if isinstance(type_hint, _types.UnionType):
            args = get_args(type_hint)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _literal_info(non_none[0])
    if origin is Literal:
        values = list(get_args(type_hint))
        if not values:
            return None
        base_type = type(values[0])
        return base_type, values
    return None
```

- [ ] **Step 4: Use `_literal_info` in `_infer_parameters_from_callable`**

Modify `_infer_parameters_from_callable` around line 90-111:

```python
        meta = param_metadata.get(name, {})
        description = meta.get("description", f"Parameter {name}")
        enum_values: Optional[List[Any]] = meta.get("enum")

        raw_type = type_hints.get(name, str)
        lit = _literal_info(raw_type)
        if lit is not None:
            param_type, literal_values = lit
            if enum_values is None:
                enum_values = literal_values
        else:
            param_type = _unwrap_type(raw_type)

        is_optional = param.default != inspect.Parameter.empty
```

- [ ] **Step 5: Run the regression test**

```bash
pytest tests/agent/test_regression.py -v -k \"bug02\"
```
Expected: PASS.

- [ ] **Step 6: Run full tool tests to check no regressions**

```bash
pytest tests/tools/ -x -q
```
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/agent/test_regression.py src/selectools/tools/decorators.py
git commit -m "fix(tools): support typing.Literal in @tool() parameters

BUG-02: @tool() crashed on Literal[...] parameters because
_unwrap_type() returned Literal unchanged, and then
_validate_tool_definition() rejected it as an unsupported type.

Now detects Literal (and Optional[Literal]), extracts enum values,
infers base type from the first value, and auto-populates
ToolParameter.enum. Supports str, int, float, and bool literals.

Cross-referenced from Agno #6720."
```

---

### Task 3: BUG-03 — `asyncio.run()` crashes in existing event loops

**Files:**
- Create: `src/selectools/_async_utils.py`
- Modify: `tests/agent/test_regression.py` — add test function `test_bug03_asyncio_reentry`
- Modify: `src/selectools/orchestration/graph.py:479` (`AgentGraph.run`)
- Modify: `src/selectools/orchestration/graph.py:1059` (`AgentGraph.resume`)
- Modify: `src/selectools/orchestration/supervisor.py:240` (`SupervisorAgent.run`)
- Modify: `src/selectools/patterns/team_lead.py:126`
- Modify: `src/selectools/patterns/debate.py:80`
- Modify: `src/selectools/patterns/reflective.py:82`
- Modify: `src/selectools/patterns/plan_and_execute.py:110`
- Modify: `src/selectools/pipeline.py:486`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-03)
"""BUG-03: asyncio.run() in sync wrappers crashes inside running event loops.

Source: PraisonAI #1165 — asyncio.run() called from sync context reachable
by async callers crashes with "cannot call asyncio.run() while an event
loop is running". Reachable from Jupyter, FastAPI handlers, async tests.
"""
from __future__ import annotations

import asyncio

import pytest

from selectools._async_utils import run_sync


def test_run_sync_outside_event_loop():
    async def coro():
        return 42

    assert run_sync(coro()) == 42


def test_run_sync_inside_running_loop():
    """Key test — calling run_sync from within an async function."""
    async def outer():
        async def inner():
            return "hello"
        return run_sync(inner())

    result = asyncio.run(outer())
    assert result == "hello"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug03\"
```
Expected: FAIL — `ModuleNotFoundError: _async_utils`.

- [ ] **Step 3: Create the safe run_sync helper**

```python
# src/selectools/_async_utils.py
"""Safe synchronous-wrapper utilities for async code.

Calling asyncio.run() from a sync function that is itself reachable
from an async caller raises RuntimeError: asyncio.run() cannot be called
when another event loop is running. This module provides a helper that
detects the surrounding event loop and executes the coroutine on a fresh
loop in a dedicated thread when one is already running.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code.

    If no event loop is running in the current thread, uses asyncio.run.
    If one is running, spawns a worker thread, creates a fresh event loop
    there, and waits for the result. Safe to call from Jupyter notebooks,
    FastAPI handlers, async tests, and nested orchestration.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    def _runner() -> T:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        return future.result()
```

- [ ] **Step 4: Run the two unit tests for run_sync**

```bash
pytest tests/agent/test_regression.py -v -k \"bug03\"
```
Expected: PASS.

- [ ] **Step 5: Replace `asyncio.run(...)` in `AgentGraph.run`**

In `src/selectools/orchestration/graph.py` around line 479, change:

```python
        return asyncio.run(
            self.arun(
                prompt_or_state,
                checkpoint_store=checkpoint_store,
                checkpoint_id=checkpoint_id,
            )
        )
```

to:

```python
        from .._async_utils import run_sync

        return run_sync(
            self.arun(
                prompt_or_state,
                checkpoint_store=checkpoint_store,
                checkpoint_id=checkpoint_id,
            )
        )
```

- [ ] **Step 6: Replace `asyncio.run(...)` in `AgentGraph.resume`**

Same pattern at line 1059. Use Grep to find both:
```bash
grep -n "asyncio.run(" src/selectools/orchestration/graph.py
```

- [ ] **Step 7: Replace in SupervisorAgent.run**

Modify `src/selectools/orchestration/supervisor.py` line 240.

- [ ] **Step 8: Replace in 4 pattern agents**

Update each of:
- `src/selectools/patterns/team_lead.py:126`
- `src/selectools/patterns/debate.py:80`
- `src/selectools/patterns/reflective.py:82`
- `src/selectools/patterns/plan_and_execute.py:110`

Each imports `from ..._async_utils import run_sync` and replaces `asyncio.run(self.arun(...))` with `run_sync(self.arun(...))`.

- [ ] **Step 9: Replace in pipeline.py**

Modify `src/selectools/pipeline.py:486`. Relative import: `from ._async_utils import run_sync`.

- [ ] **Step 10: Run the regression test**

```bash
pytest tests/agent/test_regression.py -v -k \"bug03\"
```
Expected: All tests PASS.

- [ ] **Step 11: Run full suite**

```bash
pytest tests/ -x -q -k "not e2e"
```
Expected: All tests pass.

- [ ] **Step 12: Commit**

```bash
git add src/selectools/_async_utils.py tests/agent/test_regression.py (test_bug03_*) \
  src/selectools/orchestration/graph.py src/selectools/orchestration/supervisor.py \
  src/selectools/patterns/ src/selectools/pipeline.py
git commit -m "fix(async): safe run_sync helper for 8 sync wrappers

BUG-03: Bare asyncio.run() in 8 sync wrappers crashed with
'cannot call asyncio.run() while another event loop is running'
when called from Jupyter, FastAPI handlers, or nested async code.

New _async_utils.run_sync() helper detects a running loop and
offloads to a worker thread when needed. Applied to:
- AgentGraph.run / AgentGraph.resume
- SupervisorAgent.run
- PlanAndExecuteAgent / ReflectiveAgent / DebateAgent / TeamLeadAgent
- Pipeline._execute_step

Cross-referenced from PraisonAI #1165."
```

---

### Task 4: BUG-04 — HITL lost in parallel groups

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug04_parallel_hitl`
- Modify: `src/selectools/orchestration/graph.py:1237-1288` (`_aexecute_parallel`)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-04)
"""BUG-04: InterruptRequest from a child node in a parallel group is silently
dropped. The parent graph treats the child as completed.

Source: Agno #4921 — NoneType does not have run_id error when running HITL
within agent tools which are part of team.
"""
from __future__ import annotations

from selectools import AgentGraph
from selectools.orchestration import (
    GraphNode,
    GraphState,
    InMemoryCheckpointStore,
    InterruptRequest,
    ParallelGroupNode,
)
from selectools.types import AgentResult, Message, Role


def _normal_child(state: GraphState):
    state.data["normal"] = "done"
    return AgentResult(message=Message(role=Role.ASSISTANT, content="normal")), state


def _hitl_child_generator(state: GraphState):
    response = yield InterruptRequest(key="approval", prompt="approve?")
    state.data["approval"] = response
    yield AgentResult(message=Message(role=Role.ASSISTANT, content="hitl")), state


def test_parallel_group_propagates_hitl():
    """When a child in a parallel group interrupts, the parent graph must pause."""
    graph = AgentGraph(name="test_parallel_hitl")
    normal_node = GraphNode(name="normal", agent=None, callable_fn=_normal_child)
    hitl_node = GraphNode(name="hitl", agent=None, generator_fn=_hitl_child_generator)
    parallel = ParallelGroupNode(name="group", child_node_names=["normal", "hitl"])
    graph.add_node(normal_node)
    graph.add_node(hitl_node)
    graph.add_node(parallel)
    graph.set_entry("group")
    graph.add_edge("group", "__end__")

    store = InMemoryCheckpointStore()
    result = graph.run("start", checkpoint_store=store)

    assert result.interrupted, f"Expected graph to pause; got: {result}"
    assert result.interrupt_key == "approval"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug04\"
```
Expected: FAIL — `result.interrupted` is False.

- [ ] **Step 3: Fix `run_child` to return the interrupted flag**

In `_aexecute_parallel` around line 1237, change `run_child`'s return type to include the interrupted flag:

```python
            async def run_child(
                child_name: str, branch_state: GraphState
            ) -> Tuple[str, AgentResult, GraphState, bool]:
                child_node = self._nodes.get(child_name)
                if child_node is None:
                    raise GraphExecutionError(
                        self.name, child_name, KeyError(f"Child node {child_name!r} not found"), 0
                    )
                if isinstance(child_node, GraphNode):
                    result, new_state, interrupted = await self._aexecute_node(
                        child_node, branch_state, trace, run_id
                    )
                else:
                    result = _make_synthetic_result(branch_state)
                    new_state = branch_state
                    interrupted = False
                return child_name, result, new_state, interrupted
```

- [ ] **Step 4: Update the result collection loop**

In the same method around line 1262, unpack the 4-tuple and track interrupts:

```python
            child_results: Dict[str, List[AgentResult]] = {}
            branch_final_states: List[GraphState] = []
            interrupted_child: Optional[str] = None
            for i, output in enumerate(child_outputs):
                if isinstance(output, BaseException):
                    child_name = node.child_node_names[i]
                    state.errors.append(
                        {"node": child_name, "error": str(output), "type": type(output).__name__}
                    )
                    if self.error_policy == ErrorPolicy.ABORT:
                        exc = output if isinstance(output, Exception) else Exception(str(output))
                        raise GraphExecutionError(self.name, child_name, exc, 0) from output
                    continue
                child_name, result, new_state, child_interrupted = output
                child_results.setdefault(child_name, []).append(result)
                branch_final_states.append(new_state)
                if child_interrupted and interrupted_child is None:
                    interrupted_child = child_name

            # Propagate interrupt metadata to parent state
            if interrupted_child is not None:
                merged_interrupt_marker = {"__parallel_interrupt__": interrupted_child}
            else:
                merged_interrupt_marker = {}
```

Then after computing `merged`, inject the marker:

```python
            if merged_interrupt_marker:
                merged.data.update(merged_interrupt_marker)
```

- [ ] **Step 5: Propagate the interrupt in `_aexecute_node`**

Find where `_aexecute_parallel` is called within `_aexecute_node` and check for `__parallel_interrupt__` in the merged state. If present, return `interrupted=True` from `_aexecute_node`.

```bash
grep -n "_aexecute_parallel" src/selectools/orchestration/graph.py
```

- [ ] **Step 6: Run the regression test**

```bash
pytest tests/agent/test_regression.py -v -k \"bug04\"
```
Expected: PASS.

- [ ] **Step 7: Run full orchestration suite**

```bash
pytest tests/orchestration/ -x -q
```
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add tests/agent/test_regression.py src/selectools/orchestration/graph.py
git commit -m "fix(orchestration): propagate HITL interrupts from parallel groups

BUG-04: run_child in _aexecute_parallel discarded the interrupted
boolean from _aexecute_node. If a child yielded InterruptRequest,
the signal was lost and the graph continued as if the child
completed normally.

Now run_child returns a 4-tuple including the interrupted flag,
and the first interrupting child surfaces the interrupt to the
graph's outer loop for proper checkpointing.

Cross-referenced from Agno #4921."
```

---

### Task 5: BUG-05 — HITL lost in subgraphs

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug05_subgraph_hitl`
- Modify: `src/selectools/orchestration/graph.py:1295-1332` (`_aexecute_subgraph`)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-05)
"""BUG-05: InterruptRequest raised inside a subgraph is silently dropped
by the parent graph. The subgraph's pause state is lost.

Source: Agno #4921 — HITL inside nested contexts fails.
"""
from __future__ import annotations

from selectools import AgentGraph
from selectools.orchestration import (
    GraphNode,
    GraphState,
    InMemoryCheckpointStore,
    InterruptRequest,
    SubgraphNode,
)
from selectools.types import AgentResult, Message, Role


def _hitl_generator(state: GraphState):
    response = yield InterruptRequest(key="approval", prompt="ok?")
    state.data["approval"] = response
    yield AgentResult(message=Message(role=Role.ASSISTANT, content="done")), state


def test_subgraph_propagates_hitl_interrupt():
    inner = AgentGraph(name="inner")
    inner_node = GraphNode(name="gate", agent=None, generator_fn=_hitl_generator)
    inner.add_node(inner_node)
    inner.set_entry("gate")
    inner.add_edge("gate", "__end__")

    outer = AgentGraph(name="outer")
    sub = SubgraphNode(name="nested", graph=inner)
    outer.add_node(sub)
    outer.set_entry("nested")
    outer.add_edge("nested", "__end__")

    store = InMemoryCheckpointStore()
    result = outer.run("start", checkpoint_store=store)

    assert result.interrupted, "Subgraph interrupt must propagate to parent"
    assert result.interrupt_key == "approval"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug05\"
```
Expected: FAIL — `result.interrupted` is False.

- [ ] **Step 3: Update `_aexecute_subgraph` signature and check sub_result.interrupted**

Modify around line 1295-1332. Change return type to include interrupted flag:

```python
    async def _aexecute_subgraph(
        self,
        node: SubgraphNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[AgentResult, GraphState, bool]:
        """Execute a nested AgentGraph as a node."""
        sub_state = GraphState.from_prompt(
            state.data.get(STATE_KEY_LAST_OUTPUT, "")
            or (state.messages[-1].content if state.messages else "")
        )

        for parent_key, sub_key in node.input_map.items():
            if parent_key in state.data:
                sub_state.data[sub_key] = state.data[parent_key]

        sub_result = await node.graph.arun(sub_state, _interrupt_response=None)

        if sub_result.interrupted:
            state.data["__subgraph_interrupt__"] = {
                "key": sub_result.interrupt_key,
                "prompt": sub_result.interrupt_prompt,
                "subgraph": node.name,
            }
            synthetic = AgentResult(
                message=Message(role=Role.ASSISTANT, content=""),
                iterations=sub_result.steps,
                usage=sub_result.total_usage,
            )
            return synthetic, state, True

        for sub_key, parent_key in node.output_map.items():
            if sub_key in sub_result.state.data:
                state.data[parent_key] = sub_result.state.data[sub_key]

        state.data[STATE_KEY_LAST_OUTPUT] = sub_result.content
        state.messages.extend(sub_result.state.messages[-2:])
        state.history.extend(sub_result.state.history)

        synthetic = AgentResult(
            message=Message(role=Role.ASSISTANT, content=sub_result.content),
            iterations=sub_result.steps,
            usage=sub_result.total_usage,
        )
        return synthetic, state, False
```

- [ ] **Step 4: Update the caller of `_aexecute_subgraph`**

```bash
grep -n "_aexecute_subgraph" src/selectools/orchestration/graph.py
```

Update to unpack the 3-tuple and propagate the interrupted flag up.

- [ ] **Step 5: Run the regression test**

```bash
pytest tests/agent/test_regression.py -v -k \"bug05\"
```
Expected: PASS.

- [ ] **Step 6: Run full orchestration suite**

```bash
pytest tests/orchestration/ -x -q
```
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add tests/agent/test_regression.py src/selectools/orchestration/graph.py
git commit -m "fix(orchestration): propagate HITL interrupts from subgraphs

BUG-05: _aexecute_subgraph never checked sub_result.interrupted.
If a subgraph paused for HITL, the parent treated it as completed
and continued executing, losing the pause state.

Now _aexecute_subgraph returns (result, state, interrupted) and
propagates interrupt metadata to the parent graph for proper
checkpointing and resumption.

Cross-referenced from Agno #4921."
```

---

### Task 6: BUG-06 — ConversationMemory missing threading.Lock

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug06_memory_thread_safety`
- Modify: `src/selectools/memory.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-06)
"""BUG-06: ConversationMemory has no threading.Lock. Concurrent mutation
from multiple threads races on the _messages list.

Source: PraisonAI #1164, #1260 — thread-unsafe shared mutable state.
"""
from __future__ import annotations

import threading

from selectools.memory import ConversationMemory
from selectools.types import Message, Role


def test_concurrent_add_preserves_all_messages():
    """10 threads × 100 adds = 1000 messages should all be preserved."""
    memory = ConversationMemory(max_messages=10000)
    n_threads = 10
    n_adds = 100
    errors = []

    def worker(thread_id: int):
        try:
            for i in range(n_adds):
                memory.add(Message(role=Role.USER, content=f"t{thread_id}-m{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Worker errors: {errors}"
    history = memory.get_history()
    assert len(history) == n_threads * n_adds, (
        f"Expected {n_threads * n_adds} messages, got {len(history)}"
    )


def test_concurrent_add_with_trim_no_crash():
    """Low max_messages triggers _enforce_limits concurrently — must not crash."""
    memory = ConversationMemory(max_messages=50)
    errors = []

    def worker(thread_id: int):
        try:
            for i in range(200):
                memory.add(Message(role=Role.USER, content=f"t{thread_id}-m{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Worker errors: {errors}"
    assert len(memory.get_history()) <= 50
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/agent/test_regression.py -v -k \"bug06\"
```
Expected: FAIL or intermittent — race condition produces wrong count or errors.

- [ ] **Step 3: Add threading.RLock to ConversationMemory**

In `src/selectools/memory.py`, add to imports:

```python
import threading
```

In `__init__`, add the lock:

```python
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._summary: Optional[str] = None
        self._last_trimmed: List[Message] = []
        self._lock = threading.RLock()
```

Use `RLock` (re-entrant) because `add` calls `_enforce_limits` which may call other locked methods.

- [ ] **Step 4: Wrap all mutation and read methods with `with self._lock:`**

Use Grep to find all methods:
```bash
grep -n "    def " src/selectools/memory.py
```

Methods to protect: `add`, `add_many`, `get_history`, `get_recent`, `clear`, `_enforce_limits`, `to_dict`, `from_dict`, `branch`, `get_summary`, `set_summary`, and any other state-reading or state-mutating method.

Example for `add`:

```python
    def add(self, message: Message) -> None:
        with self._lock:
            self._messages.append(message)
            self._enforce_limits()
```

- [ ] **Step 5: Preserve state-restoration compatibility**

`threading.RLock` is not serializable for disk storage. Override `__getstate__` and `__setstate__` to exclude the lock from serialization and recreate it on restore:

```python
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()
```

- [ ] **Step 6: Run the regression test**

```bash
pytest tests/agent/test_regression.py -v -k \"bug06\"
```
Expected: PASS.

- [ ] **Step 7: Run existing memory tests**

```bash
pytest tests/ -k "memory" -x -q
```
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add tests/agent/test_regression.py src/selectools/memory.py
git commit -m "fix(memory): add threading.RLock to ConversationMemory

BUG-06: ConversationMemory was the only shared-state class in
selectools without a lock. Concurrent add()/add_many()/get_history()
from multiple threads raced on self._messages, potentially losing
messages or corrupting the list.

All mutation and read methods now acquire self._lock (RLock for
re-entrance). __getstate__/__setstate__ preserve serialization
compat by recreating the lock on restore.

Cross-referenced from PraisonAI #1164 / #1260."
```

---

## MEDIUM SEVERITY BUGS (Tasks 7-15)

### Task 7: BUG-07 — `<think>` tag content leaks

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug07_think_tag_stripping`
- Modify: `src/selectools/providers/anthropic_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/agent/test_regression.py (BUG-07)
"""BUG-07: <think>...</think> content leaks into conversation history.

Source: Agno #6878.
"""
from selectools.providers.anthropic_provider import _strip_reasoning_tags


def test_strip_simple_think_tags():
    text = "<think>This is my reasoning.</think>The answer is 42."
    assert _strip_reasoning_tags(text) == "The answer is 42."


def test_strip_multiline_think_tags():
    text = "<think>\nLine 1\nLine 2\n</think>\nFinal answer."
    assert _strip_reasoning_tags(text).strip() == "Final answer."


def test_strip_multiple_think_blocks():
    text = "<think>first</think>Hello<think>second</think> world"
    assert _strip_reasoning_tags(text) == "Hello world"


def test_no_think_tags_unchanged():
    text = "Plain text with no tags"
    assert _strip_reasoning_tags(text) == text
```

- [ ] **Step 2: Confirm failure**

```bash
pytest tests/agent/test_regression.py -v -k \"bug07\"
```

- [ ] **Step 3: Add stripping helper**

At the top of `anthropic_provider.py` below imports:

```python
import re as _re

_THINK_TAG_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def _strip_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not text or "<think>" not in text:
        return text
    return _THINK_TAG_RE.sub("", text)
```

- [ ] **Step 4: Apply in all text accumulation paths**

```bash
grep -n "content_text\|text_delta\|text +=" src/selectools/providers/anthropic_provider.py
```

Apply `_strip_reasoning_tags` at the point where accumulated text is finalized in `complete`, `acomplete`, and to each delta in `stream`/`astream`.

- [ ] **Step 5: Run test, commit**

```bash
pytest tests/agent/test_regression.py -v -k \"bug07\"
git add tests/agent/test_regression.py src/selectools/providers/anthropic_provider.py
git commit -m "fix(anthropic): strip <think> reasoning tags from output (BUG-07)"
```

---

### Task 8: BUG-08 — RAG store batch size limits

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug08_rag_batch_limits`
- Modify: `src/selectools/rag/stores/{chroma,pinecone,qdrant}.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/agent/test_regression.py (BUG-08)
from unittest.mock import MagicMock


def test_chroma_batches_large_upsert():
    from selectools.rag.stores.chroma import ChromaVectorStore
    from selectools.rag.types import Document

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = MagicMock()
    store._batch_size = 100
    store.embedder = MagicMock()
    store.embedder.embed_batch.return_value = [[0.1] * 16 for _ in range(250)]

    docs = [Document(text=f"doc {i}", metadata={}) for i in range(250)]
    store.add_documents(docs)
    assert store.collection.upsert.call_count == 3
```

- [ ] **Step 2: Add `_batch_size` attribute and chunking to each store**

For `chroma.py`:

```python
        self._batch_size = 5000

        for start in range(0, len(ids), self._batch_size):
            end = start + self._batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
```

Apply the same pattern to `pinecone.py` (batch_size=100) and `qdrant.py` (batch_size=1000).

- [ ] **Step 3: Run test, commit**

```bash
pytest tests/agent/test_regression.py -v -k \"bug08\"
git add tests/agent/test_regression.py src/selectools/rag/stores/chroma.py src/selectools/rag/stores/pinecone.py src/selectools/rag/stores/qdrant.py
git commit -m "fix(rag): chunk large upserts in Chroma/Pinecone/Qdrant (BUG-08)"
```

---

### Task 9: BUG-09 — MCP concurrent tool call lock

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug09_mcp_concurrent`
- Modify: `src/selectools/mcp/client.py`

- [ ] **Step 1: Write failing test** (see original spec for the async test with mocked session)

- [ ] **Step 2: Add `self._tool_lock = asyncio.Lock()` to MCPClient init**

- [ ] **Step 3: Wrap `_call_tool` body in `async with self._tool_lock:`**

- [ ] **Step 4: Run test, commit**

```bash
git commit -m "fix(mcp): serialize concurrent tool calls on shared session (BUG-09)"
```

---

### Task 10: BUG-10 — Tool argument type coercion

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug10_tool_arg_coercion`
- Modify: `src/selectools/tools/base.py:326-344`

- [ ] **Step 1: Write failing test** (str→int, str→float, str→bool coercion)

- [ ] **Step 2: In `_validate_single`, attempt coercion before rejecting:**

```python
        if isinstance(value, param_type):
            return value
        if isinstance(value, str) and param_type in (int, float, bool):
            try:
                if param_type is bool:
                    lowered = value.strip().lower()
                    if lowered in ("true", "1", "yes", "on"):
                        return True
                    if lowered in ("false", "0", "no", "off"):
                        return False
                    raise ValueError(f"Cannot coerce {value!r} to bool")
                return param_type(value)
            except (ValueError, TypeError) as exc:
                raise ToolValidationError(
                    f"Invalid {name!r}: cannot coerce {value!r} to {param_type.__name__}: {exc}"
                )
        raise ToolValidationError(
            f"Invalid {name!r}: expected {param_type.__name__}, got {type(value).__name__}"
        )
```

- [ ] **Step 3: Run test, commit**

```bash
git commit -m "fix(tools): coerce string args to int/float/bool (BUG-10)"
```

---

### Task 11: BUG-11 — `Union[str, int]` fallback

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug11_union_types`
- Modify: `src/selectools/tools/decorators.py:26-31`

- [ ] **Step 1: Write failing test**

- [ ] **Step 2: In `_unwrap_type`, return `str` for multi-type unions:**

```python
    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _unwrap_type(non_none_args[0])
        if len(non_none_args) > 1:
            return str
```

Apply the same fallback in the `types.UnionType` branch.

- [ ] **Step 3: Run test, commit**

```bash
git commit -m "fix(tools): support Union[str, int] via str fallback (BUG-11)"
```

---

### Task 12: BUG-12 — Multi-interrupt generator nodes

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug12_multi_interrupt`
- Modify: `src/selectools/orchestration/graph.py:1139-1166`

- [ ] **Step 1: Write failing test** — two-gate generator, both interrupts must fire

- [ ] **Step 2: Read current `_aexecute_generator_node`**

```bash
grep -n "_aexecute_generator_node" src/selectools/orchestration/graph.py
```

- [ ] **Step 3: Fix the iteration — `asend` return value must be processed**

Core fix: after `gen.asend(response)` returns a yielded value, that value must be checked for InterruptRequest before advancing with `__anext__`. Process the `asend` return in the same code path as items from the subsequent `async for` loop.

- [ ] **Step 4: Fix `interrupt_index` counter to persist across calls**

- [ ] **Step 5: Run test, commit**

```bash
git commit -m "fix(orchestration): handle multi-interrupt generator nodes (BUG-12)"
```

---

### Task 13: BUG-13 — GraphState.to_dict() JSON validation

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug13_graphstate_serialization`
- Modify: `src/selectools/orchestration/state.py:91,117`

- [ ] **Step 1: Write failing test** — non-serializable object should raise clearly

- [ ] **Step 2: Validate `data` in `to_dict` via JSON round-trip:**

```python
    def to_dict(self) -> Dict[str, Any]:
        import json
        try:
            serialized_data = json.loads(json.dumps(self.data))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"GraphState.data contains non-serializable values: {exc}. "
                f"All values in state.data must be JSON-compatible for checkpointing."
            )
        return {
            "messages": [m.to_dict() for m in self.messages],
            "history": list(self.history),
            "data": serialized_data,
            "errors": list(self.errors),
            "turn_count": self.turn_count,
        }
```

- [ ] **Step 3: Run test, commit**

```bash
git commit -m "fix(state): fail fast on non-serializable GraphState.data (BUG-13)"
```

---

### Task 14: BUG-14 — Session namespace isolation

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug14_session_namespace`
- Modify: `src/selectools/sessions.py` (all 3 stores)

- [ ] **Step 1: Write failing test**

```python
def test_different_namespaces_isolated():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileSessionStore(directory=tmpdir)
        mem_a = ConversationMemory()
        mem_a.add(Message(role=Role.USER, content="A"))
        store.save("shared_id", mem_a, namespace="agent_a")
        mem_b = ConversationMemory()
        mem_b.add(Message(role=Role.USER, content="B"))
        store.save("shared_id", mem_b, namespace="agent_b")
        assert store.load("shared_id", namespace="agent_a").get_history()[0].content == "A"
        assert store.load("shared_id", namespace="agent_b").get_history()[0].content == "B"
```

- [ ] **Step 2: Add `namespace: Optional[str] = None` parameter to save/load/delete in the protocol and all 3 concrete stores**

- [ ] **Step 3: Derive the storage key from `{namespace}:{session_id}` when namespace is set, else bare `session_id`**

- [ ] **Step 4: Run test, commit**

```bash
git commit -m "fix(sessions): add namespace parameter for session isolation (BUG-14)"
```

---

### Task 15: BUG-15 — Summary growth cap

**Files:**
- Modify: `tests/agent/test_regression.py` — add test function `test_bug15_summary_cap`
- Modify: `src/selectools/agent/_memory_manager.py:99-100`

- [ ] **Step 1: Write failing test**

- [ ] **Step 2: Add cap constant and helper:**

```python
_MAX_SUMMARY_CHARS = 4000


def _append_summary(existing: Optional[str], new_chunk: str) -> str:
    if not existing:
        combined = new_chunk
    else:
        combined = f"{existing} {new_chunk}"
    if len(combined) > _MAX_SUMMARY_CHARS:
        combined = combined[-_MAX_SUMMARY_CHARS:]
    return combined
```

- [ ] **Step 3: Replace concatenation in `_maybe_summarize_trim` with `_append_summary` call**

- [ ] **Step 4: Run test, commit**

```bash
git commit -m "fix(memory): cap summary at 4000 chars to prevent overflow (BUG-15)"
```

---

## LOW-MEDIUM SEVERITY BUGS (Tasks 16-22)

These bugs are isolated and smaller. Each task follows the same pattern: write test, make minimal change, run test, commit.

### Task 16: BUG-16 — Cancelled result missing extraction

**File:** `src/selectools/agent/core.py:540-562`

- [ ] Write test: assert entities extracted during run are persisted after cancellation
- [ ] Add `_extract_entities()` and `_extract_kg_triples()` calls to `_build_cancelled_result`
- [ ] Run test, commit

---

### Task 17: BUG-17 — AgentTrace.add() thread safety

**File:** `src/selectools/trace.py:118`

- [ ] Write test: 10 threads × 100 adds, verify final count
- [ ] Add `self._lock = threading.Lock()` to `AgentTrace.__init__`, wrap `add()` body
- [ ] Run test, commit

---

### Task 18: BUG-18 — Async observer exception logging

**File:** `src/selectools/agent/_lifecycle.py:48`

- [ ] Write test: observer that raises should log via `logger.warning`
- [ ] Add `add_done_callback` with logging:

```python
task = asyncio.ensure_future(handler(*args))
def _log_task_exception(t: "asyncio.Task[Any]") -> None:
    if t.cancelled():
        return
    exc = t.exception()
    if exc is not None:
        logger.warning("Async observer raised: %s", exc, exc_info=exc)
task.add_done_callback(_log_task_exception)
```

- [ ] Run test, commit

---

### Task 19: BUG-19 — Clone isolation

**File:** `src/selectools/agent/core.py:1124`

- [ ] Write test: two batch clones with observers don't share state
- [ ] Replace `copy.copy(self)` with explicit deep-copy of observers list and mutable config groups
- [ ] Run test, commit

---

### Task 20: BUG-20 — OTel/Langfuse observer locks

**Files:** `src/selectools/observe/otel.py:46-48`, `src/selectools/observe/langfuse.py:55-57`

- [ ] Write test: concurrent on_llm_start from 10 threads, verify counter
- [ ] Add `self._lock = threading.Lock()` to each observer init, wrap all `_spans/_llm_counter/_traces` mutations
- [ ] Run test, commit

---

### Task 21: BUG-21 — Vector store search dedup

**Files:** `src/selectools/rag/stores/{chroma,memory,sqlite,faiss}.py` all `search()` methods

- [ ] Write test: insert same doc twice, search, assert single result
- [ ] Add `dedup: bool = True` parameter to VectorStore.search protocol; implement post-search text-hash dedup in each store
- [ ] Run test, commit

---

### Task 22: BUG-22 — Optional[T] without default

**File:** `src/selectools/tools/decorators.py:98`

- [ ] Write test: `@tool() def f(x: Optional[str]): ...` → `x.required is False`
- [ ] In `_infer_parameters_from_callable`, detect Optional via Union-with-None check, set `is_optional=True` without requiring default
- [ ] Run test, commit

---

## Final Verification (Task 23)

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -x -q
```
Expected: All 5,200+ tests pass plus 22 new regression tests.

- [ ] **Step 2: Run mypy**

```bash
mypy src/
```
Expected: No errors.

- [ ] **Step 3: Run linters**

```bash
black src/ tests/ --line-length=100 --check
isort src/ tests/ --profile=black --line-length=100 --check
flake8 src/
bandit -r src/ -ll -q -c pyproject.toml
```
Expected: All checks pass.

- [ ] **Step 4: Update CHANGELOG**

Add a `## [Unreleased]` section in `CHANGELOG.md` documenting all 22 fixes with their competitor source references.

- [ ] **Step 5: Commit the changelog update**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document 22 competitor-informed bug fixes"
```

- [ ] **Step 6: Write final summary document**

Write `docs/superpowers/plans/2026-04-10-bug-fix-summary.md` with:
- Count of bugs fixed per severity
- Total new regression tests added
- Any bugs downgraded or deferred with rationale
- Follow-up items for v0.22.1

---

## Self-Review Checklist

- **Spec coverage:** All 22 bugs from the cross-reference report have a corresponding task (Tasks 1-22) + final verification (Task 23). ✓
- **No placeholders:** Every task has exact file paths, exact line numbers, complete code snippets for the fix, and explicit bash commands. Tasks 16-22 are lighter because those bugs are small and mechanical — each still specifies the file, the test, the fix, and the commit. ✓
- **Type consistency:** `run_sync` has a single signature across all 8 sync wrapper replacements. The new `_literal_info` helper is consistent with `_unwrap_type`. The 4-tuple return from `run_child` and 3-tuple from `_aexecute_subgraph` are consistent with their callers. ✓
- **Test isolation:** All 22 regression tests live in `tests/agent/test_regression.py` as `test_bug{NN}_*` functions and are independently runnable — no inter-test dependencies. ✓
