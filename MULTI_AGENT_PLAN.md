# v0.18.0 Multi-Agent Orchestration — Implementation Plan

> **Status**: Ready for development — foundation work complete
> **Preceding release**: v0.17.4 (Agent Intelligence, shipped 2026-03-22)
> **Target**: Biggest feature since the library started
>
> **Foundation work completed across v0.16.5–v0.17.4**:
> - Agent decomposed into 4 mixins (`core.py` 3128 → 1448 lines)
> - `StepType` is `str, Enum` with 16 members — ready for graph step types
> - `_execute_single_tool` / `_aexecute_single_tool` extracted — graph nodes can reuse
> - `AgentObserver` (31 sync) + `AsyncAgentObserver` (28 async) + `SimpleStepObserver` shipped
> - Terminal action support (`tool.terminal`, `stop_condition`) — useful for HITL in graphs
> - `CancellationToken` — propagatable to child agent nodes for cooperative cancellation
> - `max_total_tokens` / `max_cost_usd` — budget limits propagatable to nodes
> - `model_selector` — per-iteration model switching within nodes
> - `requires_approval` on `@tool()` — per-tool HITL within nodes
> - `KnowledgeMemory` with pluggable stores — shareable across graph nodes
> - `estimate_run_tokens()` — pre-node token budget estimation
> - `GraphExecutionError` already in `exceptions.py` (stub from v0.16.4)
> - MCP client/server shipped (v0.17.1) — MCP tools work in graph nodes
> - Eval framework shipped (v0.17.0) — can evaluate graph outputs
> - 2113 tests, 49 examples, shared test fixtures in conftest.py

## Design Philosophy

LangGraph requires learning StateGraph, MessageAnnotation, Pregel channels, and a custom checkpointing API before building anything. Selectools takes the opposite approach: **agents are the primitive, composition is plain Python**.

**Core principles**:

1. **Agents are nodes, not functions** — each node is a full `Agent` with tools, provider, config, traces, observers, guardrails
2. **Edges are just Python functions** — a routing function takes the state and returns a node name via `if/elif/else`
3. **State is a typed dataclass** — no Pydantic, no annotation magic, just `@dataclass`
4. **Checkpointing is serialization** — JSON-serializable state, 3-method protocol for stores
5. **HITL reuses existing patterns** — the existing `ToolPolicy` + `confirm_action` already handles human-in-the-loop

## Design Decisions

1. **State passing**: Deep copy `state.data` between nodes (isolation for parallel execution). `messages` and `history` are append-only, shared by reference.
2. **Error handling**: Three policies — `abort` (default), `skip`, `retry` — configurable per graph.
3. **Streaming**: `astream()` yields `GraphEvent` tagged unions (`node_start`, `node_end`, `node_chunk`, `routing`, `graph_end`).
4. **Backward compatibility**: `Agent` class is unchanged. Graph is purely additive — no changes to existing APIs.
5. **Cancellation propagation**: Graph checks `CancellationToken` before each node; individual node agents inherit the token via their `AgentConfig`.
6. **Budget propagation**: Graph tracks cumulative usage across all nodes; can enforce graph-level `max_total_tokens` / `max_cost_usd` in addition to per-node limits.

## Module Structure

```
src/selectools/orchestration/
    __init__.py           # Public exports
    state.py              # GraphState, GraphEvent
    node.py               # GraphNode, default transforms
    graph.py              # AgentGraph engine, GraphResult, ErrorPolicy
    checkpoint.py         # CheckpointStore protocol + 3 backends
    supervisor.py         # SupervisorAgent (plan_and_execute, round_robin, dynamic)
```

## Build Order

```
Phase 1: Primitives       → state.py, node.py, orchestration/__init__.py
Phase 2: Graph Engine     → graph.py + exceptions.py update
Phase 3: Checkpointing    → checkpoint.py
Phase 4: Integration      → observer.py, trace.py, __init__.py updates
Phase 5: Supervisor       → supervisor.py
Phase 6: Docs & Release   → docs, examples, CHANGELOG, version bump
```

---

## Phase 1: Primitives

### `src/selectools/orchestration/state.py` (~120 lines)

**`GraphState`** dataclass:

```python
@dataclass
class GraphState:
    messages: List[Message]                       # Accumulated across nodes
    data: Dict[str, Any]                          # Inter-node KV store
    current_node: str = ""                        # Currently executing node
    history: List[Tuple[str, AgentResult]] = ...  # Ordered (node_name, result) pairs
    metadata: Dict[str, Any] = ...                # User-attached, carried through checkpoints
    errors: List[Dict[str, Any]] = ...            # Error records from failed nodes
```

Methods:
- `to_dict()` / `from_dict()` — JSON serialization (uses existing `Message.from_dict()` from `types.py`)
- `from_prompt(prompt: str) -> GraphState` — convenience factory that wraps a string into a state

**`GraphEvent`** dataclass:

```python
@dataclass
class GraphEvent:
    type: Literal["node_start", "node_end", "node_chunk", "routing", "graph_start", "graph_end", "checkpoint"]
    node_name: Optional[str] = None
    chunk: str = ""
    state: Optional[GraphState] = None
    result: Optional["GraphResult"] = None        # Forward ref
    next_node: Optional[str] = None               # For routing events
```

### `src/selectools/orchestration/node.py` (~150 lines)

**`GraphNode`** dataclass:

```python
@dataclass
class GraphNode:
    name: str
    agent: Union[Agent, Callable[[GraphState], GraphState], Callable[[GraphState], Awaitable[GraphState]]]
    input_transform: Optional[Callable[[GraphState], List[Message]]] = None
    output_transform: Optional[Callable[[AgentResult, GraphState], GraphState]] = None
    max_iterations: int = 1    # Re-execution limit in cycles
```

**Standalone defaults:**
- `default_input_transform(state) -> List[Message]` — extracts last user message or synthesizes from prior node output
- `default_output_transform(result, state) -> GraphState` — appends assistant message to `state.messages`, adds `(node_name, result)` to `state.history`

Callable nodes (non-Agent) enable pure-function transforms and sub-graph composition — an `AgentGraph` can expose `__call__` and be used as a node.

### Tests: `tests/test_orchestration_primitives.py` (~45 tests)
- GraphState construction, serialization round-trip, `from_prompt`
- GraphNode construction, default transforms
- Deep copy semantics for `state.data`
- GraphEvent construction

---

## Phase 2: Graph Engine

### `src/selectools/orchestration/graph.py` (~550 lines)

**`GraphResult`** dataclass:

```python
@dataclass
class GraphResult:
    content: str                                      # Final output text
    state: GraphState                                 # Final state
    node_results: Dict[str, List[AgentResult]]        # Per-node results
    trace: AgentTrace                                 # Composite trace
    total_usage: UsageStats                           # Aggregated across all nodes
```

**`ErrorPolicy`** — use `str, Enum` (consistent with `StepType`/`ModelType` pattern, see ADR-003):
```python
class ErrorPolicy(str, Enum):
    ABORT = "abort"
    SKIP = "skip"
    RETRY = "retry"
```

**`AgentGraph`** class:

```python
class AgentGraph:
    END: ClassVar[str] = "__end__"

    def __init__(
        self, name="graph", observers=None, error_policy="abort", max_steps=50,
        cancellation_token=None, max_total_tokens=None, max_cost_usd=None,
    ): ...

    # Node management
    def add_node(self, name, agent_or_callable, input_transform=None, output_transform=None, max_iterations=1): ...
    def add_edge(self, from_node, to_node): ...
    def add_conditional_edge(self, from_node, router_fn: Callable[[GraphState], str]): ...
    def add_parallel_nodes(self, name, node_names: List[str], merge_fn=None): ...
    def set_entry(self, node_name): ...
    def validate(self) -> List[str]: ...

    # Execution
    def run(self, prompt_or_state, checkpoint_store=None, checkpoint_id=None) -> GraphResult: ...
    async def arun(self, prompt_or_state, checkpoint_store=None, checkpoint_id=None) -> GraphResult: ...
    async def astream(self, prompt_or_state, checkpoint_store=None, checkpoint_id=None) -> AsyncGenerator[GraphEvent, None]: ...

    # Composition — makes graph usable as a node in another graph
    def __call__(self, state: GraphState) -> GraphState: ...
```

**Usage example:**

```python
from selectools.orchestration import AgentGraph

graph = AgentGraph()
graph.add_node("planner", planner_agent)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)

graph.add_edge("planner", "researcher")
graph.add_conditional_edge("researcher", lambda state: "writer" if state.data.get("ready") else "researcher")
graph.add_edge("writer", AgentGraph.END)

graph.set_entry("planner")
result = graph.run("Write a blog post about AI agents")
```

**Internal data structures:**

```python
self._nodes: Dict[str, GraphNode] = {}
self._edges: Dict[str, str] = {}                              # from_name -> to_name (static)
self._conditional_edges: Dict[str, Callable] = {}              # from_name -> router_fn
self._parallel_groups: Dict[str, Tuple[List[str], Optional[Callable]]] = {}
self._entry_node: Optional[str] = None
```

**Execution loop (pseudocode):**

```python
state = normalize_input(prompt_or_state)   # str -> GraphState.from_prompt()
if checkpoint_id:
    state, step = checkpoint_store.load(checkpoint_id)
trace = AgentTrace(metadata={"graph_name": self.name})
run_id = trace.run_id
usage = AgentUsage()

current = self._entry_node
step = 0
node_results = {}

while current != END and step < max_steps:
    step += 1
    state.current_node = current

    # Graph-level cancellation check
    if self._cancellation_token and self._cancellation_token.is_cancelled:
        break

    # Graph-level budget check
    if self._max_total_tokens and usage.total_tokens >= self._max_total_tokens:
        break
    if self._max_cost_usd and usage.total_cost_usd >= self._max_cost_usd:
        break

    if current in self._parallel_groups:
        results, state = await self._aexecute_parallel(...)
        node_results.update(results)
    else:
        node = self._nodes[current]
        result, state = await self._aexecute_node(node, state, trace, run_id)
        node_results.setdefault(current, []).append(result)
        usage.merge(result.usage)

    if checkpoint_store:
        checkpoint_store.save(run_id, state, step)

    current = self._resolve_next_node(current, state)

return GraphResult(content=..., state=state, node_results=node_results, trace=trace, total_usage=usage)
```

**Agent integration:** Each node's agent is called via `agent.run(messages)` / `agent.arun(messages)`. The agent's `parent_run_id` is set to the graph's `run_id`, creating trace hierarchy. `input_transform(state)` produces the messages, `output_transform(result, state)` merges results back.

**Parallel execution:** Uses `ThreadPoolExecutor` (sync) / `asyncio.gather` (async), matching the existing pattern in `agent/core.py` for parallel tool execution. Each branch gets `copy.deepcopy(state)`. A `merge_fn(List[GraphState]) -> GraphState` reconciles (default: concatenate messages, shallow-merge data dicts).

### Modify: `src/selectools/exceptions.py`

Add `GraphExecutionError(SelectoolsError)` with `graph_name`, `node_name`, `error`, `step` fields.

### Tests: `tests/test_orchestration_graph.py` (~120 tests)
- Linear graph (A -> B -> C -> END)
- Conditional routing (A -> B or C based on state)
- Cyclic graph with exit condition (A -> B -> A -> END)
- Parallel nodes with default and custom merge
- `arun()` mirrors of all sync tests
- `astream()` yields correct GraphEvent sequence
- Error handling: abort, skip, retry policies
- `max_steps` guard against infinite loops
- Sub-graph composition (AgentGraph as node)
- Parent trace linking (`parent_run_id` propagation)
- State isolation (deep copy verification)
- `validate()` warnings
- Graph-level cancellation (token checked between nodes)
- Graph-level budget limits (stops when cumulative usage exceeds limit)

**Mocking:** Use `LocalProvider` from `providers/stubs.py` for deterministic agent responses.

---

## Phase 3: Checkpointing

### `src/selectools/orchestration/checkpoint.py` (~250 lines)

Follows the `sessions.py` 3-backend pattern exactly.

**`CheckpointStore`** Protocol:

```python
class CheckpointStore(Protocol):
    def save(self, graph_id: str, state: GraphState, step: int) -> str: ...    # Returns checkpoint_id
    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]: ...          # Returns (state, step)
    def list(self, graph_id: str) -> List[CheckpointMetadata]: ...
    def delete(self, checkpoint_id: str) -> bool: ...
```

**`CheckpointMetadata`** dataclass: `checkpoint_id, graph_id, step, node_name, created_at`

**3 Backends:**

| Backend | Storage | Thread safety |
|---------|---------|---------------|
| `InMemoryCheckpointStore` | `Dict[str, ...]` | `threading.Lock` |
| `FileCheckpointStore(directory)` | `{dir}/{graph_id}/{id}.json` | `threading.Lock` + `os.makedirs` |
| `SQLiteCheckpointStore(db_path)` | `checkpoints` table, WAL mode | SQLite-level |

Serialization uses `GraphState.to_dict()`. Traces are excluded from checkpoints (too large, reconstructable from observer events).

### Tests: `tests/test_orchestration_checkpoint.py` (~50 tests)
- All 3 backends: save/load round-trip, list, delete
- Resume execution from checkpoint
- Thread safety for InMemory and File stores
- SQLite concurrent access

---

## Phase 4: Integration

### Modify: `src/selectools/observer.py`

Add 5 new methods to `AgentObserver` (total: 36 sync events, up from 31):

```python
def on_graph_start(self, run_id: str, graph_name: str, entry_node: str, state: Dict[str, Any]) -> None: ...
def on_graph_end(self, run_id: str, graph_name: str, steps: int, total_duration_ms: float) -> None: ...
def on_node_start(self, run_id: str, node_name: str, step: int) -> None: ...
def on_node_end(self, run_id: str, node_name: str, step: int, duration_ms: float) -> None: ...
def on_graph_routing(self, run_id: str, from_node: str, to_node: str, reason: str) -> None: ...
```

Add 5 matching async methods to `AsyncAgentObserver` (total: 33 async events, up from 28).

Update `LoggingObserver` and `SimpleStepObserver` with implementations for all 5 new events.

### Modify: `src/selectools/trace.py`

Add 4 new `StepType` enum members (total: 20, up from 16):

```python
class StepType(str, Enum):
    ...
    GRAPH_NODE_START = "graph_node_start"
    GRAPH_NODE_END = "graph_node_end"
    GRAPH_ROUTING = "graph_routing"
    GRAPH_CHECKPOINT = "graph_checkpoint"
```

The `AgentGraph` creates a root `AgentTrace` for the entire execution. Each node's `Agent` produces its own child `AgentTrace` linked via `parent_run_id`. The root trace captures node-level timeline (`graph_node_start`/`graph_node_end` steps), while child traces contain agent-internal steps (`llm_call`, `tool_execution`, etc.).

### Modify: `src/selectools/__init__.py`

Add exports:

```python
from .orchestration import (
    AgentGraph, GraphState, GraphNode, GraphResult, GraphEvent,
    GraphExecutionError,
    CheckpointStore, InMemoryCheckpointStore, FileCheckpointStore, SQLiteCheckpointStore,
    SupervisorAgent,
)
```

### Tests: `tests/test_orchestration_integration.py` (~40 tests)
- Observer events fire correctly during graph execution
- `LoggingObserver` emits JSON for all 5 new events
- `SimpleStepObserver` routes all 5 new events to callback
- New StepTypes appear in traces
- Root trace has graph steps, child traces have agent steps
- Public exports importable from `selectools` and `selectools.orchestration`

---

## Phase 5: Supervisor

### `src/selectools/orchestration/supervisor.py` (~300 lines)

```python
SupervisorStrategy = Literal["plan_and_execute", "round_robin", "dynamic"]

class SupervisorAgent:
    def __init__(
        self, agents: Dict[str, Agent], provider: Provider, strategy="plan_and_execute",
        max_rounds=10, cancellation_token=None, max_total_tokens=None, max_cost_usd=None,
    ): ...
    def run(self, prompt: str) -> GraphResult: ...
    async def arun(self, prompt: str) -> GraphResult: ...
    async def astream(self, prompt: str) -> AsyncGenerator[GraphEvent, None]: ...
```

All three strategies internally build and execute an `AgentGraph` — the supervisor is a convenience wrapper, not a separate execution engine:

| Strategy | How it works |
|----------|-------------|
| `plan_and_execute` | Supervisor LLM generates JSON plan with `{agent_name, task}` steps -> linear chain of agent nodes |
| `round_robin` | Each agent participates in each round, supervisor checks after each round whether to continue |
| `dynamic` | Router node uses LLM call to pick the best agent per step based on current state |

**Usage example:**

```python
from selectools.orchestration import SupervisorAgent

supervisor = SupervisorAgent(
    agents={"researcher": researcher, "writer": writer, "reviewer": reviewer},
    provider=OpenAIProvider(),
    strategy="plan_and_execute",
)
result = supervisor.run("Write a comprehensive blog post about AI safety")
```

### Tests: `tests/test_orchestration_supervisor.py` (~50 tests)
- `plan_and_execute` with mock LLM plan generation
- `round_robin` full cycle
- `dynamic` routing
- Error propagation from underlying graph
- Usage aggregation across supervisor runs
- Trace hierarchy: supervisor -> graph -> nodes
- `astream()` yields correct event sequence
- Cancellation propagation to graph and node agents
- Budget enforcement at supervisor level

---

## Phase 6: Docs & Release

### Documentation (per CLAUDE.md Feature Development Checklist)
- **New**: `docs/modules/ORCHESTRATION.md` — full module doc
- **Update**: `docs/ARCHITECTURE.md` — add orchestration component diagram
- **Update**: `docs/QUICKSTART.md` — multi-agent quickstart section
- **Update**: `docs/index.md` — feature table, test/example counts
- **Update**: `notebooks/getting_started.ipynb` — orchestration section

### Examples (next available: 50)
- `examples/50_agent_graph.py` — basic linear graph
- `examples/51_parallel_agents.py` — parallel node execution
- `examples/52_conditional_routing.py` — conditional edges
- `examples/53_supervisor_agent.py` — supervisor patterns
- `examples/54_graph_checkpointing.py` — checkpoint save/resume

### Release Artifacts
- Version bump: `__init__.py` + `pyproject.toml` -> `0.18.0`
- `CHANGELOG.md` entry
- `README.md` — "What's New", test count, example count
- `ROADMAP.md` — mark v0.18.0 as completed
- Git tag: `v0.18.0`
- PyPI: `python3 -m build && python3 -m twine upload dist/*`

---

## How It Beats LangGraph

| LangGraph | Selectools AgentGraph | Why better |
|-----------|----------------------|------------|
| Custom `StateGraph` with `Annotated[list, add_messages]` | Plain `GraphState` dataclass | No custom type system to learn |
| `conditionalEdges` with special return constants | Plain Python function returning a string | Debuggable, testable, IDE-friendly |
| Pregel channels for state management | `Dict[str, Any]` with merge functions | Standard Python data structures |
| Separate `compile()` step before execution | Validate + run in one step | No compilation phase, faster iteration |
| `MemorySaver` / `SqliteSaver` / `PostgresSaver` | `CheckpointStore` protocol (4 methods) | Trivial to implement custom stores |
| Node functions receive raw state | Nodes are full `Agent` instances | Inherit all Agent features: tools, traces, observers, guardrails, budget, cancellation |
| Complex interrupt/resume for HITL | Reuse existing `confirm_action` + `requires_approval` | Zero new concepts for HITL |
| Sub-graphs require `CompiledGraph` nesting | `AgentGraph.__call__` makes any graph a node | Natural composition via duck typing |
| No built-in budget enforcement | Graph-level `max_total_tokens` / `max_cost_usd` | Cost control across entire workflow |
| No cooperative cancellation | `CancellationToken` propagates to all nodes | Cancel entire workflow from any thread |

---

## Files Summary

### New (8 source + 5 test + 5 examples)

| File | Est. lines |
|------|------------|
| `src/selectools/orchestration/__init__.py` | ~30 |
| `src/selectools/orchestration/state.py` | ~120 |
| `src/selectools/orchestration/node.py` | ~150 |
| `src/selectools/orchestration/graph.py` | ~550 |
| `src/selectools/orchestration/checkpoint.py` | ~250 |
| `src/selectools/orchestration/supervisor.py` | ~300 |
| `tests/test_orchestration_primitives.py` | ~45 tests |
| `tests/test_orchestration_graph.py` | ~120 tests |
| `tests/test_orchestration_checkpoint.py` | ~50 tests |
| `tests/test_orchestration_integration.py` | ~40 tests |
| `tests/test_orchestration_supervisor.py` | ~50 tests |
| `examples/50-54_*.py` | 5 examples |

### Modified (4)

| File | Changes |
|------|---------|
| `src/selectools/observer.py` | 5 new sync + 5 async events, `LoggingObserver` + `SimpleStepObserver` |
| `src/selectools/trace.py` | 4 new `StepType` members (16 → 20) |
| `src/selectools/__init__.py` | New exports + version bump |
| `pyproject.toml` | Version bump |

> **Note**: `GraphExecutionError` already exists in `exceptions.py` — no changes needed.
>
> Agent code lives across 5 files (`agent/core.py` + 4 mixins).
> Graph node execution calls `agent.arun()` / `agent.run()` — no need to interact with mixins directly.

### ~305 new tests (total: ~2418, up from 2113)

---

## Verification

After each phase:

1. `black src/ tests/ --line-length=100 && isort src/ tests/ --profile=black --line-length=100`
2. `flake8 src/ && mypy src/`
3. `pytest tests/ -x -q` — ALL must pass

After all phases:

4. `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build` — verify docs build
5. Manual smoke test: create a 3-node linear graph with `LocalProvider` agents, verify `GraphResult` contains all node results and aggregated usage

---

## Resolved Design Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| `GraphState.data` deep copy vs shallow? | Deep copy (default) | Safety for parallel execution; opt-in shallow not worth the API complexity |
| Parallel streaming: interleaved or buffered? | Buffer per-branch | Matches existing parallel tool execution pattern; simpler consumer code |
| Graph-level guardrails? | Yes, optional | `input_guardrails` before first node, `output_guardrails` after last node |
| `SupervisorAgent.astream()`? | Yes | Consistency with `Agent` and `AgentGraph` APIs |
| `max_steps=50` default? | Keep 50 | Configurable per-graph; 50 handles most use cases without infinite loop risk |
| Cancellation propagation? | Automatic | Graph token propagates to node agents; single cancel stops everything |
