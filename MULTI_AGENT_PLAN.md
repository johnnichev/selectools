# v0.18.0 Multi-Agent Orchestration — Implementation Plan

> **Status**: Ready for development — foundation complete, architecture stress-tested
> **Preceding release**: v0.17.7 (SemanticCache, Prompt Compression, Branching — shipped 2026-03-25)
> **Target**: Biggest feature since launch. Closes the single largest competitive gap.
>
> **Foundation work completed across v0.16.5–v0.17.7**:
> - Agent decomposed into 4 mixins (`core.py` 3128 → 1448 lines)
> - `StepType` is `str, Enum` with 17 members — ready for graph step types
> - `_execute_single_tool` / `_aexecute_single_tool` extracted — graph nodes can reuse
> - `AgentObserver` (32 sync) + `AsyncAgentObserver` (29 async) + `SimpleStepObserver` shipped
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
> - 2275 tests, 54 examples, shared test fixtures in conftest.py

---

## Why This Beats LangGraph (Evidence)

This section is the design brief. Every architecture decision traces back to a specific LangGraph failure mode.

### Pain Point 1: HITL Double-Execution Bug (P0 in LangGraph)

LangGraph's `interrupt()` **restarts the entire node from the beginning** on resume. This is a documented foot-gun: if your node made an API call, sent an email, or stored a record before calling `interrupt()`, those operations re-execute on resume.

From LangGraph's own docs: *"When using interrupt, the node function is re-executed from the beginning when the graph resumes."* Maintainers acknowledge this as intended behavior.

**Our approach**: Generator nodes. A node is a Python generator that `yield`s an `InterruptRequest`. When the graph resumes, it fast-forwards to the correct yield point using cached intermediate state. Side effects before the yield are naturally protected because they were stored in `state.data` (the convention the pattern encourages) and are loaded from checkpoint on resume.

```python
async def review_node(state: GraphState) -> AsyncGenerator:
    # On re-run after resume, state.data["analysis"] is already populated
    # so this expensive call is skipped automatically
    if "analysis" not in state.data:
        state.data["analysis"] = await run_analysis(state.data["draft"])

    # Execution pauses here; state is checkpointed with analysis already stored
    approval = yield InterruptRequest(
        prompt="Approve this draft?",
        payload=state.data["analysis"],
    )

    state.data["approved"] = (approval == "yes")
    return state  # generator return value becomes new state
```

This is the single biggest differentiation opportunity in orchestration.

### Pain Point 2: StateGraph Complexity

LangGraph requires: `TypedDict` with `Annotated[list, add_messages]`, `StateGraph(MyState)`, `compile()`, `MemorySaver`, understanding Pregel channels, and the difference between `Command(goto=..., update=...)` and `Command(resume=...)`. You learn the framework before you build anything.

**Our approach**: `AgentGraph` with a `GraphState` dataclass. Edges are plain Python functions. No compilation step.

```python
graph = AgentGraph()
graph.add_node("planner", planner_agent)
graph.add_node("writer", writer_agent)
graph.add_edge("planner", "writer")
graph.add_edge("writer", AgentGraph.END)
graph.set_entry("planner")
result = graph.run("Write a blog post about AI agents")
```

### Pain Point 3: `@task` Caching Silently Breaks in Production

LangGraph's `@task` decorator (functional API) uses an in-memory cache that works in dev but silently breaks in production because the task graph is re-compiled on each Lambda/container invocation, clearing all cached state. This wasted multiple engineering weeks for multiple teams (documented in issues and community posts).

**Our approach**: Checkpointing is explicit and explicit only. `InMemoryCheckpointStore`, `FileCheckpointStore`, `SQLiteCheckpointStore` — you pick one. No silent caching.

### Pain Point 4: Paid Platform Lock-In

LangGraph Studio (visual debugging), cron scheduling, smart caching, and auth are paid LangGraph Platform features. The free tier has no persistent state, no Studio, no observability beyond basic logs.

**Our approach**: All features are self-hosted by default. `AgentGraph.to_mermaid()` gives you a diagram without any cloud dependency. `FileCheckpointStore` gives you persistence with a directory path.

### Pain Point 5: Documentation Fragmentation and API Churn

LangGraph has separate docs for LangGraph, LangGraph Platform, LangGraph Cloud, and LangGraph Studio. The Python API changed significantly between 0.1.x, 0.2.x, and 1.x, breaking thousands of tutorials.

**Our approach**: Single package, single docs site, semver. The `Agent` API from v0.13.0 still works unchanged in v0.18.0.

---

## Design Philosophy

LangGraph requires learning StateGraph, MessageAnnotation, Pregel channels, and a custom checkpointing API before building anything. Selectools takes the opposite approach: **agents are the primitive, composition is plain Python**.

**Core principles**:

1. **Agents are nodes, not functions** — each node is a full `Agent` with tools, provider, config, traces, observers, guardrails, budget limits
2. **Edges are Python functions** — routing takes the state and returns a node name via `if/elif/else`; no magic return types
3. **State is a dataclass** — no Pydantic, no annotation magic, no TypedDict gymnastics (TypedDict reducers are opt-in for power users)
4. **Checkpointing is serialization** — JSON-serializable state, 3-method protocol, trivial to implement custom stores
5. **HITL via generators** — `yield InterruptRequest(...)` pauses execution at the exact yield point; `graph.resume(id, data)` continues from there

---

## Design Decisions

1. **State passing**: Deep-copy `state.data` between parallel branches (isolation). `messages` is append-only; `history` is append-only. Deep copy is opt-out, not opt-in.
2. **Error handling**: Three policies — `abort` (default), `skip`, `retry` — configurable per graph and overridable per node.
3. **Streaming**: `astream()` yields `GraphEvent` typed union.
4. **Backward compatibility**: `Agent` class is unchanged. Graph is purely additive.
5. **Cancellation propagation**: Graph checks `CancellationToken` before each node; individual node agents inherit the token via their `AgentConfig`.
6. **Budget propagation**: Graph tracks cumulative usage across all nodes; enforces graph-level `max_total_tokens` / `max_cost_usd` in addition to per-node limits.
7. **Parallel execution**: `asyncio.gather` in async paths; `ThreadPoolExecutor` in sync paths (matching existing pattern for parallel tool execution). Each branch receives `copy.deepcopy(state)`.
8. **State merge for parallel branches**: `MergePolicy` enum (default: `LAST_WINS` on conflicts, `APPEND` on lists). Custom `merge_fn` overrides.
9. **Default input/output transforms**: Canonical key `STATE_KEY_LAST_OUTPUT = "__last_output__"` in `state.data`. `default_output_transform` stores `result.content` there. `default_input_transform` reads it; falls back to last user message in `state.messages` if absent (first-node case).
10. **Parallel groups are first-class nodes**: `add_parallel_nodes("fan_out", [...])` registers a `ParallelGroupNode` in `self._nodes["fan_out"]`. Edges reference it like any other node name. This fixes `add_edge("upstream", "fan_out")` validation.
11. **Async callable dispatch**: `_aexecute_node` dispatches based on type: `Agent` → `agent.arun()`, async callable → `await fn(state)`, async generator → interrupt machinery, sync callable → `loop.run_in_executor(None, fn, state)`.
12. **`path_map` for conditional edges**: Optional dict mapping router return values to node names. Enables compile-time validation.
13. **Per-node visit limits**: `GraphNode.max_visits = 0` (0 = unlimited). Prevents runaway loops without relying solely on global `max_steps`.

---

## Module Structure

```
src/selectools/orchestration/
├── __init__.py           # Public exports
├── state.py              # GraphState, GraphEvent, MergePolicy, InterruptRequest, Scatter,
│                         # routing primitives (goto, update), STATE_KEY_LAST_OUTPUT
├── node.py               # GraphNode, ParallelGroupNode, SubgraphNode, default transforms
├── graph.py              # AgentGraph, GraphResult, ErrorPolicy
├── checkpoint.py         # CheckpointStore protocol + 3 backends, CheckpointMetadata
└── supervisor.py         # SupervisorAgent (plan_and_execute, round_robin, dynamic)
```

---

## Build Order

```
Phase 1: Primitives       → state.py, node.py, orchestration/__init__.py
Phase 2: Graph Engine     → graph.py + exceptions.py update
Phase 3: Interrupt/Resume → graph.py (interrupt machinery), checkpoint.py (interrupt state)
Phase 4: Checkpointing    → checkpoint.py (full 3-backend impl)
Phase 5: Integration      → observer.py, trace.py, __init__.py updates
Phase 6: Supervisor       → supervisor.py
Phase 7: Visualization    → AgentGraph.to_mermaid(), visualize()
Phase 8: Docs & Release   → docs, examples, CHANGELOG, version bump
```

---

## Phase 1: Primitives

### `src/selectools/orchestration/state.py` (~180 lines)

```python
STATE_KEY_LAST_OUTPUT: str = "__last_output__"  # canonical key for inter-node handoff


class MergePolicy(str, Enum):
    LAST_WINS = "last_wins"   # conflicting keys: last parallel branch wins
    FIRST_WINS = "first_wins" # conflicting keys: first result wins
    APPEND = "append"         # list values are appended, others use LAST_WINS


@dataclass
class GraphState:
    messages: List[Message]                        # accumulated across nodes (append-only)
    data: Dict[str, Any] = field(default_factory=dict)   # inter-node KV store
    current_node: str = ""                         # currently executing node name
    history: List[Tuple[str, AgentResult]] = field(default_factory=list)  # (node_name, result)
    metadata: Dict[str, Any] = field(default_factory=dict)  # user-attached, carried via checkpoints
    errors: List[Dict[str, Any]] = field(default_factory=list)   # error records from failed nodes
    _interrupt_responses: Dict[str, Any] = field(default_factory=dict, repr=False)  # keyed by interrupt_key

    def to_dict(self) -> Dict[str, Any]: ...    # JSON-safe; excludes _interrupt_responses
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphState": ...
    @classmethod
    def from_prompt(cls, prompt: str) -> "GraphState": ...  # wraps into messages[0]


@dataclass
class InterruptRequest:
    """Yielded from generator nodes to pause execution for human input."""
    prompt: str
    payload: Any = None
    interrupt_key: str = ""   # auto-set by graph to f"{node_name}_{yield_index}"


@dataclass
class Scatter:
    """Returned from routing functions to create dynamic parallel branches."""
    node_name: str
    state_patch: Dict[str, Any] = field(default_factory=dict)  # merged into each branch's state.data


@dataclass
class GraphEvent:
    type: str   # GraphEventType enum value (str, Enum — consistent with StepType)
    node_name: Optional[str] = None
    chunk: str = ""
    state: Optional[GraphState] = None
    result: Optional["GraphResult"] = None
    next_node: Optional[str] = None
    error: Optional[Exception] = None
    interrupt_id: Optional[str] = None


class GraphEventType(str, Enum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_CHUNK = "node_chunk"
    ROUTING = "routing"
    GRAPH_START = "graph_start"
    GRAPH_END = "graph_end"
    GRAPH_INTERRUPT = "graph_interrupt"
    GRAPH_RESUME = "graph_resume"
    PARALLEL_START = "parallel_start"
    PARALLEL_END = "parallel_end"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
```

**Routing primitives** (light-weight, no DSL):

```python
@dataclass
class _Goto:
    node_name: str

@dataclass
class _Update:
    patch: Dict[str, Any]

def goto(node_name: str) -> _Goto: ...
def update(patch: Dict[str, Any]) -> _Update: ...
```

Routing functions can return: `str` (node name), `AgentGraph.END`, `Scatter`, `List[Scatter]` (dynamic fan-out), or `_Goto`. State-mutation routing is done via `_Update` combined with the node's output transform. We deliberately do NOT add `|` operator here — that belongs in the v0.18.x composability layer.

### `src/selectools/orchestration/node.py` (~200 lines)

```python
@dataclass
class GraphNode:
    name: str
    agent: Union[
        Agent,
        Callable[[GraphState], GraphState],               # sync callable
        Callable[[GraphState], Awaitable[GraphState]],    # async callable
        AsyncGenerator,                                    # generator node (HITL)
    ]
    input_transform: Optional[Callable[[GraphState], List[Message]]] = None
    output_transform: Optional[Callable[[AgentResult, GraphState], GraphState]] = None
    max_iterations: int = 1    # re-execution limit in a cycle (distinct from graph max_steps)
    max_visits: int = 0        # 0 = unlimited; graph raises if exceeded
    error_policy: Optional["ErrorPolicy"] = None   # None = inherit from graph


@dataclass
class ParallelGroupNode:
    """Registered in AgentGraph._nodes as a first-class node.

    When the graph resolves this node, it fans out to child_node_names,
    executes them in parallel, and merges results via merge_policy or merge_fn.
    """
    name: str
    child_node_names: List[str]
    merge_policy: MergePolicy = MergePolicy.LAST_WINS
    merge_fn: Optional[Callable[[List[GraphState]], GraphState]] = None


@dataclass
class SubgraphNode:
    """Wraps an AgentGraph as a node with explicit key mapping.

    input_map: parent state.data key → subgraph state.data key
    output_map: subgraph state.data key → parent state.data key
    """
    name: str
    graph: "AgentGraph"
    input_map: Dict[str, str] = field(default_factory=dict)
    output_map: Dict[str, str] = field(default_factory=dict)
```

**Default transforms:**

```python
STATE_KEY_LAST_OUTPUT = "__last_output__"  # imported from state.py

def default_input_transform(state: GraphState) -> List[Message]:
    """Reads STATE_KEY_LAST_OUTPUT from state.data; falls back to last user message."""
    last = state.data.get(STATE_KEY_LAST_OUTPUT)
    if last and isinstance(last, str):
        return [Message(role=Role.USER, content=last)]
    # First node: use last user message from state.messages
    for msg in reversed(state.messages):
        if msg.role == Role.USER:
            return [msg]
    return list(state.messages)   # fallback: pass everything


def default_output_transform(result: AgentResult, state: GraphState) -> GraphState:
    """Appends assistant message; writes STATE_KEY_LAST_OUTPUT; records history."""
    state.messages.append(Message(role=Role.ASSISTANT, content=result.content))
    state.data[STATE_KEY_LAST_OUTPUT] = result.content
    state.history.append((state.current_node, result))
    return state
```

**State merge:**

```python
def merge_states(states: List[GraphState], policy: MergePolicy) -> GraphState:
    """Merge parallel branch results into a single GraphState.

    messages: always concatenated (append semantics)
    history:  always concatenated
    errors:   always concatenated
    data:     merged per policy (LAST_WINS / FIRST_WINS / APPEND)
    """
```

### Tests: `tests/test_orchestration_primitives.py` (~55 tests)
- `GraphState` construction, `to_dict`/`from_dict` round-trip, `from_prompt`
- `_interrupt_responses` excluded from serialization
- `GraphNode`, `ParallelGroupNode`, `SubgraphNode` construction
- `default_input_transform`: first-node fallback, subsequent-node handoff, empty state
- `default_output_transform`: `STATE_KEY_LAST_OUTPUT` written, history appended
- `merge_states`: LAST_WINS, FIRST_WINS, APPEND policies; list handling
- `InterruptRequest`, `Scatter`, `GraphEvent`, `GraphEventType` construction
- Deep-copy semantics for `state.data`

---

## Phase 2: Graph Engine

### `src/selectools/orchestration/graph.py` (~650 lines)

**`GraphResult`**:

```python
@dataclass
class GraphResult:
    content: str                                      # last node's result.content
    state: GraphState                                 # final state
    node_results: Dict[str, List[AgentResult]]        # per-node results
    trace: AgentTrace                                 # composite graph-level trace
    total_usage: UsageStats                           # aggregated across all nodes
    interrupted: bool = False                         # True if execution paused for HITL
    interrupt_id: Optional[str] = None               # checkpoint_id to pass to graph.resume()
    steps: int = 0                                    # total iterations executed
```

**`ErrorPolicy`**:

```python
class ErrorPolicy(str, Enum):
    ABORT = "abort"    # raise immediately (default)
    SKIP = "skip"      # log and continue to next node (state unchanged)
    RETRY = "retry"    # retry up to error_retry_limit times, then abort
```

**`AgentGraph`**:

```python
class AgentGraph:
    END: ClassVar[str] = "__end__"

    def __init__(
        self,
        name: str = "graph",
        observers: Optional[List[AgentObserver]] = None,
        error_policy: ErrorPolicy = ErrorPolicy.ABORT,
        error_retry_limit: int = 3,
        max_steps: int = 50,
        cancellation_token: Optional[CancellationToken] = None,
        max_total_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        input_guardrails: Optional[GuardrailsPipeline] = None,
        output_guardrails: Optional[GuardrailsPipeline] = None,
    ): ...

    # --- Node management ---

    def add_node(
        self, name: str,
        agent_or_callable: Union[Agent, Callable],
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        max_iterations: int = 1,
        max_visits: int = 0,
        error_policy: Optional[ErrorPolicy] = None,
    ) -> None: ...

    def add_parallel_nodes(
        self, name: str,
        node_names: List[str],
        merge_policy: MergePolicy = MergePolicy.LAST_WINS,
        merge_fn: Optional[Callable[[List[GraphState]], GraphState]] = None,
    ) -> None:
        """Register a ParallelGroupNode in self._nodes[name]. Validates child names exist."""

    def add_subgraph(
        self, name: str, graph: "AgentGraph",
        input_map: Optional[Dict[str, str]] = None,
        output_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register a SubgraphNode in self._nodes[name]."""

    def add_edge(self, from_node: str, to_node: str) -> None: ...

    def add_conditional_edge(
        self,
        from_node: str,
        router_fn: Callable[[GraphState], Union[str, Scatter, List[Scatter]]],
        path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """path_map enables compile-time validation that all router return values are valid nodes."""

    def set_entry(self, node_name: str) -> None: ...

    def validate(self) -> List[str]:
        """Returns list of warnings (not errors). Called automatically in run()."""

    # --- Execution ---

    def run(
        self, prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
    ) -> GraphResult: ...

    async def arun(
        self, prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
    ) -> GraphResult: ...

    async def astream(
        self, prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
    ) -> AsyncGenerator[GraphEvent, None]: ...

    def resume(
        self, interrupt_id: str, response: Any,
        checkpoint_store: "CheckpointStore",
    ) -> GraphResult:
        """Resume execution after an InterruptRequest. Loads checkpoint, injects response."""

    async def aresume(
        self, interrupt_id: str, response: Any,
        checkpoint_store: "CheckpointStore",
    ) -> GraphResult: ...

    # --- Composition ---

    def __call__(self, state: GraphState) -> GraphState:
        """Makes graph usable as a callable node in another graph."""
```

**Internal data structures:**

```python
self._nodes: Dict[str, Union[GraphNode, ParallelGroupNode, SubgraphNode]] = {}
self._edges: Dict[str, str] = {}                     # from_name → to_name (static)
self._conditional_edges: Dict[str, Callable] = {}    # from_name → router_fn
self._path_maps: Dict[str, Dict[str, str]] = {}      # from_name → path_map dict
self._entry_node: Optional[str] = None
```

**Execution loop (pseudocode):**

```python
state = _normalize_input(prompt_or_state)
if checkpoint_id:
    state, step = checkpoint_store.load(checkpoint_id)
    state = _restore_interrupt_responses(state, interrupt_response)  # for resume

trace = AgentTrace(metadata={"graph_name": self.name})
run_id = str(uuid4())
usage = UsageStats()
node_results: Dict[str, List[AgentResult]] = {}
_visit_counts: Dict[str, int] = {}

_notify("on_graph_start", run_id, self.name, self._entry_node, state.to_dict())
_trace_step(StepType.GRAPH_NODE_START, ...)   # using GRAPH_NODE_START for graph-level start

current = self._entry_node
step = 0

while current != AgentGraph.END and step < self.max_steps:
    step += 1
    state.current_node = current

    # cancellation check
    if self._cancellation_token?.is_cancelled:
        break

    # graph-level budget check
    if _over_budget(usage):
        break

    # per-node visit count
    _visit_counts[current] = _visit_counts.get(current, 0) + 1
    node = self._nodes[current]
    if isinstance(node, GraphNode) and node.max_visits > 0:
        if _visit_counts[current] > node.max_visits:
            raise GraphExecutionError(f"Node {current!r} exceeded max_visits={node.max_visits}")

    _notify("on_node_start", run_id, current, step)
    _trace_step(StepType.GRAPH_NODE_START, node_name=current, step=step)

    try:
        if isinstance(node, ParallelGroupNode):
            child_results, state = await _aexecute_parallel(node, state, trace, run_id)
            node_results.update(child_results)
        elif isinstance(node, SubgraphNode):
            result, state = await _aexecute_subgraph(node, state, trace, run_id)
            node_results.setdefault(current, []).append(result)
        else:
            result, state, interrupted = await _aexecute_node(node, state, trace, run_id)
            node_results.setdefault(current, []).append(result)
            usage = usage.merge(result.usage)
            if interrupted:
                # save checkpoint and return GraphResult(interrupted=True)
                checkpoint_id = checkpoint_store.save(run_id, state, step)
                _notify("on_graph_interrupt", run_id, current, checkpoint_id)
                return GraphResult(..., interrupted=True, interrupt_id=checkpoint_id)

    except Exception as e:
        # handle per error_policy (abort/skip/retry)
        ...

    _notify("on_node_end", run_id, current, step, duration_ms)
    _trace_step(StepType.GRAPH_NODE_END, node_name=current, step=step, duration_ms=...)

    if checkpoint_store:
        cid = checkpoint_store.save(run_id, state, step)
        _trace_step(StepType.GRAPH_CHECKPOINT, checkpoint_id=cid)

    current = _resolve_next_node(current, state)
    _notify("on_graph_routing", run_id, current_prev, current, reason="")
    _trace_step(StepType.GRAPH_ROUTING, from_node=current_prev, to_node=current)

_notify("on_graph_end", run_id, self.name, step, total_duration_ms)
return GraphResult(content=state.data.get(STATE_KEY_LAST_OUTPUT, ""), ...)
```

**`_aexecute_node` dispatch:**

```python
async def _aexecute_node(
    self, node: GraphNode, state: GraphState, trace: AgentTrace, run_id: str
) -> Tuple[AgentResult, GraphState, bool]:   # (result, new_state, interrupted)

    messages = (node.input_transform or default_input_transform)(state)

    if isinstance(node.agent, Agent):
        result = await node.agent.arun(messages, parent_run_id=run_id)
        new_state = (node.output_transform or default_output_transform)(result, state)
        return result, new_state, False

    elif asyncio.iscoroutinefunction(node.agent):
        new_state = await node.agent(state)
        result = _make_synthetic_result(new_state)
        return result, new_state, False

    elif inspect.isasyncgenfunction(node.agent):
        # Generator node: HITL support
        return await _aexecute_generator_node(node, state, trace, run_id)

    elif inspect.isgeneratorfunction(node.agent):
        # Sync generator: run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute_generator_node_sync, node, state, trace, run_id)

    else:
        # Plain sync callable: run in executor
        loop = asyncio.get_event_loop()
        new_state = await loop.run_in_executor(None, node.agent, state)
        result = _make_synthetic_result(new_state)
        return result, new_state, False
```

**`_aexecute_generator_node` (HITL core):**

```python
async def _aexecute_generator_node(node, state, trace, run_id):
    """Execute async generator node, handling InterruptRequest yields."""
    gen = node.agent(state)
    interrupt_index = 0

    async for value in gen:
        if isinstance(value, InterruptRequest):
            # Check if we have a pre-loaded response for this interrupt
            interrupt_key = f"{node.name}_{interrupt_index}"
            if interrupt_key in state._interrupt_responses:
                # Resume path: inject stored response
                try:
                    value = await gen.asend(state._interrupt_responses[interrupt_key])
                    interrupt_index += 1
                    continue
                except StopAsyncIteration:
                    break
            else:
                # First-pass path: save state and signal interrupt
                value.interrupt_key = interrupt_key
                _notify("on_graph_interrupt", run_id, node.name, interrupt_key)
                _trace_step(StepType.GRAPH_INTERRUPT, node_name=node.name, interrupt_key=interrupt_key)
                synthetic = _make_synthetic_result(state)
                return synthetic, state, True  # interrupted=True
        interrupt_index = 0   # reset on non-interrupt yield (shouldn't happen but safe)

    # Generator completed via StopAsyncIteration; final value is in gen.ag_return (Python 3.12+)
    # or we rely on generator setting state.data directly
    result = _make_synthetic_result(state)
    return result, state, False
```

**`_aexecute_parallel`:**

```python
async def _aexecute_parallel(self, node: ParallelGroupNode, state, trace, run_id):
    """Fan out to child nodes in parallel using asyncio.gather."""
    _notify("on_parallel_start", run_id, node.name, node.child_node_names)
    _trace_step(StepType.GRAPH_PARALLEL_START, node_name=node.name, children=node.child_node_names)

    branch_states = [copy.deepcopy(state) for _ in node.child_node_names]

    async def run_child(child_name, branch_state):
        child_node = self._nodes[child_name]
        result, new_state, interrupted = await self._aexecute_node(child_node, branch_state, trace, run_id)
        return child_name, result, new_state

    child_outputs = await asyncio.gather(*[
        run_child(name, bstate)
        for name, bstate in zip(node.child_node_names, branch_states)
    ])

    child_results = {}
    branch_final_states = []
    for child_name, result, new_state in child_outputs:
        child_results.setdefault(child_name, []).append(result)
        branch_final_states.append(new_state)

    if node.merge_fn:
        merged = node.merge_fn(branch_final_states)
    else:
        merged = merge_states(branch_final_states, node.merge_policy)

    _notify("on_parallel_end", run_id, node.name, len(child_outputs))
    _trace_step(StepType.GRAPH_PARALLEL_END, node_name=node.name)
    return child_results, merged
```

**Routing resolution** — handles static edges, conditional edges, Scatter, `path_map` validation:

```python
def _resolve_next_node(self, current: str, state: GraphState) -> str:
    if current in self._conditional_edges:
        router = self._conditional_edges[current]
        result = router(state)
        if isinstance(result, list) and result and isinstance(result[0], Scatter):
            # Dynamic fan-out: build parallel group on the fly and execute
            return self._handle_scatter(result, state)
        if isinstance(result, Scatter):
            return self._handle_scatter([result], state)
        # Validate against path_map if provided
        if current in self._path_maps:
            pmap = self._path_maps[current]
            if pmap and result not in pmap and result != AgentGraph.END:
                raise GraphExecutionError(
                    f"Router for {current!r} returned {result!r} but path_map is {list(pmap.keys())}"
                )
            return pmap.get(result, result)
        return result
    if current in self._edges:
        return self._edges[current]
    return AgentGraph.END  # no outgoing edge = implicit END
```

**Usage example (after implementation):**

```python
from selectools.orchestration import AgentGraph, GraphState, InterruptRequest

graph = AgentGraph(name="review_pipeline")
graph.add_node("drafter", drafter_agent)
graph.add_node("reviewer", review_node)  # generator node with yield InterruptRequest
graph.add_node("publisher", publisher_agent)

graph.add_edge("drafter", "reviewer")
graph.add_conditional_edge(
    "reviewer",
    lambda state: "publisher" if state.data.get("approved") else "drafter",
    path_map={"publisher": "publisher", "drafter": "drafter"},
)
graph.add_edge("publisher", AgentGraph.END)
graph.set_entry("drafter")

# First run — pauses at review_node
store = FileCheckpointStore("./checkpoints")
result = graph.run("Write a blog post about AI safety", checkpoint_store=store)
assert result.interrupted
print(result.state.data["analysis"])  # human reviews this

# Resume with approval
final = graph.resume(result.interrupt_id, response="yes", checkpoint_store=store)
print(final.content)
```

### Modify: `src/selectools/exceptions.py`

`GraphExecutionError(SelectoolsError)` already exists as a stub. Add fields:

```python
@dataclass
class GraphExecutionError(SelectoolsError):
    message: str
    graph_name: str = ""
    node_name: str = ""
    step: int = 0
    cause: Optional[Exception] = None
```

### Tests: `tests/test_orchestration_graph.py` (~130 tests)
- Linear graph: A → B → C → END
- Conditional routing: validates `path_map`, catches invalid router returns
- Dynamic fan-out via `Scatter`: graph builds parallel group on the fly
- Cyclic graph with exit condition: `max_steps` guard + per-node `max_visits`
- Parallel nodes: `add_parallel_nodes`, fan-out/merge, each MergePolicy
- `arun()` mirrors of all sync tests
- `astream()` yields `GraphEvent` sequence in correct order
- Error handling: abort/skip/retry; `GraphExecutionError` fields populated
- Sub-graph composition: `add_subgraph` with `input_map`/`output_map`
- `ParallelGroupNode` registered in `_nodes` — `add_edge` to parallel group validates
- `_aexecute_node`: Agent, async callable, sync callable, async generator dispatch
- State isolation: deep copy verified for parallel branches
- Parent trace linking: `parent_run_id` propagated from graph `run_id`
- `validate()` warnings for unreachable nodes, missing entry
- `__call__`: graph usable as a node in another graph
- Graph-level cancellation token propagation
- Graph-level budget enforcement (total_tokens, cost_usd)

---

## Phase 3: Interrupt & Resume

This is the centerpiece feature. Implemented as part of `graph.py` and `checkpoint.py`.

### `InterruptRequest` flow (full specification):

**First-pass execution:**
1. Generator node yields `InterruptRequest(prompt, payload)`
2. `_aexecute_generator_node` sees a yielded `InterruptRequest` with no matching `_interrupt_responses` key
3. `state.data` at this point contains all intermediate values the node computed and stored before the yield
4. Graph calls `checkpoint_store.save(run_id, state, step)` — this checkpoint includes the current `state.data` with intermediate values AND the interrupt key
5. Returns `GraphResult(interrupted=True, interrupt_id=checkpoint_id, state=state)`
6. Human reads `result.state.data` or `result.interrupt_id` and makes a decision

**Resume path:**
1. `graph.resume(interrupt_id, response, checkpoint_store)` is called
2. Loads `(state, step)` from checkpoint — `state.data` has intermediate values from first pass
3. Sets `state._interrupt_responses[interrupt_key] = response`
4. Re-enters the execution loop at `step` with the loaded state
5. When the generator node re-executes, `_aexecute_generator_node` finds the matching `_interrupt_responses` entry and calls `gen.asend(response)` to inject the human's answer directly into the `yield` expression
6. Generator continues from the `yield` with the human's response as the return value

**Why this works without side-effect caching**: Because `state.data` is the natural side-effect store. The convention `if "key" not in state.data: state.data["key"] = await compute()` means expensive work is stored in state, and when state is loaded from checkpoint, the work is not re-done. We document this pattern prominently.

**Example node:**

```python
async def review_node(state: GraphState) -> AsyncGenerator:
    # Expensive work stored in state before yielding — skipped on resume
    if "analysis" not in state.data:
        state.data["analysis"] = await run_deep_analysis(state.data["draft"])

    # Pause for human review
    approval = yield InterruptRequest(
        prompt="Please review the analysis and approve or reject.",
        payload={"draft": state.data["draft"], "analysis": state.data["analysis"]},
    )

    state.data["approved"] = (approval == "yes")
    state.data[STATE_KEY_LAST_OUTPUT] = f"Review complete: {approval}"
    return state
```

### Tests: `tests/test_orchestration_interrupt.py` (~40 tests, part of graph test file or separate)
- Generator node yields `InterruptRequest` → `GraphResult.interrupted = True`
- `resume()` injects response into correct yield point
- Generator continues after `yield` with human response as expression value
- Intermediate state persisted through interrupt/resume cycle
- Multiple interrupts in sequence: second `yield` pauses a second time
- Non-generator node does not trigger interrupt machinery
- `_interrupt_responses` excluded from `to_dict()` / `from_dict()`
- Missing checkpoint raises `ValueError` on `resume()`

---

## Phase 4: Checkpointing

### `src/selectools/orchestration/checkpoint.py` (~260 lines)

Follows the `sessions.py` 3-backend pattern exactly.

**`CheckpointStore` Protocol:**

```python
class CheckpointStore(Protocol):
    def save(self, graph_id: str, state: GraphState, step: int) -> str: ...   # returns checkpoint_id
    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]: ...         # (state, step)
    def list(self, graph_id: str) -> List[CheckpointMetadata]: ...
    def delete(self, checkpoint_id: str) -> bool: ...
```

**`CheckpointMetadata`** dataclass:

```python
@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    graph_id: str
    step: int
    node_name: str
    interrupted: bool
    created_at: datetime
```

**3 Backends:**

| Backend | Storage | Thread safety |
|---------|---------|---------------|
| `InMemoryCheckpointStore` | `Dict[str, ...]` | `threading.Lock` |
| `FileCheckpointStore(directory)` | `{dir}/{graph_id}/{id}.json` | `threading.Lock` + `os.makedirs` |
| `SQLiteCheckpointStore(db_path)` | `checkpoints` table, WAL mode | SQLite-level |

Serialization uses `GraphState.to_dict()`. `_interrupt_responses` is serialized separately in a `__interrupt__` key within the checkpoint — it must survive the checkpoint/resume cycle.

### Tests: `tests/test_orchestration_checkpoint.py` (~55 tests)
- All 3 backends: save/load round-trip, list, delete
- `interrupted=True` flag persisted in `CheckpointMetadata`
- `_interrupt_responses` survives serialization round-trip
- Resume execution from mid-graph checkpoint (re-enter loop at correct step)
- Thread safety: InMemory and File concurrent writes
- SQLite WAL mode concurrent access
- `checkpoint_id` collision handling (UUID4 guarantee)

---

## Phase 5: Integration

### Modify: `src/selectools/observer.py`

Add 10 new methods to `AgentObserver` (total: **42 sync events**, up from 32):

```python
# Graph-level lifecycle
def on_graph_start(self, run_id: str, graph_name: str, entry_node: str, state: Dict[str, Any]) -> None: ...
def on_graph_end(self, run_id: str, graph_name: str, steps: int, total_duration_ms: float) -> None: ...
def on_graph_error(self, run_id: str, graph_name: str, node_name: str, error: Exception) -> None: ...
# Node-level lifecycle
def on_node_start(self, run_id: str, node_name: str, step: int) -> None: ...
def on_node_end(self, run_id: str, node_name: str, step: int, duration_ms: float) -> None: ...
# Routing
def on_graph_routing(self, run_id: str, from_node: str, to_node: str) -> None: ...
# HITL
def on_graph_interrupt(self, run_id: str, node_name: str, interrupt_id: str) -> None: ...
def on_graph_resume(self, run_id: str, node_name: str, interrupt_id: str) -> None: ...
# Parallel execution
def on_parallel_start(self, run_id: str, group_name: str, child_nodes: List[str]) -> None: ...
def on_parallel_end(self, run_id: str, group_name: str, child_count: int) -> None: ...
```

Add 10 matching async methods to `AsyncAgentObserver` (total: **39 async events**, up from 29).

Update `LoggingObserver` (emit JSON for all 10 new events) and `SimpleStepObserver` (delegate to `self._cb()`) with implementations.

### Modify: `src/selectools/trace.py`

Add 8 new `StepType` enum members (total: **25**, up from 17):

```python
class StepType(str, Enum):
    # ... existing 17 ...
    GRAPH_NODE_START   = "graph_node_start"
    GRAPH_NODE_END     = "graph_node_end"
    GRAPH_ROUTING      = "graph_routing"
    GRAPH_CHECKPOINT   = "graph_checkpoint"
    GRAPH_INTERRUPT    = "graph_interrupt"
    GRAPH_RESUME       = "graph_resume"
    GRAPH_PARALLEL_START = "graph_parallel_start"
    GRAPH_PARALLEL_END   = "graph_parallel_end"
```

The `AgentGraph` creates a root `AgentTrace` for the entire execution. Each node's `Agent` produces its own child `AgentTrace` linked via `parent_run_id`. Root trace captures graph-level steps; child traces capture agent-internal steps.

### Modify: `src/selectools/__init__.py`

```python
from .orchestration import (
    AgentGraph,
    GraphState,
    GraphNode,
    ParallelGroupNode,
    SubgraphNode,
    GraphResult,
    GraphEvent,
    GraphEventType,
    InterruptRequest,
    Scatter,
    MergePolicy,
    ErrorPolicy,
    CheckpointStore,
    InMemoryCheckpointStore,
    FileCheckpointStore,
    SQLiteCheckpointStore,
    SupervisorAgent,
    STATE_KEY_LAST_OUTPUT,
)
```

### Modify: `tests/test_phase1_design_patterns.py`

Update StepType count assertion: 17 → 25.

### Tests: `tests/test_orchestration_integration.py` (~45 tests)
- Observer events fire in correct order during graph execution
- `LoggingObserver` emits JSON for all 10 new events
- `SimpleStepObserver` routes all 10 new events to callback
- `AsyncAgentObserver` events fire in `arun()` and `astream()` paths
- All 8 new StepTypes appear in traces at correct points
- Root trace has graph steps; child traces have agent steps; `parent_run_id` links them
- Public exports importable from both `selectools` and `selectools.orchestration`
- `tests/test_phase1_design_patterns.py` StepType count: 25

---

## Phase 6: Supervisor

### `src/selectools/orchestration/supervisor.py` (~300 lines)

```python
class SupervisorStrategy(str, Enum):
    PLAN_AND_EXECUTE = "plan_and_execute"
    ROUND_ROBIN = "round_robin"
    DYNAMIC = "dynamic"


class SupervisorAgent:
    def __init__(
        self,
        agents: Dict[str, Agent],
        provider: Provider,
        strategy: SupervisorStrategy = SupervisorStrategy.PLAN_AND_EXECUTE,
        max_rounds: int = 10,
        cancellation_token: Optional[CancellationToken] = None,
        max_total_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        observers: Optional[List[AgentObserver]] = None,
    ): ...

    def run(self, prompt: str) -> GraphResult: ...
    async def arun(self, prompt: str) -> GraphResult: ...
    async def astream(self, prompt: str) -> AsyncGenerator[GraphEvent, None]: ...
```

All strategies build and execute an `AgentGraph` internally — the supervisor is a convenience wrapper:

| Strategy | How it works |
|----------|-------------|
| `plan_and_execute` | Supervisor LLM generates JSON plan `[{"agent": name, "task": str}]` → linear chain of agent nodes |
| `round_robin` | Each agent participates in each round; supervisor checks after each full round whether to continue (up to `max_rounds`) |
| `dynamic` | Router LLM node selects best agent per step based on current state + task description |

**Usage example:**

```python
from selectools.orchestration import SupervisorAgent

supervisor = SupervisorAgent(
    agents={
        "researcher": researcher_agent,
        "writer": writer_agent,
        "reviewer": reviewer_agent,
    },
    provider=OpenAIProvider(),
    strategy="plan_and_execute",
    max_rounds=5,
)
result = supervisor.run("Write a comprehensive blog post about LLM safety")
print(result.content)
print(result.total_usage)
```

### Tests: `tests/test_orchestration_supervisor.py` (~55 tests)
- `plan_and_execute`: mock LLM plan generation, sequential execution, result aggregation
- `round_robin`: full cycle, early-exit when done, `max_rounds` limit enforced
- `dynamic`: LLM router node selects agents, state reflects routing history
- Error propagation from underlying graph
- Usage aggregation across supervisor rounds
- Trace hierarchy: supervisor → graph → node agents
- `astream()` yields `GraphEvent` sequence
- Cancellation propagation to graph and all node agents
- Budget enforcement at supervisor level (halts when cumulative exceeds limit)
- All 3 strategies produce valid `GraphResult`

---

## Phase 7: Visualization

### Add methods to `AgentGraph`:

```python
def to_mermaid(self) -> str:
    """Returns a Mermaid flowchart diagram string (no external deps)."""
    # Outputs e.g.:
    # graph TD
    #   planner --> researcher
    #   researcher -->|conditional| writer
    #   researcher -->|conditional| researcher
    #   writer --> __end__

def visualize(self, format: str = "ascii") -> None:
    """Prints an ASCII graph to stdout (format='ascii', no deps).
    format='png' uses graphviz (optional dep, raises ImportError if missing).
    """
```

**No new dependencies.** ASCII visualization uses box-drawing characters. PNG requires `graphviz` (already common in ML environments, never auto-installed).

**Example output of `to_mermaid()`:**

```
graph TD
    planner["planner (Agent)"] --> researcher
    researcher["researcher (Agent)"] -->|"ready=True"| writer
    researcher -->|"ready=False"| researcher
    fan_out["fan_out [parallel: a, b, c]"] --> merge_node
    writer["writer (Agent)"] --> __end__["END"]
```

### Tests: `tests/test_orchestration_viz.py` (~15 tests)
- `to_mermaid()`: linear, conditional, cyclic, parallel groups all rendered
- `to_mermaid()` for an empty graph (no edges): no crash
- `visualize("ascii")`: executes without error
- `visualize("png")`: raises `ImportError` when graphviz not installed (mocked)

---

## Phase 8: Docs & Release

### New Docs
- `docs/modules/ORCHESTRATION.md` — full module reference (AgentGraph, GraphState, CheckpointStore, SupervisorAgent, HITL, parallel execution, subgraphs, visualization)
- `docs/modules/SUPERVISOR.md` — supervisor patterns with all 3 strategies

### Updated Docs
- `docs/ARCHITECTURE.md` — add orchestration tier diagram showing graph → node agents → providers
- `docs/QUICKSTART.md` — multi-agent quickstart section (10-line example)
- `docs/index.md` — feature table: add orchestration rows, update test/example counts
- `notebooks/getting_started.ipynb` — Steps 25-31 for all v0.18.0 features
- `README.md` — "What's New v0.18.0" section
- `ROADMAP.md` — v0.18.0 🟡 → ✅
- `CLAUDE.md` — StepType count 17→25, observer counts 32/29→42/39, examples 54→61

### Examples (55–61)

| File | Demonstrates |
|------|-------------|
| `examples/55_agent_graph_linear.py` | Basic 3-node linear pipeline with `LocalProvider` |
| `examples/56_agent_graph_parallel.py` | Parallel fan-out with `MergePolicy.APPEND` |
| `examples/57_agent_graph_conditional.py` | Conditional routing with `path_map` validation |
| `examples/58_agent_graph_hitl.py` | Generator node with `yield InterruptRequest` + `resume()` |
| `examples/59_agent_graph_checkpointing.py` | `FileCheckpointStore` save/resume mid-graph |
| `examples/60_supervisor_agent.py` | All 3 supervisor strategies |
| `examples/61_agent_graph_subgraph.py` | Nested subgraph with `input_map`/`output_map` |

### Release Artifacts
- Version: `0.17.7` → `0.18.0` in `__init__.py` + `pyproject.toml`
- `CHANGELOG.md` entry
- Git tag: `v0.18.0`
- PyPI: `python3 -m build && python3 -m twine upload dist/*`

---

## How This Beats LangGraph

| LangGraph | Selectools AgentGraph | Evidence |
|-----------|----------------------|----------|
| `interrupt()` restarts node from beginning — tools with side effects re-execute | Generator nodes with `yield InterruptRequest` — state cached before yield, resume injects response at exact yield point | LangGraph docs: *"the node function is re-executed from the beginning when the graph resumes"* |
| `StateGraph(MyState)` + `Annotated[list, add_messages]` + `compile()` + `MemorySaver` | `AgentGraph()` + `add_node()` + `run()` | Zero new concepts to learn |
| `@task` cache breaks silently in production (new container = empty cache) | `FileCheckpointStore` / `SQLiteCheckpointStore` — explicit, durable, self-hosted | Community issue: *"3 days debugging @task caching across Lambda invocations"* |
| LangGraph Studio, cron, auth = paid Platform | `to_mermaid()`, `FileCheckpointStore`, built-in observer — zero cloud dependency | LangGraph pricing page |
| No built-in budget enforcement | `max_total_tokens` / `max_cost_usd` at graph + node levels | Selectools v0.17.3+ |
| No cooperative cancellation | `CancellationToken` propagates to all nodes — cancel from any thread | Selectools v0.17.3+ |
| `CompiledGraph` nesting with `invoke()` | `AgentGraph.__call__` — any graph is a node via duck typing | No framework concept required |
| Sub-graph state mapping requires `InputTransformer`/`OutputTransformer` | `add_subgraph(input_map={"a": "b"})` — 2 dicts | Explicit over magic |
| Parallel `Send()` requires returning list of `Command(goto=node, update={...})` | `return [Scatter("worker", {"task": t}) for t in tasks]` — plain Python list | Readability |
| No built-in observability | 10 new observer events, 8 new StepTypes — every graph step is traced | Selectools observer pattern |

---

## Files Summary

### New (~8 source files)

| File | Est. lines |
|------|------------|
| `src/selectools/orchestration/__init__.py` | ~40 |
| `src/selectools/orchestration/state.py` | ~180 |
| `src/selectools/orchestration/node.py` | ~200 |
| `src/selectools/orchestration/graph.py` | ~650 |
| `src/selectools/orchestration/checkpoint.py` | ~260 |
| `src/selectools/orchestration/supervisor.py` | ~300 |
| **Total source** | **~1,630 lines** |

### New (~6 test files + 7 examples)

| File | Tests |
|------|-------|
| `tests/test_orchestration_primitives.py` | ~55 |
| `tests/test_orchestration_graph.py` | ~130 |
| `tests/test_orchestration_checkpoint.py` | ~55 |
| `tests/test_orchestration_integration.py` | ~45 |
| `tests/test_orchestration_supervisor.py` | ~55 |
| `tests/test_orchestration_viz.py` | ~15 |
| **Total new tests** | **~355** |
| **Total tests after** | **~2,630** |
| **New examples** | **7 (55–61)** |

### Modified (5 files)

| File | Changes |
|------|---------|
| `src/selectools/observer.py` | +10 sync + 10 async events; `LoggingObserver` + `SimpleStepObserver` |
| `src/selectools/trace.py` | +8 `StepType` members (17 → 25) |
| `src/selectools/exceptions.py` | Flesh out `GraphExecutionError` fields |
| `src/selectools/__init__.py` | New exports + version bump |
| `pyproject.toml` | Version bump |

---

## Verification

After each phase:
1. `black src/ tests/ --line-length=100 && isort src/ tests/ --profile=black --line-length=100`
2. `flake8 src/ && mypy src/`
3. `pytest tests/ -x -q` — ALL must pass

After all phases:
4. `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build` — verify docs build, no broken links
5. Manual smoke test: 3-node linear graph with `LocalProvider`, verify `GraphResult` fields, aggregated usage, correct event sequence

---

## Resolved Design Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| `default_input_transform` canonical key | `STATE_KEY_LAST_OUTPUT = "__last_output__"` in `state.data` | Eliminates ambiguity for all inter-node handoffs |
| Parallel groups: registered as nodes? | Yes — `add_parallel_nodes` writes to `self._nodes[name]` | Fixes `add_edge` validation; natural API |
| Async callable dispatch | `asyncio.iscoroutinefunction` → `await fn(state)`; `isasyncgenfunction` → HITL machinery; else → `run_in_executor` | Handles all 4 cases cleanly |
| Async-in-sync threading hazard | Sync parallel uses `ThreadPoolExecutor` with `agent.run()` (sync); never `asyncio.run()` inside existing loop | Matches existing parallel tool execution pattern |
| TypedDict reducers | Implement `MergePolicy` enum + `merge_states()` helper; TypedDict with `Annotated` is a power-user opt-in | Covers 90% of use cases simply; keeps 100% coverage |
| HITL mechanism | Generator nodes with `yield InterruptRequest`; `state.data` is the side-effect cache | Best Python idiom for suspend/resume; natural protection from re-execution |
| Graph-level guardrails | Yes — `input_guardrails` before first node, `output_guardrails` after last node | Already resolved in original plan |
| `path_map` for conditional edges | Optional kwarg on `add_conditional_edge` | Opt-in; enables compile-time validation without forcing it |
| Per-node visit counter | `GraphNode.max_visits = 0` (0 = unlimited); `_visit_counts` tracked per run | Prevents runaway cycles without global-only `max_steps` |
| `SubgraphNode` vs `__call__` | Both: `add_subgraph` for explicit key mapping; `__call__` for duck-typing composition | Different use cases; explicit mapping is safer |
| Streaming modes | Ship 3: full-state `values`, per-node-delta `updates`, raw-text `messages` (via `GraphEventType`) | Matches LangGraph's most-used 3 of 5; defer rest |
| `SupervisorStrategy` type | `str, Enum` (consistent with `StepType`, `ErrorPolicy`, `ModelType`) | ADR-003 pattern |
