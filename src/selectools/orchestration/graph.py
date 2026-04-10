"""
AgentGraph — multi-agent orchestration engine.

Executes a directed graph of Agent nodes (or callables) with support for:
- Linear and conditional routing (plain Python functions)
- Parallel fan-out via ParallelGroupNode / Scatter
- Human-in-the-loop via generator nodes yielding InterruptRequest
- Checkpointing and resume (pass a CheckpointStore)
- Loop and stall detection with observer events
- Graph-level budget and cancellation propagation
- Subgraph composition (SubgraphNode)
- Mermaid and ASCII visualization
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from .._async_utils import run_sync
from ..exceptions import GraphExecutionError
from ..stability import beta
from ..trace import AgentTrace, StepType, TraceStep
from ..types import AgentResult, Message, Role
from ..usage import UsageStats
from .node import (
    GraphNode,
    ParallelGroupNode,
    SubgraphNode,
    build_context_messages,
    default_input_transform,
    default_output_transform,
)
from .state import (
    STATE_KEY_LAST_OUTPUT,
    GraphEvent,
    GraphEventType,
    GraphState,
    InterruptRequest,
    MergePolicy,
    Scatter,
    _Goto,
    _Update,
    merge_states,
)

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..cancellation import CancellationToken
    from ..guardrails.pipeline import GuardrailsPipeline
    from ..observer import AgentObserver
    from .checkpoint import CheckpointStore


_STATE_KEY_PENDING_INTERRUPT: str = "__pending_interrupt_key__"


class ErrorPolicy(str, Enum):
    """How the graph handles node execution errors.

    ABORT: Raise immediately (default).
    SKIP:  Log error in state.errors and continue to next node.
    RETRY: Retry up to error_retry_limit times, then abort.
    """

    ABORT = "abort"
    SKIP = "skip"
    RETRY = "retry"


@dataclass
class GraphResult:
    """Result of a complete graph execution.

    Attributes:
        content: Last node's result content (state.data[STATE_KEY_LAST_OUTPUT]).
        state: Final GraphState after all nodes have executed.
        node_results: Per-node AgentResult lists keyed by node name.
        trace: Composite graph-level AgentTrace (includes child traces via parent_run_id).
        total_usage: Aggregated UsageStats across all nodes.
        interrupted: True if execution paused for HITL (call graph.resume() to continue).
        interrupt_id: Checkpoint ID to pass to graph.resume() when interrupted.
        steps: Total graph-level iterations executed.
        stalls: Number of stall events detected during this run.
        loops_detected: Number of hard loop events detected.
    """

    content: str
    state: GraphState
    node_results: Dict[str, List[AgentResult]]
    trace: AgentTrace
    total_usage: UsageStats
    interrupted: bool = False
    interrupt_id: Optional[str] = None
    steps: int = 0
    stalls: int = 0
    loops_detected: int = 0


def _make_synthetic_result(state: GraphState) -> AgentResult:
    """Create a minimal AgentResult from current state for non-Agent nodes."""
    content = state.data.get(STATE_KEY_LAST_OUTPUT, "")
    if not isinstance(content, str):
        content = str(content)
    return AgentResult(
        message=Message(role=Role.ASSISTANT, content=content),
        iterations=1,
        usage=UsageStats(),
    )


def _state_hash(state: GraphState) -> str:
    """Compute a stable hash of the mutable parts of GraphState for loop detection."""
    try:
        payload = json.dumps(
            {"data": state.data, "current_node": state.current_node},
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()  # nosec B324
    except Exception:
        return str(id(state))


def _to_usage_stats(obj: Any) -> UsageStats:
    """Normalise UsageStats or AgentUsage to a UsageStats value."""
    from ..usage import AgentUsage

    if isinstance(obj, AgentUsage):
        return UsageStats(
            prompt_tokens=obj.total_prompt_tokens,
            completion_tokens=obj.total_completion_tokens,
            total_tokens=obj.total_tokens,
            cost_usd=obj.total_cost_usd,
        )
    return obj  # type: ignore[no-any-return]


def _merge_usage(base: UsageStats, added: Any) -> UsageStats:
    """Add two UsageStats (or AgentUsage) together."""
    added_stats = _to_usage_stats(added)
    return UsageStats(
        prompt_tokens=base.prompt_tokens + added_stats.prompt_tokens,
        completion_tokens=base.completion_tokens + added_stats.completion_tokens,
        total_tokens=base.total_tokens + added_stats.total_tokens,
        cost_usd=base.cost_usd + added_stats.cost_usd,
    )


@beta
class AgentGraph:
    """Directed graph of agent nodes with routing, parallelism, and HITL support.

    Example::

        graph = AgentGraph()
        graph.add_node("planner", planner_agent)
        graph.add_node("writer", writer_agent)
        graph.add_edge("planner", "writer")
        graph.add_edge("writer", AgentGraph.END)
        graph.set_entry("planner")
        result = graph.run("Write a blog post about AI agents")

    For conditional routing::

        graph.add_conditional_edge(
            "writer",
            lambda state: "reviewer" if state.data.get("needs_review") else AgentGraph.END,
            path_map={"reviewer": "reviewer"},
        )

    For human-in-the-loop::

        async def review_node(state):
            if "analysis" not in state.data:
                state.data["analysis"] = await do_analysis(state.data["draft"])
            approval = yield InterruptRequest(prompt="Approve?", payload=state.data["analysis"])
            state.data["approved"] = (approval == "yes")
            state.data[STATE_KEY_LAST_OUTPUT] = "Review done"

        store = FileCheckpointStore("./checkpoints")
        result = graph.run("...", checkpoint_store=store)
        if result.interrupted:
            final = graph.resume(result.interrupt_id, "yes", checkpoint_store=store)
    """

    END: ClassVar[str] = "__end__"

    def __init__(
        self,
        name: str = "graph",
        observers: Optional[List["AgentObserver"]] = None,
        error_policy: ErrorPolicy = ErrorPolicy.ABORT,
        error_retry_limit: int = 3,
        max_steps: int = 50,
        cancellation_token: Optional["CancellationToken"] = None,
        max_total_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        input_guardrails: Optional["GuardrailsPipeline"] = None,
        output_guardrails: Optional["GuardrailsPipeline"] = None,
        enable_loop_detection: bool = True,
        stall_threshold: int = 3,
        fast_route_fn: Optional[Callable[[GraphState], Optional[str]]] = None,
    ) -> None:
        self.name = name
        self._observers: List["AgentObserver"] = observers or []
        self.error_policy = error_policy
        self.error_retry_limit = error_retry_limit
        self.max_steps = max_steps
        self._cancellation_token = cancellation_token
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
        self.enable_loop_detection = enable_loop_detection
        self.stall_threshold = stall_threshold
        self.fast_route_fn = fast_route_fn

        self._nodes: Dict[str, Union[GraphNode, ParallelGroupNode, SubgraphNode]] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, Callable] = {}
        self._path_maps: Dict[str, Dict[str, str]] = {}
        self._scatter_patches: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._entry_node: Optional[str] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def chain(
        cls,
        *agents: Any,
        names: Optional[List[str]] = None,
        **graph_kwargs: Any,
    ) -> "AgentGraph":
        """Create a linear pipeline of agents.

        Usage::

            graph = AgentGraph.chain(planner, writer, reviewer)
            result = graph.run("Write a blog post")

        Args:
            *agents: Agent instances or callables to execute in sequence.
            names: Optional node names (auto-generated as step_0, step_1, ... if omitted).
            **graph_kwargs: Passed to AgentGraph constructor (name, observers, etc.).
        """
        if not agents:
            raise ValueError("AgentGraph.chain() requires at least one agent")
        graph = cls(**graph_kwargs)
        node_names = names or [f"step_{i}" for i in range(len(agents))]
        if len(node_names) != len(agents):
            raise ValueError(f"names length ({len(node_names)}) != agents length ({len(agents)})")
        for i, (nname, agent) in enumerate(zip(node_names, agents)):
            graph.add_node(nname, agent)
            if i > 0:
                graph.add_edge(node_names[i - 1], nname)
        graph.add_edge(node_names[-1], cls.END)
        graph.set_entry(node_names[0])
        return graph

    # ------------------------------------------------------------------
    # Node and edge management
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        agent_or_callable: Any,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        max_iterations: int = 1,
        max_visits: int = 0,
        error_policy: Optional[ErrorPolicy] = None,
        context_mode: Optional[Any] = None,
        context_n: int = 6,
        next_node: Optional[str] = None,
    ) -> None:
        """Register a node in the graph.

        Args:
            name: Unique node name.
            agent_or_callable: An Agent, async callable, sync callable, or async generator function.
            input_transform: Optional custom message transformer (overrides context_mode).
            output_transform: Optional custom state updater.
            max_iterations: Re-execution limit per visit.
            max_visits: Maximum times this node may be visited (0 = unlimited).
            error_policy: Per-node error policy (None = inherit from graph).
            context_mode: Controls history forwarded to the agent (default LAST_MESSAGE).
            context_n: Message count for LAST_N mode.
            next_node: If provided, automatically add a static edge to this node.
        """
        from .state import ContextMode

        node = GraphNode(
            name=name,
            agent=agent_or_callable,
            input_transform=input_transform,
            output_transform=output_transform,
            context_mode=context_mode or ContextMode.LAST_MESSAGE,
            context_n=context_n,
            max_iterations=max_iterations,
            max_visits=max_visits,
            error_policy=error_policy,
        )
        self._nodes[name] = node
        # Auto-set entry node on first add_node call
        if self._entry_node is None:
            self._entry_node = name
        if next_node is not None:
            self.add_edge(name, next_node)

    def add_parallel_nodes(
        self,
        name: str,
        node_names: List[str],
        merge_policy: MergePolicy = MergePolicy.LAST_WINS,
        merge_fn: Optional[Callable[[List[GraphState]], GraphState]] = None,
    ) -> None:
        """Register a parallel group node.

        The group will fan out to all node_names in parallel, then merge results.

        Args:
            name: Unique name for this parallel group node.
            node_names: Child node names to execute in parallel.
            merge_policy: How to merge branch states.
            merge_fn: Custom merge function (overrides merge_policy).
        """
        node = ParallelGroupNode(
            name=name,
            child_node_names=node_names,
            merge_policy=merge_policy,
            merge_fn=merge_fn,
        )
        self._nodes[name] = node

    def add_subgraph(
        self,
        name: str,
        graph: "AgentGraph",
        input_map: Optional[Dict[str, str]] = None,
        output_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register a nested subgraph as a node.

        Args:
            name: Unique node name for this subgraph.
            graph: The nested AgentGraph to execute.
            input_map: parent state.data key → subgraph state.data key.
            output_map: subgraph state.data key → parent state.data key.
        """
        node = SubgraphNode(
            name=name,
            graph=graph,
            input_map=input_map or {},
            output_map=output_map or {},
        )
        self._nodes[name] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a static edge from from_node to to_node."""
        self._edges[from_node] = to_node

    def add_conditional_edge(
        self,
        from_node: str,
        router_fn: Callable[[GraphState], Union[str, Scatter, List[Scatter]]],
        path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a conditional edge with a routing function.

        Args:
            from_node: The node whose output triggers routing.
            router_fn: Function receiving GraphState, returning node name or Scatter(s).
            path_map: Optional dict mapping router return values to node names.
                      Enables compile-time validation of routing destinations.
        """
        self._conditional_edges[from_node] = router_fn
        if path_map:
            self._path_maps[from_node] = path_map

    def set_entry(self, node_name: str) -> None:
        """Set the entry (start) node for this graph."""
        self._entry_node = node_name

    def validate(self) -> List[str]:
        """Validate graph structure and return list of warnings.

        Called automatically in run(). Does not raise — returns warnings only.
        """
        warnings: List[str] = []
        if self._entry_node is None:
            warnings.append("No entry node set. Call set_entry() before run().")
        elif self._entry_node not in self._nodes:
            warnings.append(f"Entry node {self._entry_node!r} not in nodes.")

        # Validate edge targets
        for from_node, to_node in self._edges.items():
            if to_node != self.END and to_node not in self._nodes:
                warnings.append(f"Edge {from_node!r} → {to_node!r}: target not found.")

        # Validate path_map targets
        for from_node, pmap in self._path_maps.items():
            for _val, target in pmap.items():
                if target != self.END and target not in self._nodes:
                    warnings.append(f"path_map for {from_node!r}: target {target!r} not found.")

        # Validate parallel group children
        for node in self._nodes.values():
            if isinstance(node, ParallelGroupNode):
                for child in node.child_node_names:
                    if child not in self._nodes:
                        warnings.append(
                            f"ParallelGroupNode {node.name!r}: child {child!r} not found."
                        )

        return warnings

    # ------------------------------------------------------------------
    # Observer notification helpers
    # ------------------------------------------------------------------

    def _notify(self, event: str, run_id: str, *args: Any, **kwargs: Any) -> None:
        import logging

        for obs in self._observers:
            handler = getattr(obs, event, None)
            if handler is not None:
                try:
                    handler(run_id, *args, **kwargs)
                except Exception as exc:
                    logging.getLogger("selectools.orchestration").warning(
                        "Observer %s.%s raised: %s", type(obs).__name__, event, exc
                    )

    def _trace_step(self, trace: AgentTrace, step_type: StepType, **kwargs: Any) -> None:
        trace.add(TraceStep(type=step_type, **kwargs))

    # ------------------------------------------------------------------
    # Execution — synchronous entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
    ) -> GraphResult:
        """Execute the graph synchronously.

        Args:
            prompt_or_state: Initial user prompt string or pre-built GraphState.
            checkpoint_store: Optional store for checkpointing (required for HITL).
            checkpoint_id: Load from a previous checkpoint (for resume).

        Returns:
            GraphResult with final state, trace, and usage.
        """
        return run_sync(
            self.arun(
                prompt_or_state,
                checkpoint_store=checkpoint_store,
                checkpoint_id=checkpoint_id,
            )
        )

    # ------------------------------------------------------------------
    # Execution — async entry point
    # ------------------------------------------------------------------

    async def arun(
        self,
        prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
        _interrupt_response: Any = None,
    ) -> GraphResult:
        """Execute the graph asynchronously.

        Args:
            prompt_or_state: Initial user prompt string or pre-built GraphState.
            checkpoint_store: Optional store for checkpointing.
            checkpoint_id: Checkpoint to resume from.
            _interrupt_response: Internal use — injected response for HITL resume.

        Returns:
            GraphResult with final state, trace, and usage.
        """
        # Validate
        self.validate()
        if self._entry_node is None:
            raise GraphExecutionError(self.name, "", ValueError("No entry node set"), 0)

        # Normalize input
        if prompt_or_state is None:
            state = GraphState.from_prompt("")
        elif isinstance(prompt_or_state, str):
            state = GraphState.from_prompt(prompt_or_state)
        else:
            state = prompt_or_state

        # Load from checkpoint if requested
        start_step = 0
        if checkpoint_id and checkpoint_store:
            loaded_state, start_step = checkpoint_store.load(checkpoint_id)
            state = loaded_state
            if _interrupt_response is not None:
                pending_key = state.metadata.get(_STATE_KEY_PENDING_INTERRUPT, "")
                if pending_key:
                    state._interrupt_responses[pending_key] = _interrupt_response
                    del state.metadata[_STATE_KEY_PENDING_INTERRUPT]

        trace = AgentTrace(metadata={"graph_name": self.name})
        run_id = trace.run_id
        usage = UsageStats()
        node_results: Dict[str, List[AgentResult]] = {}
        visit_counts: Dict[str, int] = {}
        stall_count = 0
        loop_count = 0
        prev_hash: Optional[str] = None
        unchanged_streak = 0
        seen_hashes: set = set()

        if checkpoint_id and checkpoint_store:
            self._notify("on_graph_resume", run_id, state.current_node, checkpoint_id)
            self._trace_step(
                trace,
                StepType.GRAPH_RESUME,
                node_name=state.current_node,
                checkpoint_id=checkpoint_id,
            )

        # Fast-path routing
        if self.fast_route_fn is not None and not checkpoint_id:
            fast_node = self.fast_route_fn(state)
            if fast_node is not None and fast_node in self._nodes:
                self._notify("on_graph_start", run_id, self.name, fast_node, state.to_dict())
                fast_node_obj = self._nodes[fast_node]
                if isinstance(fast_node_obj, GraphNode):
                    result, state, interrupted = await self._aexecute_node(
                        fast_node_obj, state, trace, run_id
                    )
                    node_results.setdefault(fast_node, []).append(result)
                    if result.usage:
                        usage = _merge_usage(usage, result.usage)
                self._notify("on_graph_end", run_id, self.name, 1, 0.0)
                return GraphResult(
                    content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
                    state=state,
                    node_results=node_results,
                    trace=trace,
                    total_usage=usage,
                    steps=1,
                )

        # Input guardrails
        if self.input_guardrails:
            messages_str = " ".join(getattr(m, "content", str(m)) for m in state.messages)
            gr = await self.input_guardrails.acheck_input(messages_str)
            if gr and not gr.passed:
                state.errors.append({"type": "input_guardrail", "message": gr.reason or ""})

        self._notify("on_graph_start", run_id, self.name, self._entry_node, state.to_dict())
        self._trace_step(trace, StepType.GRAPH_NODE_START, node_name=self.name, step_number=0)

        current = self._entry_node
        step = start_step
        graph_start_time = time.time()

        while current != self.END and step < self.max_steps:
            step += 1
            state.current_node = current

            # Cancellation check
            if self._cancellation_token is not None and self._cancellation_token.is_cancelled:
                break

            # Budget check
            if self._over_budget(usage):
                break

            # Per-node visit count
            visit_counts[current] = visit_counts.get(current, 0) + 1
            _node_maybe = self._nodes.get(current)
            if _node_maybe is None:
                raise GraphExecutionError(
                    self.name, current, KeyError(f"Node {current!r} not found"), step
                )
            node: Union[GraphNode, ParallelGroupNode, SubgraphNode] = _node_maybe

            if isinstance(node, GraphNode) and node.max_visits > 0:
                if visit_counts[current] > node.max_visits:
                    raise GraphExecutionError(
                        self.name,
                        current,
                        RuntimeError(f"Node {current!r} exceeded max_visits={node.max_visits}"),
                        step,
                    )

            # Loop/stall detection
            if self.enable_loop_detection:
                current_hash = _state_hash(state)
                if current_hash in seen_hashes:
                    loop_count += 1
                    self._notify("on_loop_detected", run_id, current, loop_count)
                    self._trace_step(
                        trace,
                        StepType.GRAPH_LOOP_DETECTED,
                        node_name=current,
                        step_number=step,
                    )
                    raise GraphExecutionError(
                        self.name,
                        current,
                        RuntimeError(f"Hard loop detected at step {step} (identical state hash)"),
                        step,
                    )
                seen_hashes.add(current_hash)

                if current_hash == prev_hash:
                    unchanged_streak += 1
                    if unchanged_streak >= self.stall_threshold:
                        stall_count += 1
                        self._notify("on_stall_detected", run_id, current, stall_count)
                        self._trace_step(
                            trace,
                            StepType.GRAPH_STALL,
                            node_name=current,
                            step_number=step,
                        )
                else:
                    unchanged_streak = 0
                prev_hash = current_hash

            self._notify("on_node_start", run_id, current, step)
            self._trace_step(trace, StepType.GRAPH_NODE_START, node_name=current, step_number=step)
            node_start_time = time.time()

            try:
                if isinstance(node, ParallelGroupNode):
                    child_results, state = await self._aexecute_parallel(node, state, trace, run_id)
                    node_results.update(child_results)
                    # Accumulate usage from all parallel children
                    for child_list in child_results.values():
                        for child_result in child_list:
                            if child_result.usage:
                                usage = _merge_usage(usage, child_result.usage)

                elif isinstance(node, SubgraphNode):
                    result, state = await self._aexecute_subgraph(node, state, trace, run_id)
                    node_results.setdefault(current, []).append(result)
                    if result.usage:
                        usage = _merge_usage(usage, result.usage)

                else:
                    # Retry loop for RETRY policy
                    effective_policy = node.error_policy or self.error_policy
                    max_attempts = (
                        self.error_retry_limit if effective_policy == ErrorPolicy.RETRY else 1
                    )
                    for attempt in range(max_attempts):
                        try:
                            result, state, interrupted = await self._aexecute_node(
                                node, state, trace, run_id
                            )
                            if result.usage:
                                usage = _merge_usage(usage, result.usage)
                            node_results.setdefault(current, []).append(result)

                            if interrupted:
                                interrupt_key = state.metadata.get(_STATE_KEY_PENDING_INTERRUPT, "")
                                if checkpoint_store:
                                    ckpt_id = checkpoint_store.save(run_id, state, step)
                                else:
                                    ckpt_id = f"{run_id}_{step}"

                                self._notify("on_graph_interrupt", run_id, current, ckpt_id)
                                self._trace_step(
                                    trace,
                                    StepType.GRAPH_INTERRUPT,
                                    node_name=current,
                                    interrupt_key=interrupt_key,
                                    checkpoint_id=ckpt_id,
                                )
                                duration_ms = (time.time() - graph_start_time) * 1000
                                self._notify("on_graph_end", run_id, self.name, step, duration_ms)
                                return GraphResult(
                                    content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
                                    state=state,
                                    node_results=node_results,
                                    trace=trace,
                                    total_usage=usage,
                                    interrupted=True,
                                    interrupt_id=ckpt_id,
                                    steps=step,
                                    stalls=stall_count,
                                    loops_detected=loop_count,
                                )
                            break  # success

                        except Exception:
                            if attempt < max_attempts - 1:
                                await asyncio.sleep(0)  # yield
                                continue
                            raise

            except GraphExecutionError:
                raise
            except Exception as exc:
                if isinstance(node, GraphNode) and node.error_policy is not None:
                    effective_policy = node.error_policy
                else:
                    effective_policy = self.error_policy

                self._notify("on_graph_error", run_id, self.name, current, exc)
                if effective_policy == ErrorPolicy.ABORT:
                    raise GraphExecutionError(self.name, current, exc, step) from exc
                elif effective_policy == ErrorPolicy.SKIP:
                    state.errors.append(
                        {
                            "node": current,
                            "step": step,
                            "error": str(exc),
                            "type": type(exc).__name__,
                        }
                    )
                    # Fall through to routing (state unchanged)
                # RETRY was handled above

            node_duration_ms = (time.time() - node_start_time) * 1000
            self._notify("on_node_end", run_id, current, step, node_duration_ms)
            self._trace_step(
                trace,
                StepType.GRAPH_NODE_END,
                node_name=current,
                step_number=step,
                duration_ms=node_duration_ms,
            )

            # Checkpoint after each step
            if checkpoint_store:
                ckpt_id = checkpoint_store.save(run_id, state, step)
                self._trace_step(trace, StepType.GRAPH_CHECKPOINT, checkpoint_id=ckpt_id)

            # Resolve next node
            prev_node = current
            current = self._resolve_next_node(current, state, trace, run_id)

        # Output guardrails
        if self.output_guardrails:
            output_text = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            if output_text:
                gr = await self.output_guardrails.acheck_output(output_text)
                if gr and not gr.passed:
                    state.errors.append({"type": "output_guardrail", "message": gr.reason or ""})

        total_duration_ms = (time.time() - graph_start_time) * 1000
        self._trace_step(
            trace,
            StepType.GRAPH_NODE_END,
            node_name=self.name,
            step_number=step,
            duration_ms=total_duration_ms,
        )
        self._notify("on_graph_end", run_id, self.name, step, total_duration_ms)

        return GraphResult(
            content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            state=state,
            node_results=node_results,
            trace=trace,
            total_usage=usage,
            steps=step,
            stalls=stall_count,
            loops_detected=loop_count,
        )

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def astream(
        self,
        prompt_or_state: Union[str, GraphState],
        checkpoint_store: Optional["CheckpointStore"] = None,
        checkpoint_id: Optional[str] = None,
    ) -> AsyncGenerator[GraphEvent, None]:
        """Stream graph execution as GraphEvent objects.

        Yields events for node start/end, routing, interrupts, parallel,
        and the final GRAPH_END with the complete result.
        """
        if prompt_or_state is None:
            state = GraphState.from_prompt("")
        elif isinstance(prompt_or_state, str):
            state = GraphState.from_prompt(prompt_or_state)
        else:
            state = prompt_or_state

        if self._entry_node is None:
            raise GraphExecutionError(self.name, "", ValueError("No entry node set"), 0)

        self.validate()
        trace = AgentTrace(metadata={"graph_name": self.name})
        run_id = trace.run_id
        usage = UsageStats()
        node_results: Dict[str, List[AgentResult]] = {}
        visit_counts: Dict[str, int] = {}
        stall_count = 0
        loop_count = 0
        prev_hash: Optional[str] = None
        unchanged_streak = 0
        seen_hashes: set = set()

        yield GraphEvent(
            type=GraphEventType.GRAPH_START,
            node_name=self._entry_node,
            state=state,
        )
        self._notify("on_graph_start", run_id, self.name, self._entry_node, state.to_dict())
        self._trace_step(trace, StepType.GRAPH_NODE_START, node_name=self.name, step_number=0)

        current = self._entry_node
        step = 0
        graph_start = time.time()

        while current != self.END and step < self.max_steps:
            step += 1
            state.current_node = current

            if self._cancellation_token and self._cancellation_token.is_cancelled:
                break
            if self._over_budget(usage):
                break

            visit_counts[current] = visit_counts.get(current, 0) + 1
            node = self._nodes.get(current)
            if node is None:
                yield GraphEvent(
                    type=GraphEventType.ERROR,
                    node_name=current,
                    error=KeyError(f"Node {current!r} not found"),
                )
                break

            if isinstance(node, GraphNode) and node.max_visits > 0:
                if visit_counts[current] > node.max_visits:
                    yield GraphEvent(
                        type=GraphEventType.ERROR,
                        node_name=current,
                        error=RuntimeError(
                            f"Node {current!r} exceeded max_visits={node.max_visits}"
                        ),
                    )
                    break

            if self.enable_loop_detection:
                h = _state_hash(state)
                if h in seen_hashes:
                    loop_count += 1
                    self._notify("on_loop_detected", run_id, current, loop_count)
                    self._trace_step(
                        trace,
                        StepType.GRAPH_LOOP_DETECTED,
                        node_name=current,
                        step_number=step,
                    )
                    yield GraphEvent(
                        type=GraphEventType.ERROR,
                        node_name=current,
                        error=RuntimeError(f"Hard loop detected at step {step}"),
                    )
                    break
                seen_hashes.add(h)
                if h == prev_hash:
                    unchanged_streak += 1
                    if unchanged_streak >= self.stall_threshold:
                        stall_count += 1
                        self._notify("on_stall_detected", run_id, current, stall_count)
                        self._trace_step(
                            trace,
                            StepType.GRAPH_STALL,
                            node_name=current,
                            step_number=step,
                        )
                else:
                    unchanged_streak = 0
                prev_hash = h

            self._notify("on_node_start", run_id, current, step)
            self._trace_step(trace, StepType.GRAPH_NODE_START, node_name=current, step_number=step)
            yield GraphEvent(type=GraphEventType.NODE_START, node_name=current)
            node_start_time = time.time()

            try:
                if isinstance(node, ParallelGroupNode):
                    yield GraphEvent(
                        type=GraphEventType.PARALLEL_START,
                        node_name=current,
                    )
                    child_results, state = await self._aexecute_parallel(node, state, trace, run_id)
                    node_results.update(child_results)
                    for _cl in child_results.values():
                        for _cr in _cl:
                            if _cr.usage:
                                usage = _merge_usage(usage, _cr.usage)
                    yield GraphEvent(type=GraphEventType.PARALLEL_END, node_name=current)

                elif isinstance(node, SubgraphNode):
                    result, state = await self._aexecute_subgraph(node, state, trace, run_id)
                    node_results.setdefault(current, []).append(result)
                    if result.usage:
                        usage = _merge_usage(usage, result.usage)

                else:
                    result, state, interrupted = await self._aexecute_node(
                        node, state, trace, run_id
                    )
                    if result.usage:
                        usage = _merge_usage(usage, result.usage)
                    node_results.setdefault(current, []).append(result)

                    if result.content:
                        yield GraphEvent(
                            type=GraphEventType.NODE_CHUNK,
                            node_name=current,
                            chunk=result.content,
                        )

                    if interrupted:
                        interrupt_key = state.metadata.get(_STATE_KEY_PENDING_INTERRUPT, "")
                        if checkpoint_store:
                            ckpt_id = checkpoint_store.save(run_id, state, step)
                        else:
                            ckpt_id = f"{run_id}_{step}"
                        self._notify("on_graph_interrupt", run_id, current, ckpt_id)
                        self._trace_step(
                            trace,
                            StepType.GRAPH_INTERRUPT,
                            node_name=current,
                            interrupt_key=interrupt_key,
                            checkpoint_id=ckpt_id,
                        )
                        duration_ms = (time.time() - graph_start) * 1000
                        self._notify("on_graph_end", run_id, self.name, step, duration_ms)
                        yield GraphEvent(
                            type=GraphEventType.GRAPH_INTERRUPT,
                            node_name=current,
                            interrupt_id=ckpt_id,
                        )
                        break

            except Exception as exc:
                self._notify("on_graph_error", run_id, self.name, current, exc)
                yield GraphEvent(
                    type=GraphEventType.ERROR,
                    node_name=current,
                    error=exc,
                )
                if self.error_policy == ErrorPolicy.ABORT:
                    break

            node_duration_ms = (time.time() - node_start_time) * 1000
            self._notify("on_node_end", run_id, current, step, node_duration_ms)
            self._trace_step(
                trace,
                StepType.GRAPH_NODE_END,
                node_name=current,
                step_number=step,
                duration_ms=node_duration_ms,
            )
            yield GraphEvent(type=GraphEventType.NODE_END, node_name=current)

            if checkpoint_store:
                checkpoint_store.save(run_id, state, step)

            prev_node = current
            try:
                current = self._resolve_next_node(current, state, trace, run_id)
            except GraphExecutionError as exc:
                yield GraphEvent(
                    type=GraphEventType.ERROR,
                    node_name=prev_node,
                    error=exc,
                )
                break
            yield GraphEvent(
                type=GraphEventType.ROUTING,
                node_name=prev_node,
                next_node=current,
            )

        total_duration_ms = (time.time() - graph_start) * 1000
        self._trace_step(
            trace,
            StepType.GRAPH_NODE_END,
            node_name=self.name,
            step_number=step,
            duration_ms=total_duration_ms,
        )
        self._notify("on_graph_end", run_id, self.name, step, total_duration_ms)

        final_result = GraphResult(
            content=state.data.get(STATE_KEY_LAST_OUTPUT, ""),
            state=state,
            node_results=node_results,
            trace=trace,
            total_usage=usage,
            steps=step,
            stalls=stall_count,
            loops_detected=loop_count,
        )
        yield GraphEvent(
            type=GraphEventType.GRAPH_END,
            state=state,
            result=final_result,
        )

    # ------------------------------------------------------------------
    # HITL resume
    # ------------------------------------------------------------------

    def resume(
        self,
        interrupt_id: str,
        response: Any,
        checkpoint_store: "CheckpointStore",
    ) -> GraphResult:
        """Resume execution after an InterruptRequest.

        Args:
            interrupt_id: The checkpoint ID from GraphResult.interrupt_id.
            response: The human's response to inject into the generator node's yield.
            checkpoint_store: The same store used during the interrupted run.

        Returns:
            GraphResult — may be interrupted again if there are multiple yields.
        """
        return run_sync(self.aresume(interrupt_id, response, checkpoint_store))

    async def aresume(
        self,
        interrupt_id: str,
        response: Any,
        checkpoint_store: "CheckpointStore",
    ) -> GraphResult:
        """Async resume after an InterruptRequest."""
        return await self.arun(
            GraphState(),  # will be overridden by checkpoint
            checkpoint_store=checkpoint_store,
            checkpoint_id=interrupt_id,
            _interrupt_response=response,
        )

    # ------------------------------------------------------------------
    # Node execution helpers
    # ------------------------------------------------------------------

    async def _aexecute_node(
        self,
        node: GraphNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[AgentResult, GraphState, bool]:
        """Execute a single GraphNode. Returns (result, new_state, interrupted)."""
        from ..agent.core import Agent  # noqa: F811

        # Build input messages based on context_mode or custom transform
        if node.input_transform is not None:
            messages = node.input_transform(state)
        else:
            messages = build_context_messages(node, state)

        agent_or_fn = node.agent

        if isinstance(agent_or_fn, Agent):
            result = await agent_or_fn.arun(messages, parent_run_id=run_id)
            out_transform = node.output_transform or default_output_transform
            new_state = out_transform(result, state)
            return result, new_state, False

        elif inspect.isasyncgenfunction(agent_or_fn):
            return await self._aexecute_generator_node(node, state, trace, run_id)

        elif inspect.isgeneratorfunction(agent_or_fn):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._execute_generator_node_sync, node, state, trace, run_id
            )

        elif asyncio.iscoroutinefunction(agent_or_fn):
            new_state = await agent_or_fn(state)
            if new_state is None:
                new_state = state
            result = _make_synthetic_result(new_state)
            return result, new_state, False

        else:
            # Plain sync callable
            loop = asyncio.get_running_loop()
            new_state = await loop.run_in_executor(None, agent_or_fn, state)
            if new_state is None:
                new_state = state
            result = _make_synthetic_result(new_state)
            return result, new_state, False

    async def _aexecute_generator_node(
        self,
        node: GraphNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[AgentResult, GraphState, bool]:
        """Execute an async generator node with HITL support."""
        gen = node.agent(state)
        interrupt_index = 0

        async for value in gen:
            if isinstance(value, InterruptRequest):
                interrupt_key = f"{node.name}_{interrupt_index}"

                if interrupt_key in state._interrupt_responses:
                    # Resume path: inject stored response into generator
                    try:
                        await gen.asend(state._interrupt_responses[interrupt_key])
                        del state._interrupt_responses[interrupt_key]
                        interrupt_index += 1
                        continue
                    except StopAsyncIteration:
                        break
                else:
                    # First-pass: store interrupt key and signal pause
                    value.interrupt_key = interrupt_key
                    state.metadata[_STATE_KEY_PENDING_INTERRUPT] = interrupt_key
                    self._notify("on_graph_interrupt", run_id, node.name, interrupt_key)
                    self._trace_step(
                        trace,
                        StepType.GRAPH_INTERRUPT,
                        node_name=node.name,
                        interrupt_key=interrupt_key,
                    )
                    synthetic = _make_synthetic_result(state)
                    return synthetic, state, True

            interrupt_index = 0

        result = _make_synthetic_result(state)
        return result, state, False

    def _execute_generator_node_sync(
        self,
        node: GraphNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[AgentResult, GraphState, bool]:
        """Execute a sync generator node with HITL support."""
        gen = node.agent(state)
        interrupt_index = 0

        for value in gen:
            if isinstance(value, InterruptRequest):
                interrupt_key = f"{node.name}_{interrupt_index}"

                if interrupt_key in state._interrupt_responses:
                    try:
                        gen.send(state._interrupt_responses[interrupt_key])
                        del state._interrupt_responses[interrupt_key]
                        interrupt_index += 1
                        continue
                    except StopIteration:
                        break
                else:
                    value.interrupt_key = interrupt_key
                    state.metadata[_STATE_KEY_PENDING_INTERRUPT] = interrupt_key
                    self._notify("on_graph_interrupt", run_id, node.name, interrupt_key)
                    self._trace_step(
                        trace,
                        StepType.GRAPH_INTERRUPT,
                        node_name=node.name,
                        interrupt_key=interrupt_key,
                    )
                    synthetic = _make_synthetic_result(state)
                    return synthetic, state, True

            interrupt_index = 0

        result = _make_synthetic_result(state)
        return result, state, False

    async def _aexecute_parallel(
        self,
        node: ParallelGroupNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[Dict[str, List[AgentResult]], GraphState]:
        """Fan out to child nodes in parallel and merge results."""
        self._notify("on_parallel_start", run_id, node.name, node.child_node_names)
        self._trace_step(
            trace,
            StepType.GRAPH_PARALLEL_START,
            node_name=node.name,
            children=node.child_node_names,
        )

        branch_states = [copy.deepcopy(state) for _ in node.child_node_names]

        # Apply scatter patches per-branch (if this parallel group came from Scatter)
        scatter_patches = self._scatter_patches.pop(node.name, {})
        try:
            for i, child_name in enumerate(node.child_node_names):
                if child_name in scatter_patches:
                    branch_states[i].data.update(scatter_patches[child_name])

            async def run_child(
                child_name: str, branch_state: GraphState
            ) -> Tuple[str, AgentResult, GraphState]:
                child_node = self._nodes.get(child_name)
                if child_node is None:
                    raise GraphExecutionError(
                        self.name, child_name, KeyError(f"Child node {child_name!r} not found"), 0
                    )
                if isinstance(child_node, GraphNode):
                    result, new_state, _ = await self._aexecute_node(
                        child_node, branch_state, trace, run_id
                    )
                else:
                    result = _make_synthetic_result(branch_state)
                    new_state = branch_state
                return child_name, result, new_state

            child_outputs = await asyncio.gather(
                *[
                    run_child(name, bstate)
                    for name, bstate in zip(node.child_node_names, branch_states)
                ],
                return_exceptions=True,
            )

            child_results: Dict[str, List[AgentResult]] = {}
            branch_final_states: List[GraphState] = []
            for i, output in enumerate(child_outputs):
                if isinstance(output, BaseException):
                    child_name = node.child_node_names[i]
                    state.errors.append(
                        {"node": child_name, "error": str(output), "type": type(output).__name__}
                    )
                    if self.error_policy == ErrorPolicy.ABORT:
                        exc = output if isinstance(output, Exception) else Exception(str(output))
                        raise GraphExecutionError(self.name, child_name, exc, 0) from output
                    continue  # SKIP: log error and proceed
                child_name, result, new_state = output
                child_results.setdefault(child_name, []).append(result)
                branch_final_states.append(new_state)

            if not branch_final_states:
                # All children failed — return parent state unchanged
                merged = state
            elif node.merge_fn:
                merged = node.merge_fn(branch_final_states)
            else:
                merged = merge_states(branch_final_states, node.merge_policy)

            self._notify("on_parallel_end", run_id, node.name, len(child_outputs))
            self._trace_step(trace, StepType.GRAPH_PARALLEL_END, node_name=node.name)
            return child_results, merged
        except BaseException:
            # Restore scatter patches so a retry/resume can reuse them
            if scatter_patches:
                self._scatter_patches[node.name] = scatter_patches
            raise

    async def _aexecute_subgraph(
        self,
        node: SubgraphNode,
        state: GraphState,
        trace: AgentTrace,
        run_id: str,
    ) -> Tuple[AgentResult, GraphState]:
        """Execute a nested AgentGraph as a node."""
        # Build subgraph input state
        sub_state = GraphState.from_prompt(
            state.data.get(STATE_KEY_LAST_OUTPUT, "")
            or (state.messages[-1].content if state.messages else "")
        )

        # Map parent data keys to subgraph data keys
        for parent_key, sub_key in node.input_map.items():
            if parent_key in state.data:
                sub_state.data[sub_key] = state.data[parent_key]

        # Run the subgraph
        sub_result = await node.graph.arun(sub_state, _interrupt_response=None)

        # Map subgraph output keys back to parent
        for sub_key, parent_key in node.output_map.items():
            if sub_key in sub_result.state.data:
                state.data[parent_key] = sub_result.state.data[sub_key]

        # Default: write subgraph content as last output
        state.data[STATE_KEY_LAST_OUTPUT] = sub_result.content
        state.messages.extend(sub_result.state.messages[-2:])  # last 2 messages
        state.history.extend(sub_result.state.history)

        synthetic = AgentResult(
            message=Message(role=Role.ASSISTANT, content=sub_result.content),
            iterations=sub_result.steps,
            usage=sub_result.total_usage,
        )
        return synthetic, state

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_next_node(
        self, current: str, state: GraphState, trace: AgentTrace, run_id: str
    ) -> str:
        """Determine the next node to execute."""
        result: Optional[str] = None

        if current in self._conditional_edges:
            router = self._conditional_edges[current]
            try:
                raw = router(state)
            except Exception as exc:
                raise GraphExecutionError(
                    self.name,
                    current,
                    RuntimeError(f"Router function raised: {exc}"),
                    0,
                ) from exc

            if raw is None:
                raise GraphExecutionError(
                    self.name,
                    current,
                    ValueError("Router returned None — did you forget a return statement?"),
                    0,
                )

            # Handle Scatter (dynamic fan-out)
            if isinstance(raw, list):
                if not raw:
                    raise GraphExecutionError(
                        self.name,
                        current,
                        ValueError("Router returned empty list — expected Scatter items"),
                        0,
                    )
                if isinstance(raw[0], Scatter):
                    next_node = self._handle_scatter(raw, state)
                    result = next_node
                else:
                    raise GraphExecutionError(
                        self.name,
                        current,
                        TypeError(
                            f"Router returned list of {type(raw[0]).__name__},"
                            " expected list of Scatter"
                        ),
                        0,
                    )
            elif isinstance(raw, Scatter):
                next_node = self._handle_scatter([raw], state)
                result = next_node
            elif isinstance(raw, _Goto):
                result = raw.node_name
            elif isinstance(raw, _Update):
                state.data.update(raw.patch)
                # After applying patch, check for static edge from current node
                if current in self._edges:
                    result = self._edges[current]
                else:
                    result = self.END
            elif isinstance(raw, str):
                if current in self._path_maps:
                    pmap = self._path_maps[current]
                    if pmap and raw not in pmap and raw != self.END:
                        raise GraphExecutionError(
                            self.name,
                            current,
                            ValueError(
                                f"Router returned {raw!r} but path_map keys are"
                                f" {list(pmap.keys())}"
                            ),
                            0,
                        )
                    result = pmap.get(raw, raw)
                else:
                    result = raw
            else:
                raise GraphExecutionError(
                    self.name,
                    current,
                    TypeError(
                        f"Router returned {type(raw).__name__}, expected"
                        " str / Scatter / goto() / update()"
                    ),
                    0,
                )

        elif current in self._edges:
            result = self._edges[current]
        else:
            result = self.END  # no outgoing edge → implicit END

        if result is None:
            result = self.END

        self._notify("on_graph_routing", run_id, current, result)
        self._trace_step(trace, StepType.GRAPH_ROUTING, from_node=current, to_node=result)
        return result

    def _handle_scatter(self, scatters: List[Scatter], state: GraphState) -> str:
        """Handle dynamic fan-out via Scatter objects.

        Creates a temporary ParallelGroupNode and registers it for one-time execution.
        State patches are stored and applied per-branch during parallel execution.
        """
        group_name = f"__scatter_{uuid.uuid4().hex[:8]}__"
        node_names = []
        for sc in scatters:
            node_names.append(sc.node_name)

        # Store patches so _aexecute_parallel can apply them per-branch
        self._scatter_patches[group_name] = {sc.node_name: sc.state_patch for sc in scatters}

        self.add_parallel_nodes(group_name, node_names)
        return group_name

    def _over_budget(self, usage: UsageStats) -> bool:
        """Check whether cumulative usage exceeds graph-level budget limits."""
        if self.max_total_tokens and usage.total_tokens >= self.max_total_tokens:
            return True
        if self.max_cost_usd and usage.cost_usd >= self.max_cost_usd:
            return True
        return False

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> GraphState:
        """Makes this graph usable as a callable node in another graph."""
        result = self.run(state)
        return result.state

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart diagram string.

        Returns a string you can paste into any Mermaid renderer.

        Example output::

            graph TD
                planner["planner (Agent)"] --> writer
                writer["writer (Agent)"] -->|conditional| reviewer
                writer -->|conditional| __end__["END"]
        """
        lines = ["graph TD"]

        def _node_label(name: str) -> str:
            node = self._nodes.get(name)
            if node is None:
                return f'{name}["{name}"]'
            if isinstance(node, ParallelGroupNode):
                children = ", ".join(node.child_node_names)
                return f'{name}["{name} [parallel: {children}]"]'
            if isinstance(node, SubgraphNode):
                return f'{name}["{name} (subgraph)"]'
            agent = getattr(node, "agent", None)
            if agent is not None:
                agent_type = type(agent).__name__
                return f'{name}["{name} ({agent_type})"]'
            return f'{name}["{name}"]'

        # Static edges
        for from_node, to_node in self._edges.items():
            from_label = _node_label(from_node)
            if to_node == self.END:
                to_label = '__end__["END"]'
            else:
                to_label = _node_label(to_node)
            lines.append(f"    {from_label} --> {to_label}")

        # Conditional edges
        for from_node, _router_fn in self._conditional_edges.items():
            from_label = _node_label(from_node)
            if from_node in self._path_maps:
                for val, target in self._path_maps[from_node].items():
                    if target == self.END:
                        to_label = '__end__["END"]'
                    else:
                        to_label = _node_label(target)
                    lines.append(f'    {from_label} -->|"{val}"| {to_label}')
            else:
                lines.append(f'    {from_label} -->|"conditional"| ???')

        return "\n".join(lines)

    def visualize(self, format: str = "ascii") -> None:
        """Print a visualization of the graph.

        Args:
            format: "ascii" (default, no deps) or "png" (requires graphviz).
        """
        if format == "ascii":
            self._print_ascii()
        elif format == "png":
            try:
                import graphviz  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "graphviz package required for PNG output: pip install graphviz"
                ) from exc
            dot = graphviz.Digraph(name=self.name)
            for from_node, to_node in self._edges.items():
                dot.edge(from_node, to_node)
            for from_node in self._conditional_edges:
                if from_node in self._path_maps:
                    for val, target in self._path_maps[from_node].items():
                        dot.edge(from_node, target, label=val)
            dot.render(self.name, format="png", cleanup=True)
            print(f"Graph saved as {self.name}.png")
        else:
            raise ValueError(f"Unknown format {format!r}. Use 'ascii' or 'png'.")

    def _print_ascii(self) -> None:
        """Print ASCII box-drawing visualization."""
        print(f"\nGraph: {self.name}")
        print("=" * 40)
        if self._entry_node:
            print(f"Entry: {self._entry_node}")
        print()
        for name, node in self._nodes.items():
            marker = "→ " if name == self._entry_node else "  "
            if isinstance(node, ParallelGroupNode):
                print(f"{marker}[parallel] {name}: {node.child_node_names}")
            elif isinstance(node, SubgraphNode):
                print(f"{marker}[subgraph] {name}")
            else:
                agent_type = type(getattr(node, "agent", None)).__name__
                print(f"{marker}[node]     {name} ({agent_type})")

        print()
        print("Edges:")
        for from_node, to_node in self._edges.items():
            print(f"  {from_node} ──→ {to_node}")
        for from_node in self._conditional_edges:
            if from_node in self._path_maps:
                for val, target in self._path_maps[from_node].items():
                    print(f"  {from_node} ──[{val}]──→ {target}")
            else:
                print(f"  {from_node} ──[conditional]──→ ???")
        print()


__all__ = [
    "AgentGraph",
    "GraphResult",
    "ErrorPolicy",
]
