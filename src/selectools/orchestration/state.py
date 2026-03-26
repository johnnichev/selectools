"""
Graph state primitives for multi-agent orchestration.

GraphState is the shared context passed between nodes in an AgentGraph.
All state mutation is explicit — nodes receive the current state and return
a new (or mutated) state. The canonical inter-node handoff key is
STATE_KEY_LAST_OUTPUT.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..types import AgentResult, Message


STATE_KEY_LAST_OUTPUT: str = "__last_output__"


class MergePolicy(str, Enum):
    """Policy for merging parallel branch states.

    LAST_WINS: On conflicting keys, the last branch's value wins.
    FIRST_WINS: On conflicting keys, the first branch's value wins.
    APPEND: List values are appended across branches; non-list conflicts use LAST_WINS.
    """

    LAST_WINS = "last_wins"
    FIRST_WINS = "first_wins"
    APPEND = "append"


class ContextMode(str, Enum):
    """Controls what conversation history is forwarded to a node's agent.

    LAST_MESSAGE: Only the most recent user message (default). Prevents context explosion.
    LAST_N: Last N messages (configurable via GraphNode.context_n).
    FULL: Full state.messages history.
    SUMMARY: Provider-compressed summary of prior messages.
    CUSTOM: Use GraphNode.input_transform for maximum flexibility.
    """

    LAST_MESSAGE = "last_message"
    LAST_N = "last_n"
    FULL = "full"
    SUMMARY = "summary"
    CUSTOM = "custom"


@dataclass
class GraphState:
    """Shared context passed between nodes in an AgentGraph.

    Attributes:
        messages: Accumulated conversation messages across all nodes (append-only).
        data: Inter-node key-value store for arbitrary state.
        current_node: Name of the currently executing node.
        history: Ordered list of (node_name, AgentResult) tuples from completed nodes.
        metadata: User-attached data carried through checkpoints (request_id, user_id, etc.).
        errors: Records of errors from failed nodes (populated when error_policy=SKIP).
        _interrupt_responses: Keyed by interrupt_key; injected by graph.resume().
                              Excluded from to_dict() serialization.
    """

    messages: List[Any] = field(default_factory=list)  # List[Message]
    data: Dict[str, Any] = field(default_factory=dict)
    current_node: str = ""
    history: List[Tuple[str, Any]] = field(default_factory=list)  # List[Tuple[str, AgentResult]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _interrupt_responses: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def last_output(self) -> str:
        """The most recent node output (alias for data[STATE_KEY_LAST_OUTPUT])."""
        return self.data.get(STATE_KEY_LAST_OUTPUT, "")

    @last_output.setter
    def last_output(self, value: str) -> None:
        self.data[STATE_KEY_LAST_OUTPUT] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation. Excludes _interrupt_responses."""
        from ..types import AgentResult, Message  # noqa: F811

        def _serialize_msg(m: Any) -> Dict[str, Any]:
            if isinstance(m, Message):
                return m.to_dict()
            if isinstance(m, dict):
                return m
            return {"content": str(m), "role": "user"}

        def _serialize_result(r: Any) -> Dict[str, Any]:
            if isinstance(r, AgentResult):
                return {
                    "content": r.content,
                    "iterations": r.iterations,
                    "tool_calls": [
                        {"tool_name": tc.tool_name, "parameters": tc.parameters}
                        for tc in (r.tool_calls or [])
                    ],
                }
            if isinstance(r, dict):
                return r
            return {"content": str(r)}

        return {
            "messages": [_serialize_msg(m) for m in self.messages],
            "data": copy.deepcopy(self.data),
            "current_node": self.current_node,
            "history": [(name, _serialize_result(res)) for name, res in self.history],
            "metadata": copy.deepcopy(self.metadata),
            "errors": copy.deepcopy(self.errors),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphState":
        """Reconstruct a GraphState from a dictionary produced by to_dict()."""
        from ..types import Message, Role  # noqa: F811

        messages: List[Any] = []
        for m in d.get("messages", []):
            if isinstance(m, dict) and "role" in m:
                try:
                    messages.append(Message.from_dict(m))
                except Exception:  # nosec B110
                    pass
            elif isinstance(m, dict):
                messages.append(m)

        history = []
        for name, res in d.get("history", []):
            history.append((name, res))

        return cls(
            messages=messages,
            data=copy.deepcopy(d.get("data", {})),
            current_node=d.get("current_node", ""),
            history=history,
            metadata=copy.deepcopy(d.get("metadata", {})),
            errors=copy.deepcopy(d.get("errors", [])),
        )

    @classmethod
    def from_prompt(cls, prompt: str) -> "GraphState":
        """Create a new GraphState with a single user message."""
        from ..types import Message, Role  # noqa: F811

        return cls(messages=[Message(role=Role.USER, content=prompt)])


@dataclass
class InterruptRequest:
    """Yielded from generator nodes to pause execution for human input.

    When a generator node yields an InterruptRequest, the graph saves a
    checkpoint and returns GraphResult(interrupted=True). The caller then
    calls graph.resume(interrupt_id, response) to continue from the exact
    yield point.

    Attributes:
        prompt: Human-readable description of what input is needed.
        payload: Optional data to show the human (analysis results, draft, etc.).
        interrupt_key: Auto-set by the graph engine to f"{node_name}_{yield_index}".
    """

    prompt: str
    payload: Any = None
    interrupt_key: str = ""


@dataclass
class Scatter:
    """Returned from routing functions to create dynamic parallel branches.

    When a router returns List[Scatter], the graph creates a dynamic parallel
    group and fans out to child nodes, one branch per Scatter item.

    Attributes:
        node_name: Target node to execute in this branch.
        state_patch: Dict merged into branch state.data before execution.
    """

    node_name: str
    state_patch: Dict[str, Any] = field(default_factory=dict)


class GraphEventType(str, Enum):
    """Type of a GraphEvent yielded from AgentGraph.astream()."""

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


@dataclass
class GraphEvent:
    """A single event yielded from AgentGraph.astream().

    Attributes:
        type: The event type (GraphEventType value).
        node_name: Name of the node this event relates to (if applicable).
        chunk: Text chunk (for NODE_CHUNK events during agent streaming).
        state: Current graph state snapshot (for GRAPH_END events).
        result: Final graph result (for GRAPH_END events).
        next_node: Next node to execute (for ROUTING events).
        error: Exception (for ERROR events).
        interrupt_id: Checkpoint ID for resumption (for GRAPH_INTERRUPT events).
    """

    type: GraphEventType
    node_name: Optional[str] = None
    chunk: str = ""
    state: Optional[GraphState] = None
    result: Optional[Any] = None  # GraphResult (forward ref)
    next_node: Optional[str] = None
    error: Optional[Exception] = None
    interrupt_id: Optional[str] = None


# ------------------------------------------------------------------
# Routing primitives
# ------------------------------------------------------------------


@dataclass
class _Goto:
    """Internal: direct routing to a named node."""

    node_name: str


@dataclass
class _Update:
    """Internal: state patch to apply before routing."""

    patch: Dict[str, Any]


def goto(node_name: str) -> _Goto:
    """Return a routing directive that sends execution to a specific node."""
    return _Goto(node_name=node_name)


def update(patch: Dict[str, Any]) -> _Update:
    """Return a state mutation directive to apply before routing."""
    return _Update(patch=patch)


def merge_states(states: List[GraphState], policy: MergePolicy) -> GraphState:
    """Merge a list of parallel branch states into a single GraphState.

    Rules:
        messages: Always concatenated (append semantics — all branches contribute).
        history:  Always concatenated.
        errors:   Always concatenated.
        data:     Merged per the specified MergePolicy.
        metadata: Merged with LAST_WINS.
        current_node: Taken from last state.
    """
    if not states:
        raise ValueError("Cannot merge empty list of states")
    if len(states) == 1:
        return copy.deepcopy(states[0])

    merged_messages: List[Any] = []
    merged_history: List[Any] = []
    merged_errors: List[Any] = []
    merged_data: Dict[str, Any] = {}
    merged_metadata: Dict[str, Any] = {}

    for st in states:
        merged_messages.extend(st.messages)
        merged_history.extend(st.history)
        merged_errors.extend(st.errors)
        merged_metadata.update(st.metadata)

        if policy == MergePolicy.FIRST_WINS:
            for k, v in st.data.items():
                if k not in merged_data:
                    merged_data[k] = v
        elif policy == MergePolicy.APPEND:
            for k, v in st.data.items():
                if k in merged_data:
                    if isinstance(merged_data[k], list) and isinstance(v, list):
                        merged_data[k] = merged_data[k] + v
                    else:
                        merged_data[k] = v  # fallback to LAST_WINS for non-list
                else:
                    merged_data[k] = v
        else:  # LAST_WINS (default)
            merged_data.update(st.data)

    return GraphState(
        messages=merged_messages,
        data=merged_data,
        current_node=states[-1].current_node,
        history=merged_history,
        metadata=merged_metadata,
        errors=merged_errors,
    )


__all__ = [
    "STATE_KEY_LAST_OUTPUT",
    "MergePolicy",
    "ContextMode",
    "GraphState",
    "InterruptRequest",
    "Scatter",
    "GraphEventType",
    "GraphEvent",
    "goto",
    "update",
    "_Goto",
    "_Update",
    "merge_states",
]
