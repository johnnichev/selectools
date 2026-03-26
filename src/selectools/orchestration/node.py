"""
Graph node types for multi-agent orchestration.

GraphNode wraps an Agent (or callable) as a node in an AgentGraph.
ParallelGroupNode registers a set of nodes to run concurrently.
SubgraphNode embeds an AgentGraph as a node with explicit state key mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .state import STATE_KEY_LAST_OUTPUT, ContextMode, GraphState

if TYPE_CHECKING:
    from ..agent.core import Agent
    from ..types import AgentResult, Message
    from .graph import ErrorPolicy


@dataclass
class GraphNode:
    """A single node in an AgentGraph.

    Attributes:
        name: Unique node identifier within the graph.
        agent: The Agent, callable, or async generator function to execute.
        input_transform: Optional function mapping GraphState → List[Message].
                         Overrides the default ContextMode-based transform.
        output_transform: Optional function mapping (AgentResult, GraphState) → GraphState.
                          Overrides the default transform that writes STATE_KEY_LAST_OUTPUT.
        context_mode: Controls what history is forwarded to the node's agent.
                      Default LAST_MESSAGE prevents context explosion.
        context_n: Used when context_mode == LAST_N; number of messages to keep.
        max_iterations: Maximum re-execution count within a single visit.
        max_visits: Maximum times this node may execute in a graph run (0 = unlimited).
        error_policy: Per-node override; None inherits from the graph-level policy.
    """

    name: str
    agent: Any  # Agent | Callable | AsyncGenerator
    input_transform: Optional[Callable] = None
    output_transform: Optional[Callable] = None
    context_mode: ContextMode = ContextMode.LAST_MESSAGE
    context_n: int = 6
    max_iterations: int = 1
    max_visits: int = 0
    error_policy: Optional[Any] = None  # Optional[ErrorPolicy]


@dataclass
class ParallelGroupNode:
    """A first-class node that fans out to child nodes and merges results.

    Registered in AgentGraph._nodes like any other node. Edges can point
    to it by name. When the graph executes this node, it runs all
    child_node_names in parallel via asyncio.gather and merges results.

    Attributes:
        name: Unique node identifier.
        child_node_names: Names of nodes to execute in parallel.
        merge_policy: How to merge parallel branch states.
        merge_fn: Optional custom merge function — receives List[GraphState],
                  returns merged GraphState. Overrides merge_policy when set.
    """

    name: str
    child_node_names: List[str]
    merge_policy: Any = None  # MergePolicy (set in __post_init__)
    merge_fn: Optional[Callable[[List[GraphState]], GraphState]] = None

    def __post_init__(self) -> None:
        """Set default merge_policy if not provided."""
        if self.merge_policy is None:
            from .state import MergePolicy

            self.merge_policy = MergePolicy.LAST_WINS


@dataclass
class SubgraphNode:
    """Wraps an AgentGraph as a node with explicit state key mapping.

    Attributes:
        name: Unique node identifier.
        graph: The AgentGraph to execute.
        input_map: parent state.data key → subgraph state.data key.
        output_map: subgraph state.data key → parent state.data key.
    """

    name: str
    graph: Any  # AgentGraph (forward ref)
    input_map: Dict[str, str] = field(default_factory=dict)
    output_map: Dict[str, str] = field(default_factory=dict)


def _last_user_message(msgs: List[Any]) -> Optional[Any]:
    """Return the most recent user-role message, or None if not found."""
    for msg in reversed(msgs):
        role = getattr(msg, "role", None)
        if role is not None:
            role_val = role.value if hasattr(role, "value") else str(role)
            if role_val == "user":
                return msg
    return None


def default_input_transform(state: GraphState) -> List[Any]:
    """Default input transform: reads STATE_KEY_LAST_OUTPUT from state.data.

    Falls back to the last user message in state.messages if absent (first-node case).
    Falls back to all messages if state.messages is empty.
    """
    from ..types import Message, Role  # noqa: F811

    last = state.data.get(STATE_KEY_LAST_OUTPUT)
    if last and isinstance(last, str):
        return [Message(role=Role.USER, content=last)]

    msg = _last_user_message(state.messages)
    if msg is not None:
        return [msg]

    return list(state.messages)


def default_output_transform(result: Any, state: GraphState) -> GraphState:
    """Default output transform: appends assistant message and writes STATE_KEY_LAST_OUTPUT.

    Appends the result's content as an ASSISTANT message to state.messages,
    stores result.content in state.data[STATE_KEY_LAST_OUTPUT], and
    appends (current_node, result) to state.history.
    """
    from ..types import Message, Role  # noqa: F811

    state.messages.append(Message(role=Role.ASSISTANT, content=result.content or ""))
    state.data[STATE_KEY_LAST_OUTPUT] = result.content or ""
    state.history.append((state.current_node, result))
    return state


def build_context_messages(node: GraphNode, state: GraphState) -> List[Any]:
    """Build the input message list for a node based on its context_mode.

    Called by the graph engine when no custom input_transform is set.
    """
    from ..types import Message, Role  # noqa: F811

    mode = node.context_mode

    if mode == ContextMode.LAST_MESSAGE:
        msg = _last_user_message(state.messages)
        if msg is not None:
            return [msg]
        return [state.messages[-1]] if state.messages else []

    elif mode == ContextMode.LAST_N:
        return list(state.messages[-node.context_n :])

    elif mode == ContextMode.FULL:
        return list(state.messages)

    elif mode == ContextMode.SUMMARY:
        # Return a synthetic summary message — caller should handle via provider
        # For now, return last N messages (summary generation is expensive and provider-dependent)
        summary = state.data.get("__context_summary__", "")
        if summary:
            return [Message(role=Role.USER, content=summary)]
        return list(state.messages[-8:])  # fallback: last 8

    elif mode == ContextMode.CUSTOM:
        # Should have been handled by input_transform — return safe fallback
        return default_input_transform(state)

    return default_input_transform(state)


__all__ = [
    "GraphNode",
    "ParallelGroupNode",
    "SubgraphNode",
    "default_input_transform",
    "default_output_transform",
    "build_context_messages",
]
