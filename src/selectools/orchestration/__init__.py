"""
Multi-agent orchestration for selectools.

Provides AgentGraph for building directed graphs of Agent nodes with:
- Conditional routing (plain Python functions)
- Parallel fan-out (ParallelGroupNode + Scatter)
- Human-in-the-loop (generator nodes + InterruptRequest + resume)
- Checkpointing (InMemory / File / SQLite backends)
- Subgraph composition
- Loop and stall detection
- Budget and cancellation propagation

Also provides SupervisorAgent for high-level coordination:
- plan_and_execute, round_robin, dynamic, magentic strategies

Example::

    from selectools.orchestration import AgentGraph, GraphState

    graph = AgentGraph()
    graph.add_node("planner", planner_agent)
    graph.add_node("writer", writer_agent)
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", AgentGraph.END)
    graph.set_entry("planner")
    result = graph.run("Write a blog post about AI agents")
    print(result.content)
"""

from .checkpoint import (
    CheckpointMetadata,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from .graph import AgentGraph, ErrorPolicy, GraphResult
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
    ContextMode,
    GraphEvent,
    GraphEventType,
    GraphState,
    InterruptRequest,
    MergePolicy,
    Scatter,
    goto,
    merge_states,
    update,
)
from .supervisor import ModelSplit, SupervisorAgent, SupervisorStrategy

__all__ = [
    # Core graph
    "AgentGraph",
    "GraphResult",
    "ErrorPolicy",
    # State
    "GraphState",
    "GraphEvent",
    "GraphEventType",
    "MergePolicy",
    "ContextMode",
    "InterruptRequest",
    "Scatter",
    "STATE_KEY_LAST_OUTPUT",
    "goto",
    "update",
    "merge_states",
    # Nodes
    "GraphNode",
    "ParallelGroupNode",
    "SubgraphNode",
    "default_input_transform",
    "default_output_transform",
    "build_context_messages",
    # Checkpointing
    "CheckpointMetadata",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "SQLiteCheckpointStore",
    # Supervisor
    "SupervisorAgent",
    "SupervisorStrategy",
    "ModelSplit",
]
