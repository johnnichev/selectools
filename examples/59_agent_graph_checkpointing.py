"""
Example 59: Checkpointing with AgentGraph

Demonstrates durable mid-graph persistence using FileCheckpointStore:
- Save checkpoints after each node
- Resume from any checkpoint
- All three backends: InMemory, File, SQLite

Uses LocalProvider so no API keys are needed.
"""

import os
import tempfile

from selectools import Agent, AgentConfig
from selectools.orchestration import (
    STATE_KEY_LAST_OUTPUT,
    AgentGraph,
    FileCheckpointStore,
    GraphState,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from selectools.providers.stubs import LocalProvider
from selectools.tools.decorators import tool


@tool()
def process_data(data: str) -> str:
    """Process some data."""
    return f"Processed: {data}"


def make_agent(name: str, response: str) -> Agent:
    return Agent(
        config=AgentConfig(model="gpt-4o-mini"),
        provider=LocalProvider(responses=[response]),
        tools=[process_data],
    )


def build_graph() -> AgentGraph:
    step1 = make_agent("step1", "Step 1 complete: data ingested")
    step2 = make_agent("step2", "Step 2 complete: data transformed")
    step3 = make_agent("step3", "Step 3 complete: data published")

    graph = AgentGraph(name="data_pipeline")
    graph.add_node("ingest", step1)
    graph.add_node("transform", step2)
    graph.add_node("publish", step3)
    graph.add_edge("ingest", "transform")
    graph.add_edge("transform", "publish")
    graph.add_edge("publish", AgentGraph.END)
    graph.set_entry("ingest")
    return graph


def demo_inmemory():
    print("=== InMemoryCheckpointStore ===")
    store = InMemoryCheckpointStore()
    graph = build_graph()
    result = graph.run("Process dataset", checkpoint_store=store)
    print(f"Result: {result.content}")

    # List checkpoints
    metas = store.list(result.trace.run_id)
    print(f"Checkpoints saved: {len(metas)}")
    if metas:
        print(f"  Latest: step={metas[-1].step}, node={metas[-1].node_name}")


def demo_file():
    print("\n=== FileCheckpointStore ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileCheckpointStore(tmpdir)
        graph = build_graph()
        result = graph.run("Process dataset", checkpoint_store=store)
        print(f"Result: {result.content}")

        # Show files on disk
        for graph_dir in os.listdir(tmpdir):
            files = os.listdir(os.path.join(tmpdir, graph_dir))
            print(f"Files saved: {len(files)} checkpoints in {tmpdir}/{graph_dir}/")


def demo_sqlite():
    print("\n=== SQLiteCheckpointStore ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "checkpoints.db")
        store = SQLiteCheckpointStore(db_path)
        graph = build_graph()
        result = graph.run("Process dataset", checkpoint_store=store)
        print(f"Result: {result.content}")

        metas = store.list(result.trace.run_id)
        print(f"SQLite checkpoints: {len(metas)}")


def main():
    demo_inmemory()
    demo_file()
    demo_sqlite()


if __name__ == "__main__":
    main()
