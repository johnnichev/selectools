"""
Example 68: PostgresCheckpointStore for AgentGraph

Demonstrates using PostgreSQL as a checkpoint backend (v0.19.0):
- Create a PostgresCheckpointStore with a connection string
- Save, load, list, and delete checkpoints
- Compare with the existing SQLite and InMemory backends

NOTE: This example requires a running PostgreSQL instance and psycopg2:
    pip install psycopg2-binary

Since most environments won't have Postgres available, this example
uses InMemoryCheckpointStore as a stand-in and shows the Postgres
API surface alongside it.

Run:
    python examples/68_postgres_checkpoints.py
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from selectools.orchestration import AgentGraph, GraphState, InMemoryCheckpointStore
from selectools.orchestration.checkpoint import CheckpointMetadata

# --- PostgresCheckpointStore skeleton ---
# In v0.19.0 this ships as selectools.orchestration.PostgresCheckpointStore.
# Here we define a stand-in that mirrors the API using in-memory storage
# so the example runs without a real Postgres instance.


@dataclass
class PostgresCheckpointStore:
    """PostgreSQL-backed checkpoint store (v0.19.0).

    Uses psycopg2 for connection pooling and transactional safety.
    Supports concurrent access from multiple processes.

    Args:
        dsn: PostgreSQL connection string, e.g.
             "postgresql://user:pass@localhost:5432/mydb"
        table_name: Table to store checkpoints in (auto-created).
        pool_size: Connection pool size. Default: 5.
    """

    dsn: str
    table_name: str = "selectools_checkpoints"
    pool_size: int = 5

    def __post_init__(self) -> None:
        # In production: create psycopg2 connection pool and init table
        # For this demo, delegate to InMemoryCheckpointStore
        self._delegate = InMemoryCheckpointStore()
        self._connected = True

    def save(self, graph_id: str, state: GraphState, step: int) -> str:
        """Persist checkpoint to PostgreSQL. Returns checkpoint_id."""
        return self._delegate.save(graph_id, state, step)

    def load(self, checkpoint_id: str) -> Tuple[GraphState, int]:
        """Load checkpoint from PostgreSQL by ID."""
        return self._delegate.load(checkpoint_id)

    def list(self, graph_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a graph run."""
        return self._delegate.list(graph_id)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID."""
        return self._delegate.delete(checkpoint_id)

    @property
    def is_connected(self) -> bool:
        return self._connected


def main() -> None:
    print("=" * 60)
    print("PostgresCheckpointStore Demo")
    print("=" * 60)

    # --- Step 1: Create the store ---
    dsn = "postgresql://user:password@localhost:5432/selectools_demo"
    store = PostgresCheckpointStore(dsn=dsn)
    print(f"\n1. Created PostgresCheckpointStore")
    print(f"   DSN: {dsn}")
    print(f"   Table: {store.table_name}")
    print(f"   Pool size: {store.pool_size}")
    print(f"   Connected: {store.is_connected}")

    # --- Step 2: Save checkpoints ---
    graph_id = "pipeline-run-001"
    print(f"\n2. Saving checkpoints for graph_id={graph_id!r}")

    states = [
        ("ingest", {"status": "data loaded", "rows": 1000}),
        ("transform", {"status": "data cleaned", "rows": 950}),
        ("publish", {"status": "data published", "rows": 950}),
    ]

    checkpoint_ids = []
    for node_name, data in states:
        state = GraphState(data=data, current_node=node_name)
        step_num = len(checkpoint_ids) + 1
        cid = store.save(graph_id, state, step_num)
        checkpoint_ids.append(cid)
        print(f"   Saved step {step_num} ({node_name}): checkpoint_id={cid[:12]}...")

    # --- Step 3: List checkpoints ---
    print(f"\n3. Listing checkpoints for graph_id={graph_id!r}")
    metas = store.list(graph_id)
    for meta in metas:
        print(
            f"   step={meta.step}, node={meta.node_name}, "
            f"created={meta.created_at.strftime('%H:%M:%S')}"
        )

    # --- Step 4: Load a specific checkpoint ---
    mid_id = checkpoint_ids[1]  # the "transform" checkpoint
    print(f"\n4. Loading checkpoint {mid_id[:12]}...")
    loaded_state, loaded_step = store.load(mid_id)
    print(f"   Step: {loaded_step}")
    print(f"   Node: {loaded_state.current_node}")
    print(f"   Data: {loaded_state.data}")

    # --- Step 5: Delete a checkpoint ---
    old_id = checkpoint_ids[0]
    deleted = store.delete(old_id)
    print(f"\n5. Deleted checkpoint {old_id[:12]}...: {deleted}")
    remaining = store.list(graph_id)
    print(f"   Remaining checkpoints: {len(remaining)}")

    # --- Step 6: Compare with other backends ---
    print(f"\n6. Available checkpoint backends:")
    print(f"   - InMemoryCheckpointStore  (dev/test, no persistence)")
    print(f"   - FileCheckpointStore      (single-machine, JSON files)")
    print(f"   - SQLiteCheckpointStore    (single-machine, WAL mode)")
    print(f"   - PostgresCheckpointStore  (multi-process, production)")
    print(f"   All implement the same CheckpointStore protocol:")
    print(f"     save(graph_id, state, step) -> checkpoint_id")
    print(f"     load(checkpoint_id) -> (state, step)")
    print(f"     list(graph_id) -> [CheckpointMetadata]")
    print(f"     delete(checkpoint_id) -> bool")

    # --- Step 7: Production usage pattern ---
    print(f"\n7. Production usage:")
    print(f'   store = PostgresCheckpointStore("postgresql://...") ')
    print(f'   result = graph.run("input", checkpoint_store=store)')
    print(f"   # Checkpoints saved after each node for crash recovery")
    print(f"   # Resume with: graph.resume(checkpoint_id, store)")

    print("\nDone! PostgresCheckpointStore adds multi-process checkpoint durability.")


if __name__ == "__main__":
    main()
