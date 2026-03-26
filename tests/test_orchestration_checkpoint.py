"""Tests for checkpoint backends."""

from __future__ import annotations

import os
import tempfile
import threading

import pytest

from selectools.orchestration.checkpoint import (
    CheckpointMetadata,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from selectools.orchestration.state import GraphState


def _make_state(content: str = "test", node: str = "node_a", data: dict = None) -> GraphState:
    state = GraphState.from_prompt(content)
    state.current_node = node
    if data:
        state.data.update(data)
    return state


class TestInMemoryCheckpointStore:
    def test_save_returns_checkpoint_id(self):
        store = InMemoryCheckpointStore()
        state = _make_state()
        cid = store.save("graph_1", state, step=1)
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_load_round_trip(self):
        store = InMemoryCheckpointStore()
        state = _make_state(data={"key": "value"})
        cid = store.save("g1", state, step=3)
        loaded_state, loaded_step = store.load(cid)
        assert loaded_step == 3
        assert loaded_state.data["key"] == "value"

    def test_load_missing_raises_valueerror(self):
        store = InMemoryCheckpointStore()
        with pytest.raises(ValueError, match="not found"):
            store.load("nonexistent_id")

    def test_list_returns_metadata(self):
        store = InMemoryCheckpointStore()
        state = _make_state()
        store.save("graph_1", state, 1)
        store.save("graph_1", state, 2)
        store.save("graph_2", state, 1)
        metas = store.list("graph_1")
        assert len(metas) == 2
        assert all(m.graph_id == "graph_1" for m in metas)

    def test_delete_returns_true(self):
        store = InMemoryCheckpointStore()
        state = _make_state()
        cid = store.save("g1", state, 1)
        result = store.delete(cid)
        assert result is True
        with pytest.raises(ValueError):
            store.load(cid)

    def test_delete_nonexistent_returns_false(self):
        store = InMemoryCheckpointStore()
        assert store.delete("no_such_id") is False

    def test_interrupt_responses_survive_round_trip(self):
        store = InMemoryCheckpointStore()
        state = _make_state()
        state._interrupt_responses["node_a_0"] = "approved"
        cid = store.save("g1", state, 1)
        loaded_state, _ = store.load(cid)
        assert loaded_state._interrupt_responses.get("node_a_0") == "approved"

    def test_interrupted_flag_in_metadata(self):
        store = InMemoryCheckpointStore()
        state = _make_state()
        state.metadata["__pending_interrupt_key__"] = "node_a_0"
        cid = store.save("g1", state, 1)
        metas = store.list("g1")
        assert metas[0].interrupted is True

    def test_thread_safety(self):
        store = InMemoryCheckpointStore()
        errors = []

        def worker(i):
            try:
                state = _make_state(data={"i": i})
                cid = store.save("g", state, i)
                loaded, step = store.load(cid)
                assert loaded.data["i"] == i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_list_empty_graph_returns_empty(self):
        store = InMemoryCheckpointStore()
        metas = store.list("no_such_graph")
        assert metas == []

    def test_list_sorted_by_created_at(self):
        """Checkpoints are returned in ascending creation order."""
        store = InMemoryCheckpointStore()
        state = _make_state()
        for i in range(3):
            store.save("g1", state, i)
        metas = store.list("g1")
        steps = [m.step for m in metas]
        assert steps == sorted(steps)

    def test_checkpoint_metadata_fields(self):
        store = InMemoryCheckpointStore()
        state = _make_state(node="my_node")
        cid = store.save("my_graph", state, step=7)
        metas = store.list("my_graph")
        assert len(metas) == 1
        meta = metas[0]
        assert meta.checkpoint_id == cid
        assert meta.graph_id == "my_graph"
        assert meta.step == 7
        assert meta.node_name == "my_node"
        assert meta.interrupted is False


class TestFileCheckpointStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state(data={"x": 42})
            cid = store.save("g1", state, 5)
            loaded, step = store.load(cid)
            assert step == 5
            assert loaded.data["x"] == 42

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            store.save("g1", state, 1)
            store.save("g1", state, 2)
            metas = store.list("g1")
            assert len(metas) == 2

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            cid = store.save("g1", state, 1)
            assert store.delete(cid) is True
            with pytest.raises(ValueError):
                store.load(cid)

    def test_files_created_on_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            cid = store.save("my_graph", state, 1)
            graph_dir = os.path.join(tmpdir, "my_graph")
            assert os.path.isdir(graph_dir)
            files = os.listdir(graph_dir)
            assert any(f.endswith(".json") for f in files)

    def test_interrupt_responses_in_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            state._interrupt_responses["n_0"] = "response"
            cid = store.save("g", state, 1)
            loaded, _ = store.load(cid)
            assert loaded._interrupt_responses.get("n_0") == "response"

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            assert store.delete("no_such_id") is False

    def test_list_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            metas = store.list("no_such_graph")
            assert metas == []

    def test_interrupted_metadata_in_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            state.metadata["__pending_interrupt_key__"] = "node_0"
            cid = store.save("g1", state, 1)
            metas = store.list("g1")
            assert metas[0].interrupted is True

    def test_multiple_graphs_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            state = _make_state()
            store.save("g1", state, 1)
            store.save("g2", state, 1)
            assert len(store.list("g1")) == 1
            assert len(store.list("g2")) == 1


class TestSQLiteCheckpointStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state(data={"hello": "world"})
            cid = store.save("g1", state, 2)
            loaded, step = store.load(cid)
            assert step == 2
            assert loaded.data["hello"] == "world"

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state()
            store.save("g1", state, 1)
            store.save("g1", state, 2)
            store.save("g2", state, 1)
            metas = store.list("g1")
            assert len(metas) == 2
            assert all(m.graph_id == "g1" for m in metas)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state()
            cid = store.save("g1", state, 1)
            assert store.delete(cid) is True
            with pytest.raises(ValueError):
                store.load(cid)

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            with pytest.raises(ValueError):
                store.load("no_such_id")

    def test_interrupted_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state()
            state.metadata["__pending_interrupt_key__"] = "node_0"
            cid = store.save("g1", state, 1)
            metas = store.list("g1")
            assert metas[0].interrupted is True

    def test_interrupt_responses_survive_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state()
            state._interrupt_responses["node_0"] = "yes"
            cid = store.save("g1", state, 1)
            loaded, _ = store.load(cid)
            assert loaded._interrupt_responses.get("node_0") == "yes"

    def test_sorted_by_created_at(self):
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state()
            for i in range(3):
                store.save("g1", state, i)
                time.sleep(0.01)  # ensure distinct timestamps
            metas = store.list("g1")
            steps = [m.step for m in metas]
            assert steps == sorted(steps)

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            assert store.delete("no_such_id") is False

    def test_list_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            metas = store.list("no_such_graph")
            assert metas == []

    def test_checkpoint_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "checkpoints.db")
            store = SQLiteCheckpointStore(db)
            state = _make_state(node="exec_node")
            cid = store.save("my_graph", state, step=5)
            metas = store.list("my_graph")
            assert len(metas) == 1
            meta = metas[0]
            assert meta.checkpoint_id == cid
            assert meta.graph_id == "my_graph"
            assert meta.step == 5
            assert meta.node_name == "exec_node"
            assert meta.interrupted is False
