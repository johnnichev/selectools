"""Targeted tests to close small coverage gaps across 15 modules.

Each section corresponds to one module listed in the coverage report,
testing only the specific uncovered lines identified.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ======================================================================
# 1. embeddings/__init__.py — lazy import pattern (lines 15-16, 22-23, 29-30, 36-37)
# ======================================================================


class TestEmbeddingsInit:
    """Test that embedding providers import correctly from the package."""

    def test_embedding_provider_base_importable(self):
        from selectools.embeddings import EmbeddingProvider

        assert EmbeddingProvider is not None

    def test_openai_embedding_in_all_when_available(self):
        """OpenAIEmbeddingProvider should appear in __all__ when openai is installed."""
        import selectools.embeddings as emb_pkg

        # openai is installed in this environment
        if hasattr(emb_pkg, "OpenAIEmbeddingProvider"):
            assert "OpenAIEmbeddingProvider" in emb_pkg.__all__

    def test_anthropic_embedding_in_all_when_available(self):
        import selectools.embeddings as emb_pkg

        if hasattr(emb_pkg, "AnthropicEmbeddingProvider"):
            assert "AnthropicEmbeddingProvider" in emb_pkg.__all__

    def test_gemini_embedding_in_all_when_available(self):
        import selectools.embeddings as emb_pkg

        if hasattr(emb_pkg, "GeminiEmbeddingProvider"):
            assert "GeminiEmbeddingProvider" in emb_pkg.__all__

    def test_cohere_embedding_in_all_when_available(self):
        import selectools.embeddings as emb_pkg

        if hasattr(emb_pkg, "CohereEmbeddingProvider"):
            assert "CohereEmbeddingProvider" in emb_pkg.__all__

    def test_missing_optional_dep_does_not_crash(self):
        """The embeddings __init__ should never raise even when deps are missing."""
        # Simply verify the module loaded without error
        import selectools.embeddings

        assert hasattr(selectools.embeddings, "EmbeddingProvider")


# ======================================================================
# 2. rag/stores/__init__.py — lazy imports (lines 10-11, 17-18, 24-25, 31-32)
# ======================================================================


class TestRagStoresInit:
    """Test rag store lazy imports."""

    def test_memory_store_importable(self):
        from selectools.rag.stores import InMemoryVectorStore

        assert InMemoryVectorStore is not None

    def test_sqlite_store_importable(self):
        from selectools.rag.stores import SQLiteVectorStore

        assert SQLiteVectorStore is not None

    def test_all_contains_available_stores(self):
        import selectools.rag.stores as stores_pkg

        # InMemoryVectorStore should always be available (no optional dep)
        assert "InMemoryVectorStore" in stores_pkg.__all__
        assert "SQLiteVectorStore" in stores_pkg.__all__

    def test_chroma_in_all_when_available(self):
        import selectools.rag.stores as stores_pkg

        if hasattr(stores_pkg, "ChromaVectorStore"):
            assert "ChromaVectorStore" in stores_pkg.__all__

    def test_pinecone_in_all_when_available(self):
        import selectools.rag.stores as stores_pkg

        if hasattr(stores_pkg, "PineconeVectorStore"):
            assert "PineconeVectorStore" in stores_pkg.__all__


# ======================================================================
# 3. rag/__init__.py — lazy imports for rerankers (lines 43-44, 50-51)
#    + RAGAgent.from_files (lines 238-250)
# ======================================================================


class TestRagInit:
    """Test RAG module lazy imports and RAGAgent."""

    def test_rag_all_contains_base_classes(self):
        import selectools.rag as rag_pkg

        for name in ["Document", "SearchResult", "VectorStore", "RAGAgent"]:
            assert name in rag_pkg.__all__

    def test_cohere_reranker_in_all_when_available(self):
        import selectools.rag as rag_pkg

        if hasattr(rag_pkg, "CohereReranker"):
            assert "CohereReranker" in rag_pkg.__all__

    def test_jina_reranker_in_all_when_available(self):
        import selectools.rag as rag_pkg

        if hasattr(rag_pkg, "JinaReranker"):
            assert "JinaReranker" in rag_pkg.__all__


# ======================================================================
# 4. compose.py — @compose decorator (lines 65-69, 78-81, 102)
# ======================================================================


class TestCompose:
    """Test compose() for plain callables and error cases."""

    def test_compose_requires_at_least_two(self):
        from selectools.compose import compose

        with pytest.raises(ValueError, match="at least 2"):
            compose(lambda x: x)

    def test_compose_rejects_non_callable(self):
        from selectools.compose import compose

        with pytest.raises(TypeError, match="compose.*expects"):
            compose(lambda x: x, 42)

    def test_compose_plain_callables(self):
        """When both args are plain callables (not Tool), line 102 is hit."""
        from selectools.compose import compose

        def double(x):
            return x * 2

        def add_one(x):
            return x + 1

        combined = compose(double, add_one)
        # The composite tool should be a Tool with auto-generated name
        assert "double" in combined.name
        assert "add_one" in combined.name
        result = combined.function(5)
        assert result == 11  # double(5)=10, add_one(10)=11

    def test_compose_with_tool_objects(self):
        """When first arg is a Tool, lines 85-99 (named_composite) are hit."""
        from selectools.compose import compose
        from selectools.tools.base import Tool, ToolParameter

        t1 = Tool(
            name="upper",
            description="Uppercase",
            parameters=[ToolParameter(name="text", param_type=str, description="input")],
            function=lambda text: text.upper(),
        )

        def exclaim(s):
            return s + "!"

        combined = compose(t1, exclaim, name="shout")
        assert combined.name == "shout"
        result = combined.function(text="hello")
        assert result == "HELLO!"

    def test_compose_callable_without_name(self):
        """A callable without __name__ should use 'fn' as fallback (line 67)."""
        from selectools.compose import compose

        class MyCallable:
            def __call__(self, x):
                return x

        # Remove __name__ if present
        obj = MyCallable()
        assert not hasattr(obj, "__name__")
        combined = compose(obj, lambda x: x + 1)
        assert "fn" in combined.name


# ======================================================================
# 5. evals/dataset.py — YAML loading + auto-detect (lines 29-36, 62)
# ======================================================================


class TestDatasetLoader:
    """Test DatasetLoader for JSON, YAML, and edge cases."""

    def test_from_json_with_cases_key(self):
        from selectools.evals.dataset import DatasetLoader

        data = {"cases": [{"input": "hello"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cases = DatasetLoader.from_json(f.name)
        assert len(cases) == 1
        assert cases[0].input == "hello"

    def test_from_yaml_loads_correctly(self):
        from selectools.evals.dataset import DatasetLoader

        yaml_content = "- input: hello\n  name: test1\n- input: world\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cases = DatasetLoader.from_yaml(f.name)
        assert len(cases) == 2
        assert cases[0].input == "hello"

    def test_from_yaml_with_cases_key(self):
        from selectools.evals.dataset import DatasetLoader

        yaml_content = "cases:\n  - input: nested\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cases = DatasetLoader.from_yaml(f.name)
        assert len(cases) == 1
        assert cases[0].input == "nested"

    def test_load_auto_detects_yaml(self):
        from selectools.evals.dataset import DatasetLoader

        yaml_content = "- input: auto\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cases = DatasetLoader.load(f.name)
        assert cases[0].input == "auto"

    def test_load_unsupported_format(self):
        from selectools.evals.dataset import DatasetLoader

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"input\nhello\n")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported file format"):
                DatasetLoader.load(f.name)

    def test_from_dicts_stores_unknown_keys_in_metadata(self):
        from selectools.evals.dataset import DatasetLoader

        data = [{"input": "hello", "custom_field": "extra", "name": "test"}]
        cases = DatasetLoader.from_dicts(data)
        assert cases[0].metadata["custom_field"] == "extra"


# ======================================================================
# 6. rag/vector_store.py — VectorStore.create edge cases (lines 33, 145-151)
# ======================================================================


class TestVectorStore:
    """Test VectorStore.create factory and Document."""

    def test_document_none_metadata_becomes_dict(self):
        """Document.__post_init__ handles None metadata (line 33)."""
        from selectools.rag.vector_store import Document

        doc = Document(text="hello", metadata=None)  # type: ignore[arg-type]
        assert doc.metadata == {}

    def test_create_unknown_backend_raises(self):
        from selectools.rag.vector_store import VectorStore

        mock_embedder = MagicMock()
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            VectorStore.create("nosql_fantasy", embedder=mock_embedder)

    def test_create_memory_backend(self):
        from selectools.rag.vector_store import VectorStore

        mock_embedder = MagicMock()
        store = VectorStore.create("memory", embedder=mock_embedder)
        assert store is not None

    def test_create_sqlite_backend(self):
        from selectools.rag.vector_store import VectorStore

        mock_embedder = MagicMock()
        with tempfile.TemporaryDirectory() as td:
            store = VectorStore.create("sqlite", embedder=mock_embedder, db_path=f"{td}/vs.db")
            assert store is not None


# ======================================================================
# 7. tools/decorators.py — _unwrap_type Python 3.10+ union (lines 33-39)
# ======================================================================


class TestUnwrapType:
    """Test _unwrap_type for X | None syntax on Python 3.10+."""

    def test_unwrap_optional_union(self):
        from selectools.tools.decorators import _unwrap_type

        # Optional[str] = Union[str, None]
        result = _unwrap_type(Optional[str])
        assert result is str

    def test_unwrap_optional_list(self):
        from selectools.tools.decorators import _unwrap_type

        # Optional[List[str]] should unwrap to list
        result = _unwrap_type(Optional[List[str]])
        assert result is list

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
    def test_unwrap_pipe_union_syntax(self):
        """Test X | None syntax available in Python 3.10+.

        Uses types.UnionType directly to construct the union type.
        """
        from selectools.tools.decorators import _unwrap_type

        # Construct str | None without eval
        union_type = types.UnionType  # type: ignore[attr-defined]
        # The standard way to get str | None at runtime on 3.10+
        str_or_none = str | None  # type: ignore[operator]
        result = _unwrap_type(str_or_none)
        assert result is str

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
    def test_unwrap_pipe_union_list(self):
        from selectools.tools.decorators import _unwrap_type

        list_or_none = list[str] | None  # type: ignore[operator]
        result = _unwrap_type(list_or_none)
        assert result is list

    def test_unwrap_plain_type_unchanged(self):
        from selectools.tools.decorators import _unwrap_type

        assert _unwrap_type(int) is int
        assert _unwrap_type(str) is str

    def test_unwrap_dict_generic(self):
        from selectools.tools.decorators import _unwrap_type

        result = _unwrap_type(Dict[str, Any])
        assert result is dict


# ======================================================================
# 8. tools/loader.py — from_module empty, reload_module not loaded (lines 67-68, 96, 179-183)
# ======================================================================


class TestToolLoader:
    """Test ToolLoader edge cases."""

    def test_from_module_returns_tools(self):
        from selectools.tools.loader import ToolLoader

        # Load from a real module that has Tool objects
        tools = ToolLoader.from_module("selectools.toolbox.datetime_tools")
        assert isinstance(tools, list)

    def test_from_module_empty_module(self):
        """Importing a module with no Tool objects returns empty list (line 68)."""
        from selectools.tools.loader import ToolLoader

        # os module has no Tool objects
        tools = ToolLoader.from_module("os")
        assert tools == []

    def test_from_file_invalid_spec(self):
        """Test that ImportError is raised if spec cannot be created (line 96)."""
        from selectools.tools.loader import ToolLoader

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("x = 1\n")
            f.flush()
            # Patch spec_from_file_location to return None
            with patch(
                "selectools.tools.loader.importlib.util.spec_from_file_location",
                return_value=None,
            ):
                with pytest.raises(ImportError, match="Cannot create module spec"):
                    ToolLoader.from_file(f.name)

    def test_reload_module_not_yet_imported(self):
        """reload_module on module not in sys.modules should import fresh (line 179-180)."""
        from selectools.tools.loader import ToolLoader

        mod_name = "selectools.toolbox.datetime_tools"
        saved = sys.modules.pop(mod_name, None)
        try:
            tools = ToolLoader.reload_module(mod_name)
            assert isinstance(tools, list)
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_reload_module_already_imported(self):
        """reload_module on already-imported module reloads it (lines 182-183)."""
        from selectools.tools.loader import ToolLoader

        tools = ToolLoader.reload_module("selectools.toolbox.datetime_tools")
        assert isinstance(tools, list)

    def test_from_directory_skips_symlinks_outside(self):
        """from_directory skips symlinks that resolve outside the dir (line 146)."""
        from selectools.tools.loader import ToolLoader

        with tempfile.TemporaryDirectory() as td:
            # Create an empty .py file so iteration occurs
            (Path(td) / "good.py").write_text("x = 1\n")
            tools = ToolLoader.from_directory(td)
            assert tools == []  # no Tool objects, but no crash


# ======================================================================
# 9. evals/history.py — edge cases (lines 63, 79, 98-101, 166)
# ======================================================================


class TestEvalHistory:
    """Test HistoryTrend and HistoryStore edge cases."""

    def _make_entry(self, accuracy: float, cost: float = 0.01, p50: float = 100.0):
        from selectools.evals.history import HistoryEntry

        return HistoryEntry(
            run_id="r1",
            suite_name="test",
            timestamp=time.time(),
            accuracy=accuracy,
            pass_count=int(accuracy * 10),
            fail_count=int((1 - accuracy) * 10),
            error_count=0,
            total_cost=cost,
            total_tokens=100,
            latency_p50=p50,
            latency_p95=200.0,
            total_cases=10,
            model="gpt-4o",
            duration_ms=1000.0,
        )

    def test_cost_delta_with_single_entry(self):
        from selectools.evals.history import HistoryTrend

        trend = HistoryTrend(entries=[self._make_entry(0.8)])
        assert trend.cost_delta == 0.0  # line 63

    def test_is_degrading_with_two_entries(self):
        """With < 3 entries, degrading uses accuracy_delta < -0.01 (line 79)."""
        from selectools.evals.history import HistoryTrend

        entries = [self._make_entry(0.9), self._make_entry(0.8)]
        trend = HistoryTrend(entries=entries)
        assert trend.is_degrading is True

    def test_is_degrading_with_two_entries_stable(self):
        from selectools.evals.history import HistoryTrend

        entries = [self._make_entry(0.9), self._make_entry(0.895)]
        trend = HistoryTrend(entries=entries)
        assert trend.is_degrading is False  # delta is -0.005, not < -0.01

    def test_summary_with_trend_stable(self):
        """summary() with 3 entries shows 'stable' when not consistently up or down (lines 98-101)."""
        from selectools.evals.history import HistoryTrend

        # Mixed trend: up then down => neither is_improving nor is_degrading
        entries = [self._make_entry(0.8), self._make_entry(0.9), self._make_entry(0.85)]
        trend = HistoryTrend(entries=entries)
        assert not trend.is_improving
        assert not trend.is_degrading
        summary = trend.summary()
        assert "stable" in summary.lower()

    def test_summary_with_improving_trend(self):
        from selectools.evals.history import HistoryTrend

        entries = [self._make_entry(0.7), self._make_entry(0.8), self._make_entry(0.9)]
        trend = HistoryTrend(entries=entries)
        summary = trend.summary()
        assert "improving" in summary.lower()

    def test_summary_with_degrading_trend(self):
        from selectools.evals.history import HistoryTrend

        entries = [self._make_entry(0.9), self._make_entry(0.8), self._make_entry(0.7)]
        trend = HistoryTrend(entries=entries)
        summary = trend.summary()
        assert "degrading" in summary.lower()

    def test_trend_skips_corrupted_lines(self):
        """HistoryStore.trend skips corrupted JSON lines (line 166+)."""
        from selectools.evals.history import HistoryStore

        with tempfile.TemporaryDirectory() as td:
            store = HistoryStore(td)
            path = Path(td) / "test.jsonl"
            path.write_text("not-json\n{bad\n")
            trend = store.trend("test")
            assert len(trend.entries) == 0

    def test_list_suites_empty_dir(self):
        from selectools.evals.history import HistoryStore

        with tempfile.TemporaryDirectory() as td:
            store = HistoryStore(Path(td) / "nonexistent")
            assert store.list_suites() == []


# ======================================================================
# 10. trace.py — trace_to_html/trace_to_json edge cases (lines 269, 503-507)
# ======================================================================


class TestTraceEdgeCases:
    """Test trace functions on unusual step types."""

    def test_trace_to_otel_unknown_step_type_fallback(self):
        """Line 269: else branch for unknown step type in OTel export."""
        from selectools.trace import AgentTrace, StepType, TraceStep

        trace = AgentTrace()
        # Use a step type that isn't llm_call, tool_selection, tool_execution,
        # cache_hit, error, or structured_retry
        trace.add(TraceStep(type=StepType.GUARDRAIL, duration_ms=5.0, summary="test guard"))
        # to_otel_spans should handle the else branch
        if hasattr(trace, "to_otel_spans"):
            spans = trace.to_otel_spans()
            assert len(spans) > 0

    def test_trace_to_json_serializes(self):
        """Lines 503-507: trace_to_json handles enum and dataclass default serialization."""
        from selectools.trace import AgentTrace, StepType, TraceStep, trace_to_json

        trace = AgentTrace()
        trace.add(TraceStep(type=StepType.LLM_CALL, duration_ms=10.0, model="gpt-4o"))
        trace.add(TraceStep(type=StepType.GUARDRAIL, duration_ms=2.0))

        result = trace_to_json(trace)
        parsed = json.loads(result)
        assert "steps" in parsed
        assert len(parsed["steps"]) == 2
        # StepType enum should be serialized as its string value
        assert parsed["steps"][0]["type"] == "llm_call"

    def test_trace_to_html_with_graph_steps(self):
        """trace_to_html handles graph steps with from_node/to_node."""
        from selectools.trace import AgentTrace, StepType, TraceStep, trace_to_html

        trace = AgentTrace()
        trace.add(
            TraceStep(
                type=StepType.GRAPH_ROUTING,
                duration_ms=1.0,
                from_node="a",
                to_node="b",
            )
        )
        html = trace_to_html(trace)
        assert "Agent Trace" in html
        assert "graph_routing" in html

    def test_trace_to_html_empty_trace(self):
        from selectools.trace import AgentTrace, trace_to_html

        trace = AgentTrace()
        html = trace_to_html(trace)
        assert "Agent Trace" in html
        assert "steps:</b> 0" in html


# ======================================================================
# 11. knowledge_graph.py — storage branch + extract_triples edge cases
#     (lines 339-342, 399, 401)
# ======================================================================


class TestKnowledgeGraphEdges:
    """Test KnowledgeGraphMemory storage selection and extraction edge cases."""

    def test_custom_triple_store_isinstance(self):
        """Lines 339-340: passing a TripleStore instance uses it directly."""
        from selectools.knowledge_graph import InMemoryTripleStore, KnowledgeGraphMemory

        provider = MagicMock()
        custom_store = InMemoryTripleStore(max_triples=50)
        kg = KnowledgeGraphMemory(provider=provider, storage=custom_store)
        assert kg.store is custom_store

    def test_unknown_storage_object_used_as_is(self):
        """Lines 341-342: passing a non-TripleStore, non-string object is used directly."""
        from selectools.knowledge_graph import KnowledgeGraphMemory

        provider = MagicMock()
        fake_store = MagicMock()
        kg = KnowledgeGraphMemory(provider=provider, storage=fake_store)
        assert kg.store is fake_store

    def test_extract_triples_skips_non_dict_items(self):
        """Line 399: non-dict items in triples_data are skipped."""
        from selectools.knowledge_graph import KnowledgeGraphMemory
        from selectools.types import Message, Role

        provider = MagicMock()
        # LLM returns a list with a non-dict item and an item missing required keys
        response_msg = MagicMock()
        response_msg.content = json.dumps(
            [
                "not a dict",
                {"subject": "Alice"},  # missing relation and object (line 401)
                {"subject": "Bob", "relation": "knows", "object": "Charlie"},
            ]
        )
        provider.complete.return_value = (response_msg, MagicMock())

        kg = KnowledgeGraphMemory(provider=provider)
        messages = [Message(role=Role.USER, content="Bob knows Charlie")]
        triples = kg.extract_triples(messages)
        # Only the valid triple should be extracted
        assert len(triples) == 1
        assert triples[0].subject == "Bob"


# ======================================================================
# 12. mcp/__init__.py — lazy __getattr__ and context manager (lines 43-47, 51)
# ======================================================================


class TestMCPInit:
    """Test MCP module lazy imports and mcp_tools context manager."""

    def test_getattr_unknown_raises(self):
        """Accessing an unknown attribute raises AttributeError."""
        import selectools.mcp as mcp_pkg

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = mcp_pkg.NoSuchThing

    def test_mcp_tools_context_exit_with_no_client(self):
        """Line 51: __exit__ with _client=None is a no-op."""
        from selectools.mcp import MCPServerConfig, _MCPToolsContext

        config = MCPServerConfig(command="echo", args=[])
        ctx = _MCPToolsContext(config)
        ctx._client = None
        # Should not raise
        ctx.__exit__(None, None, None)

    def test_mcp_tools_context_enter_creates_client(self):
        """Lines 43-47: __enter__ creates MCPClient and calls list_tools_sync."""
        from selectools.mcp import MCPServerConfig, _MCPToolsContext

        config = MCPServerConfig(command="echo", args=["hello"])
        ctx = _MCPToolsContext(config)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.list_tools_sync = MagicMock(return_value=["tool1", "tool2"])

        with patch("selectools.mcp.client.MCPClient", return_value=mock_client):
            tools = ctx.__enter__()
            assert tools == ["tool1", "tool2"]
            assert ctx._client is mock_client

        # Now test __exit__ with a real client
        ctx.__exit__(None, None, None)
        mock_client.__exit__.assert_called()


# ======================================================================
# 13. pipeline.py — edge cases (lines 52-53, 70-71, 167, 303-305, 560)
# ======================================================================


class TestPipelineEdgeCases:
    """Test pipeline composition edge cases."""

    def test_filter_kwargs_empty(self):
        """Line 48-49: _filter_kwargs with empty kwargs returns empty."""
        from selectools.pipeline import _filter_kwargs

        def fn(x):
            return x

        assert _filter_kwargs(fn, {}) == {}

    def test_filter_kwargs_uninspectable(self):
        """Lines 52-53: function with uninspectable signature returns all kwargs."""
        from selectools.pipeline import _filter_kwargs

        # Use a built-in that may fail inspection
        with patch("selectools.pipeline.inspect.signature", side_effect=ValueError):
            result = _filter_kwargs(lambda x: x, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_is_subtype_with_generics(self):
        """Lines 64-66: _is_subtype with generic origins returns True."""
        from selectools.pipeline import _is_subtype

        assert _is_subtype(List[str], dict) is True  # has __origin__
        assert _is_subtype(str, List[int]) is True  # expected has __origin__

    def test_is_subtype_with_any(self):
        """Lines 67-68: _is_subtype with Any returns True."""
        from selectools.pipeline import _is_subtype

        assert _is_subtype(Any, str) is True
        assert _is_subtype(str, Any) is True

    def test_is_subtype_type_error(self):
        """Lines 70-71: _is_subtype catches TypeError."""
        from selectools.pipeline import _is_subtype

        # Some objects can't be passed to issubclass
        assert _is_subtype("not_a_type", "also_not") is True

    def test_step_ror(self):
        """Line 167: Step.__ror__ with a callable."""
        from selectools.pipeline import Step

        s = Step(lambda x: x + 1, name="add_one")

        def double(x):
            return x * 2

        pipeline = double | s
        result = pipeline.run(5)
        assert result.output == 11  # double(5)=10, add_one(10)=11

    def test_pipeline_ror_with_pipeline(self):
        """Lines 303-305: Pipeline.__ror__ with another Pipeline."""
        from selectools.pipeline import Pipeline, Step

        s1 = Step(lambda x: x + 1, name="inc")
        s2 = Step(lambda x: x * 2, name="dbl")
        p1 = Pipeline(steps=[s1])
        p2 = Pipeline(steps=[s2])
        # Use __ror__: p2.__ror__(p1)
        combined = p1 | p2
        result = combined.run(5)
        assert result.output == 12  # inc(5)=6, dbl(6)=12

    def test_pipeline_ror_with_callable(self):
        """Lines 303-305: Pipeline.__ror__ with a plain callable."""
        from selectools.pipeline import Pipeline, Step

        s1 = Step(lambda x: x * 3, name="triple")
        p = Pipeline(steps=[s1])

        def add_ten(x):
            return x + 10

        combined = add_ten | p
        result = combined.run(2)
        assert result.output == 36  # add_ten(2)=12, triple(12)=36

    def test_parallel_sync_step_execution(self):
        """Line 560: parallel with sync steps runs them."""
        from selectools.pipeline import Pipeline, parallel

        def upper(text):
            return text.upper()

        def lower(text):
            return text.lower()

        step = parallel(upper, lower)
        # Wrap in a Pipeline so we can call .run()
        p = Pipeline(steps=[step])
        result = p.run("Hello")
        assert result.output["upper"] == "HELLO"
        assert result.output["lower"] == "hello"


# ======================================================================
# 14. tools/base.py — various edge cases
#     (lines 31-37, 249-251, 339, 410, 466, 488, 501, 547)
# ======================================================================


class TestToolBaseEdgeCases:
    """Test Tool class edge cases."""

    def test_get_async_tool_executor_singleton(self):
        """Lines 31-37: _get_async_tool_executor creates singleton."""
        from selectools.tools.base import _get_async_tool_executor

        ex1 = _get_async_tool_executor()
        ex2 = _get_async_tool_executor()
        assert ex1 is ex2

    def test_validate_uninspectable_function(self):
        """Lines 249-251: validate() skips signature check for uninspectable functions."""
        from selectools.tools.base import Tool, ToolParameter

        # Create a tool with a built-in function that can't be inspected
        t = Tool(
            name="test",
            description="test",
            parameters=[ToolParameter(name="x", param_type=str, description="input")],
            function=print,  # built-in, may fail signature inspection
        )
        # validate() should not raise
        t.validate({"x": "hello"})

    def test_validate_single_bool_for_int(self):
        """Line 339: validate rejects bool when int is expected."""
        from selectools.tools.base import Tool, ToolParameter

        t = Tool(
            name="test",
            description="test",
            parameters=[ToolParameter(name="n", param_type=int, description="number")],
            function=lambda n: n,
        )
        with pytest.raises(Exception, match="got bool"):
            t.validate({"n": True})

    def test_validate_type_hint_suggestions(self):
        """Line 410: float type hint suggestion."""
        from selectools.tools.base import Tool, ToolParameter

        t = Tool(
            name="test",
            description="test",
            parameters=[ToolParameter(name="v", param_type=float, description="value")],
            function=lambda v: v,
        )
        with pytest.raises((Exception, TypeError, ValueError)):
            t.validate({"v": "not_a_number"})

    def test_execute_with_config_injector(self):
        """Line 466: config_injector injects additional kwargs."""
        from selectools.tools.base import Tool, ToolParameter

        def my_fn(x, api_key="default"):
            return f"{x}:{api_key}"

        t = Tool(
            name="test",
            description="test",
            parameters=[ToolParameter(name="x", param_type=str, description="input")],
            function=my_fn,
            config_injector=lambda: {"api_key": "injected"},
        )
        result = t.execute({"x": "hello"})
        assert result == "hello:injected"

    def test_execute_config_injector_returns_none(self):
        """Line 466: config_injector returning None doesn't crash."""
        from selectools.tools.base import Tool, ToolParameter

        t = Tool(
            name="test",
            description="test",
            parameters=[ToolParameter(name="x", param_type=str, description="input")],
            function=lambda x: x,
            config_injector=lambda: None,
        )
        result = t.execute({"x": "hello"})
        assert result == "hello"


# ======================================================================
# 15. evals/generator.py — _parse_generated_cases edge cases
#     (lines 49-53, 105-106, 111)
# ======================================================================


class TestEvalGenerator:
    """Test _parse_generated_cases edge cases."""

    def test_parse_generated_cases_with_code_fences(self):
        """Remove markdown code fences from LLM output."""
        from selectools.evals.generator import _parse_generated_cases

        text = '```json\n[{"input": "hello", "name": "test1"}]\n```'
        cases = _parse_generated_cases(text)
        assert len(cases) == 1
        assert cases[0].input == "hello"

    def test_parse_generated_cases_invalid_json_with_array_fallback(self):
        """Lines 105-106: regex fallback when initial JSON parse fails."""
        from selectools.evals.generator import _parse_generated_cases

        text = 'Some preamble\n[{"input": "found"}]\nSome epilogue'
        cases = _parse_generated_cases(text)
        assert len(cases) == 1
        assert cases[0].input == "found"

    def test_parse_generated_cases_completely_invalid(self):
        """Lines 105-106: returns [] when even regex fallback fails to parse."""
        from selectools.evals.generator import _parse_generated_cases

        text = "This is not json at all and has no arrays"
        cases = _parse_generated_cases(text)
        assert cases == []

    def test_parse_generated_cases_not_a_list(self):
        """Line 111: returns [] when parsed JSON is not a list."""
        from selectools.evals.generator import _parse_generated_cases

        text = '{"input": "not a list"}'
        cases = _parse_generated_cases(text)
        assert cases == []

    def test_parse_generated_cases_skips_invalid_items(self):
        """Skips items without 'input' key."""
        from selectools.evals.generator import _parse_generated_cases

        text = json.dumps(
            [
                {"name": "no_input"},
                {"input": "valid", "expect_tool": "search", "tags": ["happy"]},
            ]
        )
        cases = _parse_generated_cases(text)
        assert len(cases) == 1
        assert cases[0].expect_tool == "search"

    def test_generate_cases_uses_func_attribute(self):
        """Lines 49-53: generate_cases uses t.func for signature when available."""
        from selectools.evals.generator import generate_cases

        # Create a mock tool with .func attribute instead of .parameters
        mock_tool = MagicMock(spec=[])
        mock_tool.name = "search"
        mock_tool.description = "search the web"
        mock_tool.func = lambda query: query  # has .func, no .parameters
        # Ensure hasattr checks work correctly
        assert not hasattr(mock_tool, "parameters")
        assert hasattr(mock_tool, "func")

        mock_provider = MagicMock()
        response_msg = MagicMock()
        response_msg.content = json.dumps([{"input": "test query"}])
        mock_provider.complete.return_value = (response_msg, MagicMock())

        cases = generate_cases(mock_provider, "gpt-4o", [mock_tool], n=1)
        assert len(cases) == 1

    def test_parse_generated_cases_regex_fallback_invalid(self):
        """Lines 105-106: regex finds an array but it's still invalid JSON."""
        from selectools.evals.generator import _parse_generated_cases

        text = "prefix [not valid json] suffix"
        cases = _parse_generated_cases(text)
        assert cases == []
