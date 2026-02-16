"""
Tests for hybrid search combining vector and BM25 retrieval.
"""

from __future__ import annotations

from typing import Any, List

import pytest

try:
    import numpy  # noqa: F401

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from selectools.rag import Document, SearchResult
from selectools.rag.hybrid import FusionMethod, HybridSearcher


class MockEmbedder:
    """
    Mock embedding provider that produces deterministic embeddings.

    Embeds text by hashing each word and mapping to a vector. Documents
    sharing words will have higher cosine similarity.
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            h = hash(word) % self._dim
            vec[h] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy required for vector store")
class TestHybridSearcher:
    """Test HybridSearcher combining vector + BM25."""

    @pytest.fixture()
    def documents(self) -> List[Document]:
        return [
            Document(
                text="Python is a popular programming language for data science and AI",
                metadata={"topic": "python", "source": "intro.txt"},
            ),
            Document(
                text="Java is widely used in enterprise applications and Android development",
                metadata={"topic": "java", "source": "intro.txt"},
            ),
            Document(
                text="Machine learning algorithms use Python extensively for model training",
                metadata={"topic": "ml", "source": "guide.txt"},
            ),
            Document(
                text="JavaScript powers modern web browsers and Node.js servers",
                metadata={"topic": "js", "source": "web.txt"},
            ),
            Document(
                text="The selectools library provides agent tool calling for Python",
                metadata={"topic": "python", "source": "readme.md"},
            ),
            Document(
                text="Django and Flask are popular Python web frameworks for REST APIs",
                metadata={"topic": "python", "source": "frameworks.txt"},
            ),
        ]

    @pytest.fixture()
    def searcher(self, documents: List[Document]) -> HybridSearcher:
        from selectools.rag.stores.memory import InMemoryVectorStore

        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)
        searcher = HybridSearcher(vector_store=store)
        searcher.add_documents(documents)
        return searcher

    def test_search_returns_results(self, searcher: HybridSearcher) -> None:
        results = searcher.search("Python programming", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_results_are_search_result_type(self, searcher: HybridSearcher) -> None:
        results = searcher.search("Python")
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.document, Document)
            assert isinstance(r.score, float)

    def test_search_results_sorted_by_score(self, searcher: HybridSearcher) -> None:
        results = searcher.search("Python programming", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, searcher: HybridSearcher) -> None:
        results = searcher.search("Python", top_k=2)
        assert len(results) <= 2

    def test_exact_keyword_boosted(self, searcher: HybridSearcher) -> None:
        """BM25 should boost documents with the exact term 'selectools'."""
        results = searcher.search("selectools library", top_k=3)
        assert len(results) > 0
        texts = [r.document.text for r in results]
        assert any("selectools" in t for t in texts)

    def test_rrf_fusion_default(self) -> None:
        """Default fusion should be RRF."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store)
        assert searcher.fusion == FusionMethod.RRF

    def test_weighted_fusion(self, documents: List[Document]) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(
            vector_store=store,
            fusion="weighted",
            vector_weight=0.7,
            keyword_weight=0.3,
        )
        searcher.add_documents(documents)

        results = searcher.search("Python programming", top_k=3)
        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_vector_only_weight(self, documents: List[Document]) -> None:
        """With keyword_weight=0, should behave like pure vector search."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(
            vector_store=store,
            vector_weight=1.0,
            keyword_weight=0.0,
        )
        searcher.add_documents(documents)

        results = searcher.search("Python", top_k=3)
        assert len(results) > 0

    def test_keyword_only_weight(self, documents: List[Document]) -> None:
        """With vector_weight=0, should behave like pure BM25 search."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(
            vector_store=store,
            vector_weight=0.0,
            keyword_weight=1.0,
        )
        searcher.add_documents(documents)

        results = searcher.search("selectools", top_k=3)
        assert len(results) >= 1
        assert "selectools" in results[0].document.text

    def test_search_with_metadata_filter(self, searcher: HybridSearcher) -> None:
        results = searcher.search("Python", top_k=5, filter={"topic": "python"})
        for r in results:
            assert r.document.metadata.get("topic") == "python"

    def test_empty_query(self, searcher: HybridSearcher) -> None:
        results = searcher.search("", top_k=5)
        assert isinstance(results, list)

    def test_clear(self, searcher: HybridSearcher) -> None:
        searcher.clear()
        results = searcher.search("Python", top_k=5)
        assert len(results) == 0

    def test_index_existing_documents(self) -> None:
        """Test building BM25 index from pre-existing vector store documents."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        docs = [
            Document(text="Python programming"),
            Document(text="Java development"),
        ]
        store.add_documents(docs)

        searcher = HybridSearcher(vector_store=store)
        searcher.index_existing_documents(docs)

        results = searcher.search("Python", top_k=2)
        assert len(results) > 0

    def test_custom_rrf_k(self, documents: List[Document]) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store, rrf_k=10)
        searcher.add_documents(documents)

        results = searcher.search("Python", top_k=3)
        assert len(results) > 0

    def test_custom_candidate_counts(self, searcher: HybridSearcher) -> None:
        results = searcher.search(
            "Python programming",
            top_k=2,
            vector_top_k=3,
            keyword_top_k=3,
        )
        assert len(results) <= 2

    def test_add_documents_returns_ids(self, documents: List[Document]) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store)
        ids = searcher.add_documents(documents)
        assert len(ids) == len(documents)
        assert all(isinstance(i, str) for i in ids)

    def test_no_embedder_raises(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore.__new__(InMemoryVectorStore)
        store.embedder = None
        store.documents = []
        store.embeddings = None
        store.ids = []
        store._id_counter = 0

        searcher = HybridSearcher(vector_store=store)
        with pytest.raises(ValueError, match="embedding provider"):
            searcher.search("test")


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy required for vector store")
class TestFusionMethods:
    """Test specific fusion algorithm behavior."""

    def test_rrf_scores_are_positive(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store, fusion="rrf")
        searcher.add_documents(
            [
                Document(text="Python programming"),
                Document(text="Java development"),
            ]
        )

        results = searcher.search("Python", top_k=2)
        for r in results:
            assert r.score > 0

    def test_weighted_scores_bounded(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(
            vector_store=store,
            fusion="weighted",
            vector_weight=0.5,
            keyword_weight=0.5,
        )
        searcher.add_documents(
            [
                Document(text="Python programming language"),
                Document(text="Java programming language"),
                Document(text="Rust systems programming"),
            ]
        )

        results = searcher.search("programming language", top_k=3)
        for r in results:
            assert 0 <= r.score <= 1.0

    def test_rrf_deduplicates_documents(self) -> None:
        """Same document from both sources should appear once with combined score."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store, fusion="rrf")
        searcher.add_documents(
            [
                Document(text="Python programming language guide"),
            ]
        )

        results = searcher.search("Python programming", top_k=5)
        assert len(results) == 1

    def test_invalid_fusion_method_raises(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        with pytest.raises(ValueError):
            HybridSearcher(vector_store=store, fusion="invalid")

    def test_min_max_normalise_single_result(self) -> None:
        normalised = HybridSearcher._min_max_normalise(
            [
                SearchResult(document=Document(text="only one"), score=0.5),
            ]
        )
        assert normalised == [1.0]

    def test_min_max_normalise_empty(self) -> None:
        normalised = HybridSearcher._min_max_normalise([])
        assert normalised == []

    def test_min_max_normalise_range(self) -> None:
        results = [
            SearchResult(document=Document(text="a"), score=1.0),
            SearchResult(document=Document(text="b"), score=5.0),
            SearchResult(document=Document(text="c"), score=3.0),
        ]
        normalised = HybridSearcher._min_max_normalise(results)
        assert normalised[0] == pytest.approx(0.0)
        assert normalised[1] == pytest.approx(1.0)
        assert normalised[2] == pytest.approx(0.5)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy required for vector store")
class TestHybridSearchTool:
    """Test the HybridSearchTool for agent integration."""

    @pytest.fixture()
    def hybrid_tool(self) -> Any:
        from selectools.rag.stores.memory import InMemoryVectorStore
        from selectools.rag.tools import HybridSearchTool

        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)
        searcher = HybridSearcher(vector_store=store)
        searcher.add_documents(
            [
                Document(
                    text="Selectools is a Python library for AI agents",
                    metadata={"source": "readme.md"},
                ),
                Document(
                    text="Install with pip install selectools",
                    metadata={"source": "install.md", "page": 1},
                ),
                Document(
                    text="Java Spring Boot enterprise framework",
                    metadata={"source": "other.md"},
                ),
            ]
        )
        return HybridSearchTool(searcher=searcher, top_k=3)

    def test_tool_has_search_method(self, hybrid_tool: Any) -> None:
        assert hasattr(hybrid_tool, "search_knowledge_base")

    def test_tool_is_decorated(self, hybrid_tool: Any) -> None:
        assert hybrid_tool.search_knowledge_base.name == "search_knowledge_base"

    def test_tool_search_returns_string(self, hybrid_tool: Any) -> None:
        result = hybrid_tool.search_knowledge_base.function(hybrid_tool, "selectools library")
        assert isinstance(result, str)
        assert "selectools" in result.lower()

    def test_tool_search_includes_source(self, hybrid_tool: Any) -> None:
        result = hybrid_tool.search_knowledge_base.function(hybrid_tool, "install selectools")
        assert "install.md" in result

    def test_tool_search_includes_page(self, hybrid_tool: Any) -> None:
        result = hybrid_tool.search_knowledge_base.function(hybrid_tool, "install selectools")
        assert "page 1" in result

    def test_tool_search_no_results(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore
        from selectools.rag.tools import HybridSearchTool

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store)
        searcher.add_documents([Document(text="Python programming")])

        ht = HybridSearchTool(searcher=searcher, score_threshold=999.0)
        result = ht.search_knowledge_base.function(ht, "Python")
        assert "No relevant information found" in result

    def test_tool_structured_search(self, hybrid_tool: Any) -> None:
        results = hybrid_tool.search("selectools")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_tool_scores_hidden(self) -> None:
        from selectools.rag.stores.memory import InMemoryVectorStore
        from selectools.rag.tools import HybridSearchTool

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store)
        searcher.add_documents([Document(text="Python programming")])

        ht = HybridSearchTool(searcher=searcher, include_scores=False)
        result = ht.search_knowledge_base.function(ht, "Python")
        assert "Relevance:" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
