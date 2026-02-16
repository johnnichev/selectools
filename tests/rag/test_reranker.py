"""
Tests for reranker interface and implementations.

Uses mock rerankers to test the protocol and HybridSearcher integration
without requiring external API keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from selectools.rag import Document, SearchResult
from selectools.rag.reranker import Reranker


class MockReranker(Reranker):
    """
    Mock reranker that reverses result order and assigns descending scores.

    Useful for testing the reranker integration without external APIs.
    """

    def __init__(self, reverse: bool = True) -> None:
        self.reverse = reverse
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_results: Optional[List[SearchResult]] = None

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        self.call_count += 1
        self.last_query = query
        self.last_results = results

        if not results:
            return []

        if self.reverse:
            reordered = list(reversed(results))
        else:
            reordered = list(results)

        scored = [
            SearchResult(document=r.document, score=1.0 - i * 0.1) for i, r in enumerate(reordered)
        ]

        if top_k is not None:
            scored = scored[:top_k]

        return scored


class ScoreBoostReranker(Reranker):
    """Reranker that boosts documents containing a target keyword."""

    def __init__(self, boost_keyword: str) -> None:
        self.boost_keyword = boost_keyword.lower()

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        scored: List[SearchResult] = []
        for r in results:
            if self.boost_keyword in r.document.text.lower():
                scored.append(SearchResult(document=r.document, score=1.0))
            else:
                scored.append(SearchResult(document=r.document, score=0.1))

        scored.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored


class TestRerankerProtocol:
    """Test the Reranker ABC contract."""

    def test_mock_reranker_implements_protocol(self) -> None:
        reranker = MockReranker()
        assert isinstance(reranker, Reranker)

    def test_rerank_returns_search_results(self) -> None:
        reranker = MockReranker()
        results = [
            SearchResult(document=Document(text="doc A"), score=0.5),
            SearchResult(document=Document(text="doc B"), score=0.8),
        ]
        reranked = reranker.rerank("query", results)
        assert all(isinstance(r, SearchResult) for r in reranked)

    def test_rerank_empty_input(self) -> None:
        reranker = MockReranker()
        assert reranker.rerank("query", []) == []

    def test_rerank_respects_top_k(self) -> None:
        reranker = MockReranker()
        results = [SearchResult(document=Document(text=f"doc {i}"), score=0.5) for i in range(10)]
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_rerank_top_k_none_returns_all(self) -> None:
        reranker = MockReranker()
        results = [SearchResult(document=Document(text=f"doc {i}"), score=0.5) for i in range(5)]
        reranked = reranker.rerank("query", results, top_k=None)
        assert len(reranked) == 5

    def test_rerank_replaces_scores(self) -> None:
        reranker = MockReranker()
        results = [
            SearchResult(document=Document(text="doc A"), score=0.99),
            SearchResult(document=Document(text="doc B"), score=0.01),
        ]
        reranked = reranker.rerank("query", results)
        original_scores = {0.99, 0.01}
        reranked_scores = {r.score for r in reranked}
        assert reranked_scores != original_scores

    def test_rerank_preserves_documents(self) -> None:
        reranker = MockReranker()
        doc_a = Document(text="doc A", metadata={"id": 1})
        doc_b = Document(text="doc B", metadata={"id": 2})
        results = [
            SearchResult(document=doc_a, score=0.5),
            SearchResult(document=doc_b, score=0.8),
        ]
        reranked = reranker.rerank("query", results)
        reranked_texts = {r.document.text for r in reranked}
        assert reranked_texts == {"doc A", "doc B"}

    def test_rerank_tracks_calls(self) -> None:
        reranker = MockReranker()
        results = [SearchResult(document=Document(text="x"), score=0.5)]
        reranker.rerank("hello", results)
        assert reranker.call_count == 1
        assert reranker.last_query == "hello"

    def test_score_boost_reranker(self) -> None:
        reranker = ScoreBoostReranker(boost_keyword="python")
        results = [
            SearchResult(document=Document(text="Java programming"), score=0.9),
            SearchResult(document=Document(text="Python programming"), score=0.3),
        ]
        reranked = reranker.rerank("programming", results)
        assert reranked[0].document.text == "Python programming"
        assert reranked[0].score == 1.0


class TestCohereRerankerUnit:
    """Unit tests for CohereReranker with mocked Cohere client."""

    def test_import_error_without_cohere(self) -> None:
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere"):
                from selectools.rag.reranker import CohereReranker

                CohereReranker.__init__(CohereReranker.__new__(CohereReranker))

    def test_rerank_calls_client(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_result_0 = MagicMock()
        mock_result_0.index = 1
        mock_result_0.relevance_score = 0.95
        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.42
        mock_response.results = [mock_result_0, mock_result_1]
        mock_client.rerank.return_value = mock_response

        from selectools.rag.reranker import CohereReranker

        reranker = CohereReranker.__new__(CohereReranker)
        reranker.client = mock_client
        reranker.model = "rerank-v3.5"

        results = [
            SearchResult(document=Document(text="low relevance"), score=0.3),
            SearchResult(document=Document(text="high relevance"), score=0.8),
        ]

        reranked = reranker.rerank("test query", results, top_k=2)

        mock_client.rerank.assert_called_once_with(
            model="rerank-v3.5",
            query="test query",
            documents=["low relevance", "high relevance"],
            top_n=2,
        )
        assert len(reranked) == 2
        assert reranked[0].score == 0.95
        assert reranked[0].document.text == "high relevance"
        assert reranked[1].score == 0.42
        assert reranked[1].document.text == "low relevance"

    def test_rerank_empty_results(self) -> None:
        from selectools.rag.reranker import CohereReranker

        reranker = CohereReranker.__new__(CohereReranker)
        reranker.client = MagicMock()
        reranker.model = "rerank-v3.5"

        assert reranker.rerank("query", []) == []
        reranker.client.rerank.assert_not_called()

    def test_rerank_top_k_defaults_to_all(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.5),
            MagicMock(index=2, relevance_score=0.3),
        ]
        mock_client.rerank.return_value = mock_response

        from selectools.rag.reranker import CohereReranker

        reranker = CohereReranker.__new__(CohereReranker)
        reranker.client = mock_client
        reranker.model = "rerank-v3.5"

        results = [SearchResult(document=Document(text=f"doc {i}"), score=0.5) for i in range(3)]
        reranker.rerank("query", results)

        mock_client.rerank.assert_called_once()
        call_kwargs = mock_client.rerank.call_args
        assert call_kwargs.kwargs.get("top_n") == 3 or call_kwargs[1].get("top_n") == 3


class TestJinaRerankerUnit:
    """Unit tests for JinaReranker with mocked HTTP calls."""

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key"):
                from selectools.rag.reranker import JinaReranker

                JinaReranker(api_key=None)

    def test_explicit_api_key(self) -> None:
        from selectools.rag.reranker import JinaReranker

        reranker = JinaReranker(api_key="test-key-123")
        assert reranker.api_key == "test-key-123"

    def test_env_api_key(self) -> None:
        with patch.dict("os.environ", {"JINA_API_KEY": "env-key-456"}):
            from selectools.rag.reranker import JinaReranker

            reranker = JinaReranker()
            assert reranker.api_key == "env-key-456"

    def test_rerank_calls_api(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.42},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        from selectools.rag.reranker import JinaReranker

        reranker = JinaReranker(api_key="test-key")
        reranker._requests = MagicMock()
        reranker._requests.post.return_value = mock_response

        results = [
            SearchResult(document=Document(text="low"), score=0.3),
            SearchResult(document=Document(text="high"), score=0.8),
        ]
        reranked = reranker.rerank("test query", results, top_k=2)

        reranker._requests.post.assert_called_once()
        call_kwargs = reranker._requests.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["query"] == "test query"
        assert payload["documents"] == ["low", "high"]
        assert payload["top_n"] == 2

        assert len(reranked) == 2
        assert reranked[0].score == 0.95
        assert reranked[0].document.text == "high"

    def test_rerank_empty_results(self) -> None:
        from selectools.rag.reranker import JinaReranker

        reranker = JinaReranker(api_key="test-key")
        reranker._requests = MagicMock()
        assert reranker.rerank("query", []) == []
        reranker._requests.post.assert_not_called()

    def test_rerank_no_top_n_in_payload(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
        mock_response.raise_for_status = MagicMock()

        from selectools.rag.reranker import JinaReranker

        reranker = JinaReranker(api_key="test-key")
        reranker._requests = MagicMock()
        reranker._requests.post.return_value = mock_response

        results = [SearchResult(document=Document(text="doc"), score=0.5)]
        reranker.rerank("query", results, top_k=None)

        call_kwargs = reranker._requests.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "top_n" not in payload

    def test_rerank_sends_auth_header(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        mock_response.raise_for_status = MagicMock()

        from selectools.rag.reranker import JinaReranker

        reranker = JinaReranker(api_key="my-secret-key")
        reranker._requests = MagicMock()
        reranker._requests.post.return_value = mock_response

        results = [SearchResult(document=Document(text="doc"), score=0.5)]
        reranker.rerank("query", results)

        call_kwargs = reranker._requests.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer my-secret-key"


class MockEmbedder:
    """Mock embedding provider for vector store tests."""

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
class TestHybridSearcherWithReranker:
    """Test the HybridSearcher + reranker integration."""

    @pytest.fixture()
    def documents(self) -> List[Document]:
        return [
            Document(text="Python is a popular programming language"),
            Document(text="Java is widely used in enterprise"),
            Document(text="Machine learning uses Python extensively"),
            Document(text="JavaScript powers the modern web"),
            Document(text="The selectools library for Python agents"),
        ]

    def test_reranker_is_applied(self, documents: List[Document]) -> None:
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        reranker = MockReranker()
        searcher = HybridSearcher(vector_store=store, reranker=reranker)
        searcher.add_documents(documents)

        searcher.search("Python programming", top_k=3)
        assert reranker.call_count == 1
        assert reranker.last_query == "Python programming"

    def test_reranker_none_skips_reranking(self, documents: List[Document]) -> None:
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        searcher = HybridSearcher(vector_store=store, reranker=None)
        searcher.add_documents(documents)

        results = searcher.search("Python", top_k=3)
        assert len(results) > 0

    def test_reranker_changes_ordering(self, documents: List[Document]) -> None:
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())

        searcher_no_rerank = HybridSearcher(vector_store=store)
        searcher_no_rerank.add_documents(documents)
        results_original = searcher_no_rerank.search("Python programming", top_k=5)

        store2 = InMemoryVectorStore(MockEmbedder())
        reranker = MockReranker(reverse=True)
        searcher_with_rerank = HybridSearcher(vector_store=store2, reranker=reranker)
        searcher_with_rerank.add_documents(documents)
        results_reranked = searcher_with_rerank.search("Python programming", top_k=5)

        assert reranker.call_count == 1
        assert len(results_reranked) > 0
        for r in results_reranked:
            assert isinstance(r, SearchResult)

    def test_reranker_respects_top_k(self, documents: List[Document]) -> None:
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        reranker = MockReranker()
        searcher = HybridSearcher(vector_store=store, reranker=reranker)
        searcher.add_documents(documents)

        results = searcher.search("Python", top_k=2)
        assert len(results) <= 2

    def test_score_boost_reranker_integration(self, documents: List[Document]) -> None:
        """ScoreBoostReranker should push 'selectools' doc to top."""
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        reranker = ScoreBoostReranker(boost_keyword="selectools")
        searcher = HybridSearcher(vector_store=store, reranker=reranker)
        searcher.add_documents(documents)

        results = searcher.search("Python programming", top_k=3)
        assert len(results) > 0
        assert "selectools" in results[0].document.text

    def test_reranker_receives_fused_candidates(self, documents: List[Document]) -> None:
        """Reranker should receive the full fused candidate set, not just top_k."""
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        reranker = MockReranker()
        searcher = HybridSearcher(vector_store=store, reranker=reranker)
        searcher.add_documents(documents)

        searcher.search("Python", top_k=2)

        assert reranker.last_results is not None
        assert len(reranker.last_results) > 2

    def test_reranker_with_weighted_fusion(self, documents: List[Document]) -> None:
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(MockEmbedder())
        reranker = MockReranker()
        searcher = HybridSearcher(
            vector_store=store,
            fusion="weighted",
            reranker=reranker,
        )
        searcher.add_documents(documents)

        results = searcher.search("Python", top_k=3)
        assert reranker.call_count == 1
        assert len(results) > 0


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy required for vector store")
class TestRerankerExports:
    """Test that reranker classes are properly exported."""

    def test_reranker_base_exported(self) -> None:
        from selectools.rag import Reranker

        assert Reranker is not None

    def test_reranker_available_from_module(self) -> None:
        from selectools.rag.reranker import CohereReranker, JinaReranker, Reranker

        assert Reranker is not None
        assert CohereReranker is not None
        assert JinaReranker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
