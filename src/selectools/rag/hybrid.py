"""
Hybrid search combining vector (semantic) and BM25 (keyword) retrieval.

Fuses results from both retrieval methods using Reciprocal Rank Fusion (RRF)
or a weighted linear combination, providing better recall and precision than
either method alone.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .bm25 import BM25
from .vector_store import Document, SearchResult

if TYPE_CHECKING:
    from .reranker import Reranker
    from .vector_store import VectorStore


class FusionMethod(Enum):
    """Score fusion strategies for combining retrieval results."""

    RRF = "rrf"
    """Reciprocal Rank Fusion -- rank-based, no score normalisation needed."""

    WEIGHTED = "weighted"
    """Weighted linear combination of min-max normalised scores."""


class HybridSearcher:
    """
    Hybrid search combining vector similarity and BM25 keyword retrieval.

    Retrieves candidates from both a ``VectorStore`` (semantic) and a ``BM25``
    index (keyword), then fuses the two ranked lists into a single result set
    using either Reciprocal Rank Fusion (RRF) or weighted linear combination.

    Args:
        vector_store: The vector store for semantic search.
        bm25: Optional pre-built BM25 index. If ``None``, a new BM25 instance
              is created internally.
        vector_weight: Weight for vector/semantic results (default: 0.5).
        keyword_weight: Weight for BM25/keyword results (default: 0.5).
        fusion: Fusion strategy -- ``"rrf"`` (default) or ``"weighted"``.
        rrf_k: RRF constant (default: 60). Only used when ``fusion="rrf"``.
        reranker: Optional ``Reranker`` instance (e.g. ``CohereReranker``,
                  ``JinaReranker``). When provided, fused candidates are
                  re-scored by the reranker before returning final results.

    Example:
        >>> from selectools.rag import Document, VectorStore
        >>> from selectools.rag.hybrid import HybridSearcher
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = VectorStore.create("memory", embedder=embedder)
        >>>
        >>> searcher = HybridSearcher(vector_store=store)
        >>> docs = [
        ...     Document(text="Python is a programming language"),
        ...     Document(text="Machine learning with neural networks"),
        ... ]
        >>> searcher.add_documents(docs)
        >>> results = searcher.search("Python programming", top_k=5)

    Example with reranker:
        >>> from selectools.rag.reranker import CohereReranker
        >>>
        >>> searcher = HybridSearcher(
        ...     vector_store=store,
        ...     reranker=CohereReranker(model="rerank-v3.5"),
        ... )
        >>> results = searcher.search("Python programming", top_k=3)
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        bm25: Optional[BM25] = None,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        fusion: str = "rrf",
        rrf_k: int = 60,
        reranker: Optional["Reranker"] = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25 or BM25()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.fusion = FusionMethod(fusion)
        self.rrf_k = rrf_k
        self.reranker = reranker

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to both the vector store and the BM25 index.

        Args:
            documents: Documents to add.
            embeddings: Optional pre-computed embeddings for the vector store.

        Returns:
            List of document IDs from the vector store.
        """
        ids = self.vector_store.add_documents(documents, embeddings)
        self.bm25.add_documents(documents)
        return ids

    def index_existing_documents(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from documents already in the vector store.

        Use this when the vector store was populated before the
        ``HybridSearcher`` was created.

        Args:
            documents: Documents to index in BM25 (should match vector store contents).
        """
        self.bm25.index_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        vector_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search using both semantic and keyword retrieval, then fuse results.

        Retrieves ``vector_top_k`` candidates from the vector store and
        ``keyword_top_k`` candidates from BM25, then fuses them using the
        configured fusion strategy. When a ``reranker`` is configured, the
        fused candidates are re-scored before the final ``top_k`` cut.

        Args:
            query: Search query string.
            top_k: Number of final results to return (default: 5).
            filter: Optional metadata filter applied to both retrievers.
            vector_top_k: Number of vector candidates to retrieve.
                          Defaults to ``top_k * 2`` for better fusion coverage.
            keyword_top_k: Number of keyword candidates to retrieve.
                           Defaults to ``top_k * 2`` for better fusion coverage.

        Returns:
            List of ``SearchResult`` objects sorted by fused (or reranked)
            score, highest first.
        """
        candidate_k = top_k * 2
        v_top_k = vector_top_k or candidate_k
        k_top_k = keyword_top_k or candidate_k

        vector_results = self._vector_search(query, v_top_k, filter)
        keyword_results = self.bm25.search(query, top_k=k_top_k, filter=filter)

        if self.fusion == FusionMethod.RRF:
            fused = self._fuse_rrf(vector_results, keyword_results)
        else:
            fused = self._fuse_weighted(vector_results, keyword_results)

        if self.reranker is not None:
            return self.reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]

    def clear(self) -> None:
        """Clear both the vector store and BM25 index."""
        self.vector_store.clear()
        self.bm25.clear()

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Run semantic search via the vector store."""
        if self.vector_store.embedder is None:
            raise ValueError("Vector store does not have an embedding provider configured.")

        query_embedding = self.vector_store.embedder.embed_query(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )

    def _fuse_rrf(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).

        RRF score = w_v / (k + rank_v) + w_k / (k + rank_k)

        Documents appearing in only one list receive their single-source
        RRF contribution.
        """
        doc_scores: Dict[int, float] = {}
        doc_map: Dict[int, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            key = id(result.document)
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            doc_map[key] = result

        for rank, result in enumerate(keyword_results):
            matched_key = self._find_matching_key(result.document, doc_map)
            if matched_key is not None:
                key = matched_key
            else:
                key = id(result.document)
                doc_map[key] = result

            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score

        ranked_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)
        return [
            SearchResult(document=doc_map[k].document, score=doc_scores[k]) for k in ranked_keys
        ]

    def _fuse_weighted(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Fuse results using weighted linear combination of normalised scores.

        Scores from each source are min-max normalised to [0, 1] before
        combining with the configured weights.
        """
        v_normalised = self._min_max_normalise(vector_results)
        k_normalised = self._min_max_normalise(keyword_results)

        doc_scores: Dict[int, float] = {}
        doc_map: Dict[int, SearchResult] = {}

        for result, norm_score in zip(vector_results, v_normalised):
            key = id(result.document)
            doc_scores[key] = self.vector_weight * norm_score
            doc_map[key] = result

        for result, norm_score in zip(keyword_results, k_normalised):
            matched_key = self._find_matching_key(result.document, doc_map)
            if matched_key is not None:
                key = matched_key
            else:
                key = id(result.document)
                doc_map[key] = result
                doc_scores[key] = 0.0

            doc_scores[key] = doc_scores.get(key, 0.0) + self.keyword_weight * norm_score

        ranked_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)
        return [
            SearchResult(document=doc_map[k].document, score=doc_scores[k]) for k in ranked_keys
        ]

    @staticmethod
    def _min_max_normalise(results: List[SearchResult]) -> List[float]:
        """Normalise scores to [0, 1] range using min-max scaling."""
        if not results:
            return []

        scores = [r.score for r in results]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s

        if span == 0:
            return [1.0] * len(scores)

        return [(s - min_s) / span for s in scores]

    @staticmethod
    def _find_matching_key(
        document: Document,
        doc_map: Dict[int, SearchResult],
    ) -> Optional[int]:
        """
        Find an existing key in doc_map whose document matches by text content.

        This handles the case where vector and BM25 results reference
        different Document instances that represent the same underlying document.
        """
        for key, result in doc_map.items():
            if result.document.text == document.text:
                return key
        return None


__all__ = ["FusionMethod", "HybridSearcher"]
