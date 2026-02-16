"""
Reranker interface and implementations for improving search relevance.

Rerankers re-score a candidate list produced by an initial retrieval step
(vector search, BM25, or hybrid) using a cross-encoder model, yielding
significantly better precision than bi-encoder similarity alone.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """
    Abstract base class for reranker implementations.

    A reranker takes a query and a list of candidate ``SearchResult`` objects,
    re-scores them using a cross-encoder model, and returns the top results
    sorted by relevance.

    All rerankers must implement this interface so they can be used
    interchangeably in ``HybridSearcher`` or standalone pipelines.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-score and re-order search results for the given query.

        Args:
            query: The search query.
            results: Candidate results from an initial retrieval step.
            top_k: Maximum number of results to return. If ``None``,
                   all results are returned (re-ordered).

        Returns:
            List of ``SearchResult`` objects sorted by the reranker's
            relevance score (highest first). The ``score`` field is
            replaced with the reranker's relevance score.
        """
        pass


class CohereReranker(Reranker):
    """
    Reranker using the Cohere Rerank API.

    Requires the ``cohere`` package (``pip install cohere`` or
    ``pip install selectools[rag]``).

    Args:
        model: Cohere rerank model (default: ``"rerank-v3.5"``).
        api_key: Cohere API key. Defaults to the ``COHERE_API_KEY``
                 environment variable.

    Example:
        >>> from selectools.rag.reranker import CohereReranker
        >>> from selectools.rag import SearchResult, Document
        >>>
        >>> reranker = CohereReranker()
        >>> candidates = [
        ...     SearchResult(document=Document(text="Python is great"), score=0.8),
        ...     SearchResult(document=Document(text="Java is popular"), score=0.7),
        ... ]
        >>> reranked = reranker.rerank("best programming language", candidates, top_k=1)
    """

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package required for CohereReranker. "
                "Install with: pip install cohere  (or pip install selectools[rag])"
            ) from e

        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-score results using the Cohere Rerank API.

        Args:
            query: The search query.
            results: Candidate search results.
            top_k: Max results to return (default: all).

        Returns:
            Re-scored ``SearchResult`` list sorted by relevance.
        """
        if not results:
            return []

        documents = [r.document.text for r in results]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or len(results),
        )

        reranked: List[SearchResult] = []
        for item in response.results:
            original = results[item.index]
            reranked.append(
                SearchResult(
                    document=original.document,
                    score=item.relevance_score,
                )
            )

        return reranked


class JinaReranker(Reranker):
    """
    Reranker using the Jina AI Rerank API.

    Calls the Jina ``POST /v1/rerank`` endpoint via HTTP. Requires only
    the ``requests`` library (no additional SDK needed).

    Args:
        model: Jina rerank model (default: ``"jina-reranker-v2-base-multilingual"``).
        api_key: Jina API key. Defaults to the ``JINA_API_KEY``
                 environment variable.
        api_url: API endpoint (default: ``"https://api.jina.ai/v1/rerank"``).

    Example:
        >>> from selectools.rag.reranker import JinaReranker
        >>> from selectools.rag import SearchResult, Document
        >>>
        >>> reranker = JinaReranker()
        >>> candidates = [
        ...     SearchResult(document=Document(text="Python is great"), score=0.8),
        ...     SearchResult(document=Document(text="Java is popular"), score=0.7),
        ... ]
        >>> reranked = reranker.rerank("best programming language", candidates, top_k=1)
    """

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        api_key: Optional[str] = None,
        api_url: str = "https://api.jina.ai/v1/rerank",
    ) -> None:
        try:
            import requests  # type: ignore[import-untyped]

            self._requests = requests
        except ImportError as e:
            raise ImportError(
                "requests package required for JinaReranker. " "Install with: pip install requests"
            ) from e

        if api_key is None:
            import os

            api_key = os.environ.get("JINA_API_KEY")
            if not api_key:
                raise ValueError(
                    "Jina API key is required. Set the JINA_API_KEY environment variable "
                    "or pass api_key='...' to JinaReranker()."
                )

        self.model = model
        self.api_key = api_key
        self.api_url = api_url

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Re-score results using the Jina Rerank API.

        Args:
            query: The search query.
            results: Candidate search results.
            top_k: Max results to return (default: all).

        Returns:
            Re-scored ``SearchResult`` list sorted by relevance.
        """
        if not results:
            return []

        documents = [r.document.text for r in results]

        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        response = self._requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        reranked: List[SearchResult] = []
        for item in data["results"]:
            original = results[item["index"]]
            reranked.append(
                SearchResult(
                    document=original.document,
                    score=float(item["relevance_score"]),
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked


__all__ = ["Reranker", "CohereReranker", "JinaReranker"]
