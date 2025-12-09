"""Cohere embedding provider implementation."""

from __future__ import annotations

import logging
from typing import List, Optional, cast

from .provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class CohereEmbeddingProvider(EmbeddingProvider):
    """
    Cohere embedding provider.

    Supports models: embed-english-v3.0, embed-multilingual-v3.0, embed-english-light-v3.0

    Example:
        >>> from selectools.embeddings import CohereEmbeddingProvider
        >>> from selectools.models import Cohere
        >>>
        >>> embedder = CohereEmbeddingProvider(
        ...     model=Cohere.Embeddings.EMBED_V3.id
        ... )
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        truncate: str = "END",
    ):
        """
        Initialize Cohere embedding provider.

        Args:
            model: Model name (default: embed-english-v3.0)
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            truncate: Truncation strategy: "NONE", "START", or "END" (default: "END")
        """
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package required for Cohere embeddings. " "Install with: pip install cohere"
            ) from e

        self.client = cohere.Client(api_key=api_key)
        self.model = model
        self.truncate = truncate
        self._dimension = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        if "v3" in self.model:
            return 1024
        elif "v2" in self.model:
            return 4096
        else:
            logger.warning(
                f"Unknown Cohere embedding model '{self.model}', " f"assuming dimension 1024"
            )
            return 1024

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document",
            truncate=self.truncate,
        )
        return cast(List[float], response.embeddings[0])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch (more efficient than individual calls).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
            truncate=self.truncate,
        )
        return cast(List[List[float]], response.embeddings)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (uses 'search_query' input type for optimal search).

        Cohere models distinguish between document and query embeddings.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query",
            truncate=self.truncate,
        )
        return cast(List[float], response.embeddings[0])

    @property
    def dimension(self) -> int:
        """
        Get the embedding vector dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self._dimension


__all__ = ["CohereEmbeddingProvider"]
