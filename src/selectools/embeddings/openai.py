"""OpenAI embedding provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding models.

    Supports models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.models import OpenAI
        >>>
        >>> embedder = OpenAIEmbeddingProvider(
        ...     model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id
        ... )
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: Model name (default: text-embedding-3-small)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            dimensions: Optional dimensions for truncation (only for v3 models)
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI package required for OpenAI embeddings. " "Install with: pip install openai"
            ) from e

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self._dimension = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        # Use custom dimensions if specified (only for v3 models)
        if self.dimensions is not None:
            return self.dimensions

        # Default dimensions for each model
        if "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-3-large" in self.model:
            return 3072
        elif "ada-002" in self.model:
            return 1536
        else:
            # Default fallback
            logger.warning(
                f"Unknown OpenAI embedding model '{self.model}', " f"assuming dimension 1536"
            )
            return 1536

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        kwargs: Dict[str, Any] = {"input": text, "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)
        return response.data[0].embedding

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

        kwargs: Dict[str, Any] = {"input": texts, "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**kwargs)

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (same as embed_text for OpenAI).

        OpenAI models don't distinguish between document and query embeddings.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.embed_text(query)

    @property
    def dimension(self) -> int:
        """
        Get the embedding vector dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self._dimension


__all__ = ["OpenAIEmbeddingProvider"]
