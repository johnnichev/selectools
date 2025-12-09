"""Anthropic/Voyage embedding provider implementation."""

from __future__ import annotations

import logging
from typing import List, Optional, cast

from .provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """
    Anthropic embedding provider using Voyage AI models.

    Anthropic partners with Voyage AI for embedding models.
    Supports models: voyage-3, voyage-3-lite

    Example:
        >>> from selectools.embeddings import AnthropicEmbeddingProvider
        >>> from selectools.models import Anthropic
        >>>
        >>> embedder = AnthropicEmbeddingProvider(
        ...     model=Anthropic.Embeddings.VOYAGE_3_LITE.id
        ... )
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model: str = "voyage-3-lite",
        api_key: Optional[str] = None,
        truncate: bool = True,
    ):
        """
        Initialize Anthropic/Voyage embedding provider.

        Args:
            model: Model name (default: voyage-3-lite)
            api_key: Voyage API key (defaults to VOYAGE_API_KEY env var)
            truncate: Whether to truncate inputs longer than context window
        """
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "voyageai package required for Voyage embeddings. "
                "Install with: pip install voyageai"
            ) from e

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.truncate = truncate
        self._dimension = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        if "voyage-3" in self.model:
            return 1024
        elif "voyage-2" in self.model:
            return 1024
        else:
            logger.warning(
                f"Unknown Voyage embedding model '{self.model}', " f"assuming dimension 1024"
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
            input_type="document",
            truncation=self.truncate,
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
            input_type="document",
            truncation=self.truncate,
        )
        return cast(List[List[float]], response.embeddings)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (uses 'query' input type for optimal search).

        Voyage models distinguish between document and query embeddings.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
            truncation=self.truncate,
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


__all__ = ["AnthropicEmbeddingProvider"]
