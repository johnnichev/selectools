"""
Base embedding provider interface for all embedding implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure consistency
    across different backend implementations (OpenAI, Anthropic/Voyage, Gemini, Cohere).
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch (more efficient than individual calls).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (may differ from document embedding for some providers).

        Some embedding models distinguish between document embedding (for storage)
        and query embedding (for search). For providers that don't distinguish,
        this should behave the same as embed_text().

        Args:
            query: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the embedding vector dimension for this model.

        Returns:
            Integer dimension of embedding vectors
        """
        pass


__all__ = ["EmbeddingProvider"]
