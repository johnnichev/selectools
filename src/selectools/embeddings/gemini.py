"""Google Gemini embedding provider implementation."""

from __future__ import annotations

import logging
from typing import List, Optional

from .provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider.

    Supports models: text-embedding-001, text-embedding-004

    Example:
        >>> from selectools.embeddings import GeminiEmbeddingProvider
        >>> from selectools.models import Gemini
        >>>
        >>> embedder = GeminiEmbeddingProvider(
        ...     model=Gemini.Embeddings.EMBEDDING_004.id
        ... )
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: Optional[str] = None,
        task_type: str = "retrieval_document",
    ):
        """
        Initialize Gemini embedding provider.

        Args:
            model: Model name (default: text-embedding-004)
            api_key: Google API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            task_type: Task type for embeddings (retrieval_document, retrieval_query,
                      semantic_similarity, classification, clustering)
        """
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package required for Gemini embeddings. "
                "Install with: pip install google-genai"
            ) from e

        # Initialize client with API key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Will use GEMINI_API_KEY or GOOGLE_API_KEY env vars
            self.client = genai.Client()

        self.model = model
        self.task_type = task_type
        self._dimension = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        if "embedding-004" in self.model or "embedding-001" in self.model:
            return 768
        else:
            logger.warning(
                f"Unknown Gemini embedding model '{self.model}', " f"assuming dimension 768"
            )
            return 768

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        from google.genai import types

        config = types.EmbedContentConfig(task_type=self.task_type)
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("No embedding returned")
        return response.embeddings[0].values

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

        from google.genai import types

        config = types.EmbedContentConfig(task_type=self.task_type)
        embeddings = []
        # Gemini API handles batching internally
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=config,
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise ValueError("No embedding returned")
            embeddings.append(response.embeddings[0].values)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (uses 'retrieval_query' task type for optimal search).

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        from google.genai import types

        config = types.EmbedContentConfig(task_type="retrieval_query")
        response = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=config,
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("No embedding returned")
        return response.embeddings[0].values

    @property
    def dimension(self) -> int:
        """
        Get the embedding vector dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self._dimension


__all__ = ["GeminiEmbeddingProvider"]
