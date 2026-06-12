"""Google Gemini embedding provider implementation."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..stability import beta
from .provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@beta
class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider.

    Supports models: gemini-embedding-001, gemini-embedding-2

    Requests do not set ``output_dimensionality``, so vectors come back at
    each model's default dimensionality (3072 for gemini-embedding-001 and
    gemini-embedding-2, which support MRL truncation but are not truncated
    here). See ``_get_model_dimension`` for the full dimension/MRL story.

    Example:
        >>> from selectools.embeddings import GeminiEmbeddingProvider
        >>> from selectools.models import Gemini
        >>>
        >>> embedder = GeminiEmbeddingProvider(
        ...     model=Gemini.Embeddings.EMBEDDING_001.id
        ... )
        >>> embedding = embedder.embed_text("Hello, world!")
        >>> print(f"Dimension: {len(embedding)}")
    """

    def __init__(
        self,
        # Default changed from text-embedding-004 (retired by Google) to its
        # documented replacement gemini-embedding-001.
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        task_type: str = "retrieval_document",
    ):
        """
        Initialize Gemini embedding provider.

        Args:
            model: Model name (default: gemini-embedding-001)
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
        """Get the embedding dimension for the model.

        The declared dimension must match what the API actually returns. This
        provider never passes ``output_dimensionality`` in its requests (see
        ``embed_text``/``embed_texts``/``embed_query``), so the API returns
        each model's default, full-size vector:

        - ``gemini-embedding-001`` and ``gemini-embedding-2``: 3072 by
          default. Both are trained with Matryoshka Representation Learning
          (MRL), so vectors *can* be truncated server-side (Google recommends
          768, 1536, or 3072) by passing ``output_dimensionality`` — but this
          provider does not request truncation, and existing vector store
          indexes built with it hold 3072-dim vectors. Note that
          ``gemini-embedding-001`` only normalizes the full 3072-dim output;
          truncated outputs would need manual re-normalization.
        - ``text-embedding-004`` (retired by Google): natively 768.

        Source: https://ai.google.dev/gemini-api/docs/embeddings
        """
        if "embedding-004" in self.model:
            return 768
        if "embedding-001" in self.model or "embedding-2" in self.model:
            return 3072
        logger.warning(f"Unknown Gemini embedding model '{self.model}', assuming dimension 3072")
        return 3072

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
        # TODO: Use batch embed_content API for better performance with large text lists.
        # Currently makes one API call per text due to Gemini SDK limitations.
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


__stability__ = "beta"

__all__ = ["GeminiEmbeddingProvider"]
