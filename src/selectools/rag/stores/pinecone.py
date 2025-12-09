"""Pinecone vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

from ..vector_store import Document, SearchResult, VectorStore


class PineconeVectorStore(VectorStore):
    """
    Pinecone-based vector store wrapper.

    Wraps the Pinecone client for cloud-hosted vector storage with advanced features
    like namespace isolation, metadata filtering, and high-performance search.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import PineconeVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = PineconeVectorStore(
        ...     embedder,
        ...     index_name="my-index",
        ...     namespace="docs"
        ... )
        >>>
        >>> # Add documents (uploaded to Pinecone cloud)
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search with metadata filtering
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(
        ...     query_emb,
        ...     top_k=1,
        ...     filter={"source": {"$eq": "test"}}
        ... )
    """

    embedder: "EmbeddingProvider"

    def __init__(
        self,
        embedder: "EmbeddingProvider",  # noqa: F821
        index_name: str,
        namespace: str = "",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **pinecone_kwargs,
    ):
        """
        Initialize Pinecone vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings
            index_name: Name of the Pinecone index
            namespace: Namespace within the index (default: "" for default namespace)
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
            **pinecone_kwargs: Additional arguments to pass to Pinecone client
        """
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError(
                "pinecone-client package required for Pinecone vector store. "
                "Install with: pip install pinecone-client"
            ) from e

        self.embedder = embedder
        self.index_name = index_name
        self.namespace = namespace

        # Initialize Pinecone client
        if api_key:
            self.client = Pinecone(api_key=api_key, **pinecone_kwargs)
        else:
            # Will use PINECONE_API_KEY env var
            self.client = Pinecone(**pinecone_kwargs)

        # Get index
        self.index = self.client.Index(index_name)

    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        # Generate IDs and prepare vectors
        vectors = []
        ids = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{hash(doc.text)}_{i}"
            ids.append(doc_id)

            # Pinecone format: (id, values, metadata)
            # Store text in metadata since Pinecone doesn't store original documents
            metadata = doc.metadata.copy()
            metadata["text"] = doc.text

            vectors.append((doc_id, embedding, metadata))

        # Upsert to Pinecone (batch operation)
        self.index.upsert(vectors=vectors, namespace=self.namespace)

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (Pinecone filter format)

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        # Query Pinecone
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,
        )

        # Convert to SearchResult objects
        search_results = []
        for match in query_response.matches:
            # Extract text from metadata
            metadata = match.metadata or {}
            text = metadata.pop("text", "")

            doc = Document(text=text, metadata=metadata)
            search_results.append(SearchResult(document=doc, score=match.score))

        return search_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        self.index.delete(ids=ids, namespace=self.namespace)

    def clear(self) -> None:
        """Clear all documents from the namespace."""
        # Delete all vectors in the namespace
        self.index.delete(delete_all=True, namespace=self.namespace)


__all__ = ["PineconeVectorStore"]
