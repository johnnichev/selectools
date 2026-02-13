"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

from ..vector_store import Document, SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store wrapper.

    Wraps the ChromaDB client for persistent vector storage with advanced features
    like metadata filtering, automatic batching, and hybrid search capabilities.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import ChromaVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = ChromaVectorStore(
        ...     embedder,
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_db"
        ... )
        >>>
        >>> # Add documents (persisted to disk)
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search with metadata filtering
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(
        ...     query_emb,
        ...     top_k=1,
        ...     filter={"source": "test"}
        ... )
    """

    embedder: "EmbeddingProvider"

    def __init__(
        self,
        embedder: "EmbeddingProvider",  # noqa: F821
        collection_name: str = "selectools_docs",
        persist_directory: Optional[str] = None,
        **chroma_kwargs: Any,
    ) -> None:
        """
        Initialize Chroma vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database (None for in-memory)
            **chroma_kwargs: Additional arguments to pass to ChromaDB client
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb package required for Chroma vector store. "
                "Install with: pip install chromadb"
            ) from e

        self.embedder = embedder
        self.collection_name = collection_name

        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory, **chroma_kwargs)
        else:
            self.client = chromadb.Client(**chroma_kwargs)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

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

        # Generate IDs
        ids = [f"doc_{hash(doc.text)}_{i}" for i, doc in enumerate(documents)]

        # Extract texts and metadata
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add to Chroma collection
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)  # type: ignore

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
            filter: Optional metadata filter (Chroma where clause)

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        # Convert filter to Chroma where clause format
        where = None
        if filter:
            # Simple equality filters
            # For more complex filters, users can pass Chroma-formatted where clauses
            where = filter

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if (
            results["ids"]
            and len(results["ids"][0]) > 0
            and results["documents"]
            and results["metadatas"]
            and results["distances"]
        ):
            for i in range(len(results["ids"][0])):
                doc = Document(
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {},  # type: ignore
                )

                # Chroma returns distances, convert to similarity scores
                # For cosine similarity, distance = 1 - similarity
                # So similarity = 1 - distance
                distance = results["distances"][0][i]
                score = 1.0 - distance

                search_results.append(SearchResult(document=doc, score=score))

        return search_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        self.collection.delete(ids=ids)

    def clear(self) -> None:
        """Clear all documents from the store."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


__all__ = ["ChromaVectorStore"]
