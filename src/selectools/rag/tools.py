"""Pre-built RAG tools for knowledge base search and semantic similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ..tools import tool

if TYPE_CHECKING:
    from .vector_store import VectorStore


class RAGTool:
    """
    Pre-built tool for retrieval-augmented generation (RAG).

    Searches a knowledge base and returns relevant context for the LLM to use
    in generating responses.

    Example:
        >>> from selectools import Agent, OpenAIProvider
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag import VectorStore, RAGTool, DocumentLoader
        >>>
        >>> # Set up vector store
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = VectorStore.create("memory", embedder=embedder)
        >>>
        >>> # Load and add documents
        >>> docs = DocumentLoader.from_directory("./docs")
        >>> store.add_documents(docs)
        >>>
        >>> # Create RAG tool
        >>> rag_tool = RAGTool(vector_store=store, top_k=3)
        >>>
        >>> # Use with agent
        >>> agent = Agent(tools=[rag_tool.search_knowledge_base], provider=OpenAIProvider())
        >>> response = agent.run("What are the main features?")
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        top_k: int = 3,
        score_threshold: float = 0.0,
        include_scores: bool = True,
    ):
        """
        Initialize RAG tool.

        Args:
            vector_store: Vector store containing the knowledge base
            top_k: Number of documents to retrieve (default: 3)
            score_threshold: Minimum similarity score to include (default: 0.0)
            include_scores: Whether to include similarity scores in results (default: True)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.include_scores = include_scores

    @tool(
        description=(
            "Search the knowledge base for relevant information to answer the user's question. "
            "Use this when you need facts, documentation, or context from the knowledge base. "
            "Returns the most relevant passages found."
        )
    )
    def search_knowledge_base(self, query: str) -> str:
        """
        Search for relevant information in the knowledge base.

        Args:
            query: The question or search query

        Returns:
            Relevant information from the knowledge base with sources

        Example:
            >>> result = rag_tool.search_knowledge_base("How do I install the library?")
            >>> print(result)
        """
        # Get query embedding
        if self.vector_store.embedder is None:
            raise ValueError("Vector store does not have an embedding provider configured.")

        query_embedding = self.vector_store.embedder.embed_query(query)

        # Search vector store
        results = self.vector_store.search(query_embedding=query_embedding, top_k=self.top_k)

        # Filter by score threshold
        results = [r for r in results if r.score >= self.score_threshold]

        if not results:
            return (
                "No relevant information found in the knowledge base. "
                "The answer may not be available in the provided documents."
            )

        # Format results
        context_parts = []
        for i, result in enumerate(results):
            # Extract source information
            source = result.document.metadata.get("source", "Unknown")
            filename = result.document.metadata.get("filename", source)
            page = result.document.metadata.get("page")

            # Build source string
            source_str = f"{filename}"
            if page is not None:
                source_str += f" (page {page})"

            # Build context part
            if self.include_scores:
                header = f"[Source {i+1}: {source_str}, Relevance: {result.score:.2f}]"
            else:
                header = f"[Source {i+1}: {source_str}]"

            context_parts.append(f"{header}\n{result.document.text}\n")

        return "\n".join(context_parts)


class SemanticSearchTool:
    """
    Tool for pure semantic search without LLM integration.

    Returns structured search results for applications that need direct access
    to similarity scores and metadata.

    Example:
        >>> search_tool = SemanticSearchTool(vector_store=store)
        >>> results = search_tool.search("machine learning")
        >>> for result in results:
        ...     print(f"{result.document.metadata['source']}: {result.score}")
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Initialize semantic search tool.

        Args:
            vector_store: Vector store to search
            top_k: Number of results to return (default: 5)
            score_threshold: Minimum similarity score (default: 0.0)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(self, query: str, filter: Optional[dict] = None) -> List:
        """
        Search for semantically similar documents.

        Args:
            query: Search query
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects

        Example:
            >>> results = search_tool.search("python", filter={"type": "tutorial"})
        """
        # Get query embedding
        if self.vector_store.embedder is None:
            raise ValueError("Vector store does not have an embedding provider configured.")

        query_embedding = self.vector_store.embedder.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding, top_k=self.top_k, filter=filter
        )

        # Filter by score threshold
        results = [r for r in results if r.score >= self.score_threshold]

        return results

    @tool(
        description=(
            "Perform a semantic search to find similar documents or passages. "
            "Returns structured results with similarity scores and metadata."
        )
    )
    def semantic_search(self, query: str) -> str:
        """
        Perform semantic search and return formatted results.

        Args:
            query: Search query

        Returns:
            Formatted search results

        Example:
            >>> result = search_tool.semantic_search("neural networks")
        """
        results = self.search(query)

        if not results:
            return "No similar documents found."

        # Format results
        output_parts = [f"Found {len(results)} similar documents:\n"]

        for i, result in enumerate(results):
            source = result.document.metadata.get("source", "Unknown")
            filename = result.document.metadata.get("filename", source)

            output_parts.append(
                f"{i+1}. {filename} (similarity: {result.score:.2f})\n"
                f"   {result.document.text[:200]}...\n"
            )

        return "\n".join(output_parts)


__all__ = ["RAGTool", "SemanticSearchTool"]
