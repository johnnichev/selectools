"""RAG (Retrieval-Augmented Generation) module for document-based Q&A."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .bm25 import BM25
from .chunking import ContextualChunker, RecursiveTextSplitter, SemanticChunker, TextSplitter
from .hybrid import FusionMethod, HybridSearcher
from .loaders import DocumentLoader
from .reranker import Reranker
from .tools import HybridSearchTool, RAGTool, SemanticSearchTool
from .vector_store import Document, SearchResult, VectorStore

if TYPE_CHECKING:
    from ..agent import Agent, AgentConfig
    from ..providers.base import Provider

__all__ = [
    "Document",
    "SearchResult",
    "VectorStore",
    "DocumentLoader",
    "TextSplitter",
    "RecursiveTextSplitter",
    "SemanticChunker",
    "ContextualChunker",
    "RAGTool",
    "SemanticSearchTool",
    "HybridSearchTool",
    "HybridSearcher",
    "FusionMethod",
    "BM25",
    "Reranker",
    "RAGAgent",
]

# Reranker implementations are imported conditionally (optional dependencies)
try:
    from .reranker import CohereReranker  # noqa: F401

    __all__.append("CohereReranker")
except ImportError:
    pass

try:
    from .reranker import JinaReranker  # noqa: F401

    __all__.append("JinaReranker")
except ImportError:
    pass


class RAGAgent:
    """
    High-level API for creating RAG-enabled agents.

    Provides convenient methods to create agents that can answer questions
    about your documents using retrieval-augmented generation.

    Example:
        >>> from selectools import OpenAIProvider
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag import RAGAgent, VectorStore, DocumentLoader
        >>>
        >>> # Set up components
        >>> embedder = OpenAIEmbeddingProvider()
        >>> vector_store = VectorStore.create("memory", embedder=embedder)
        >>>
        >>> # Create RAG agent from directory
        >>> agent = RAGAgent.from_directory(
        ...     directory="./docs",
        ...     provider=OpenAIProvider(),
        ...     vector_store=vector_store
        ... )
        >>>
        >>> # Ask questions about your documents
        >>> response = agent.run("What are the main features?")
    """

    @staticmethod
    def from_documents(
        documents: List[Document],
        provider: "Provider",
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        score_threshold: float = 0.0,
        additional_tools: Optional[List] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> "Agent":
        """
        Create a RAG agent from a list of documents.

        Args:
            documents: List of documents to use as knowledge base
            provider: LLM provider for the agent
            vector_store: Vector store for document embeddings
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            top_k: Number of documents to retrieve per query (default: 3)
            score_threshold: Minimum similarity score (default: 0.0)
            additional_tools: Optional list of additional tools to add
            agent_config: Optional agent configuration

        Returns:
            Configured Agent with RAG tool

        Example:
            >>> from selectools.rag import Document
            >>> docs = [
            ...     Document(text="Python is great", metadata={"source": "intro.txt"}),
            ...     Document(text="JavaScript is popular", metadata={"source": "js.txt"})
            ... ]
            >>> agent = RAGAgent.from_documents(
            ...     documents=docs,
            ...     provider=OpenAIProvider(),
            ...     vector_store=vector_store
            ... )
        """
        from ..agent import Agent

        # Chunk documents
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.split_documents(documents)

        # Add to vector store
        vector_store.add_documents(chunked_docs)

        # Create RAG tool
        rag_tool = RAGTool(vector_store=vector_store, top_k=top_k, score_threshold=score_threshold)

        # Build tools list
        tools = [rag_tool.search_knowledge_base]
        if additional_tools:
            tools.extend(additional_tools)

        # Create and return agent
        return Agent(tools=tools, provider=provider, config=agent_config)

    @staticmethod
    def from_directory(
        directory: str,
        provider: "Provider",
        vector_store: VectorStore,
        glob_pattern: str = "**/*.txt",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        score_threshold: float = 0.0,
        additional_tools: Optional[List] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> "Agent":
        """
        Create a RAG agent from a directory of documents.

        Args:
            directory: Path to directory containing documents
            provider: LLM provider for the agent
            vector_store: Vector store for document embeddings
            glob_pattern: Glob pattern to match files (default: "**/*.txt")
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            top_k: Number of documents to retrieve per query (default: 3)
            score_threshold: Minimum similarity score (default: 0.0)
            additional_tools: Optional list of additional tools to add
            agent_config: Optional agent configuration

        Returns:
            Configured Agent with RAG tool

        Example:
            >>> agent = RAGAgent.from_directory(
            ...     directory="./docs",
            ...     provider=OpenAIProvider(),
            ...     vector_store=vector_store,
            ...     glob_pattern="**/*.md"
            ... )
            >>> response = agent.run("Summarize the documentation")
        """
        # Load documents
        documents = DocumentLoader.from_directory(directory=directory, glob_pattern=glob_pattern)

        if not documents:
            raise ValueError(f"No documents found in {directory} matching {glob_pattern}")

        # Use from_documents to create agent
        return RAGAgent.from_documents(
            documents=documents,
            provider=provider,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_tools=additional_tools,
            agent_config=agent_config,
        )

    @staticmethod
    def from_files(
        file_paths: List[str],
        provider: "Provider",
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        score_threshold: float = 0.0,
        additional_tools: Optional[List] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> "Agent":
        """
        Create a RAG agent from a list of file paths.

        Args:
            file_paths: List of file paths to load
            provider: LLM provider for the agent
            vector_store: Vector store for document embeddings
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            top_k: Number of documents to retrieve per query (default: 3)
            score_threshold: Minimum similarity score (default: 0.0)
            additional_tools: Optional list of additional tools to add
            agent_config: Optional agent configuration

        Returns:
            Configured Agent with RAG tool

        Example:
            >>> agent = RAGAgent.from_files(
            ...     file_paths=["doc1.txt", "doc2.pdf", "doc3.md"],
            ...     provider=OpenAIProvider(),
            ...     vector_store=vector_store
            ... )
        """
        # Load all files
        documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                docs = DocumentLoader.from_pdf(file_path)
            else:
                docs = DocumentLoader.from_file(file_path)
            documents.extend(docs)

        if not documents:
            raise ValueError("No documents could be loaded from the provided file paths")

        # Use from_documents to create agent
        return RAGAgent.from_documents(
            documents=documents,
            provider=provider,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_tools=additional_tools,
            agent_config=agent_config,
        )
