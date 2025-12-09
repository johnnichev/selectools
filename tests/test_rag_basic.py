"""
Basic integration tests for RAG functionality.

Tests the core RAG workflow without requiring external API calls.
"""

import pytest

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy required for RAG")
class TestRAGBasics:
    """Test basic RAG functionality with mocked embeddings."""

    def test_document_creation(self):
        """Test creating documents with metadata."""
        from selectools.rag import Document

        doc = Document(
            text="Hello world",
            metadata={"source": "test.txt", "page": 1},
        )

        assert doc.text == "Hello world"
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["page"] == 1
        assert doc.embedding is None

    def test_document_loader_from_text(self):
        """Test loading documents from text."""
        from selectools.rag import DocumentLoader

        docs = DocumentLoader.from_text("Test document", metadata={"source": "memory"})

        assert len(docs) == 1
        assert docs[0].text == "Test document"
        assert docs[0].metadata["source"] == "memory"

    def test_text_splitter(self):
        """Test text splitting into chunks."""
        from selectools.rag import TextSplitter

        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a longer text that needs to be split."

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # First chunk should be ~10 chars
        assert len(chunks[0]) <= 12  # Allow some flexibility for separators

    def test_recursive_text_splitter(self):
        """Test recursive text splitting."""
        from selectools.rag import RecursiveTextSplitter

        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_in_memory_vector_store_mock(self):
        """Test in-memory vector store with mock embedder."""
        from selectools.rag import Document
        from selectools.rag.stores import InMemoryVectorStore

        # Create a simple mock embedder
        class MockEmbedder:
            def embed_text(self, text):
                # Simple hash-based embedding
                return [float(hash(text) % 100) for _ in range(10)]

            def embed_texts(self, texts):
                return [self.embed_text(t) for t in texts]

            def embed_query(self, query):
                return self.embed_text(query)

            @property
            def dimension(self):
                return 10

        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        # Add documents
        docs = [
            Document(text="Python programming", metadata={"topic": "code"}),
            Document(text="Machine learning", metadata={"topic": "ai"}),
        ]

        ids = store.add_documents(docs)

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

        # Search
        query_embedding = embedder.embed_query("programming")
        results = store.search(query_embedding, top_k=1)

        assert len(results) <= 1
        if results:
            assert hasattr(results[0], "document")
            assert hasattr(results[0], "score")

    def test_rag_tool_creation(self):
        """Test creating a RAG tool."""
        from selectools.rag import RAGTool
        from selectools.rag.stores import InMemoryVectorStore

        # Create mock embedder
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0] * 10

            def embed_texts(self, texts):
                return [[1.0] * 10 for _ in texts]

            def embed_query(self, query):
                return [1.0] * 10

            @property
            def dimension(self):
                return 10

        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        rag_tool = RAGTool(vector_store=store, top_k=3)

        assert rag_tool.vector_store == store
        assert rag_tool.top_k == 3
        assert hasattr(rag_tool.search_knowledge_base, "run")
        assert callable(rag_tool.search_knowledge_base.run)

    def test_vector_store_factory(self):
        """Test vector store factory method."""
        from selectools.rag import VectorStore

        # Create mock embedder
        class MockEmbedder:
            @property
            def dimension(self):
                return 10

        embedder = MockEmbedder()

        # Test creating memory store
        store = VectorStore.create("memory", embedder=embedder)
        assert store is not None

        # Test invalid backend
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            VectorStore.create("invalid_backend", embedder=embedder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
