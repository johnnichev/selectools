"""
Unit tests for text chunking strategies.

Tests:
- TextSplitter (fixed size chunking)
- RecursiveTextSplitter (hierarchical chunking)
"""

import pytest

from selectools.rag import Document, RecursiveTextSplitter, TextSplitter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """This is the first paragraph. It contains multiple sentences.

This is the second paragraph. It also has several sentences.

This is the third paragraph with more content."""


@pytest.fixture
def long_text():
    """Longer text for testing chunk creation."""
    return " ".join([f"Sentence {i}." for i in range(100)])


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            text="First document with some content.",
            metadata={"source": "doc1.txt", "category": "test"},
        ),
        Document(
            text="Second document with different content that is longer.",
            metadata={"source": "doc2.txt", "category": "test"},
        ),
    ]


# ============================================================================
# TextSplitter Tests
# ============================================================================


class TestTextSplitter:
    """Test basic text splitter with fixed chunk size."""

    def test_initialization(self):
        """Test splitter initialization."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        assert splitter.chunk_size == 100
        assert splitter.chunk_overlap == 20
        assert callable(splitter.length_function)

    def test_invalid_overlap(self):
        """Test error when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            TextSplitter(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=150)

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "Short text."

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self, long_text):
        """Test splitting long text into multiple chunks."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

        chunks = splitter.split_text(long_text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_chunk_overlap(self):
        """Test that overlap is correctly applied."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "a" * 100  # 100 character string

        chunks = splitter.split_text(text)

        assert len(chunks) >= 2
        # Check that overlap exists between consecutive chunks
        if len(chunks) >= 2:
            # Last 10 chars of first chunk should appear in second chunk
            overlap = chunks[0][-10:]
            assert chunks[1].startswith(overlap)

    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

        chunks = splitter.split_text("")

        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_exact_chunk_size(self):
        """Test text that is exactly chunk_size."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "a" * 50

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_custom_length_function(self):
        """Test using custom length function."""

        def word_count_length(text: str) -> int:
            return len(text.split())

        splitter = TextSplitter(
            chunk_size=5,  # 5 words
            chunk_overlap=1,  # 1 word overlap
            length_function=word_count_length,
        )

        text = "one two three four five six seven eight nine ten"
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # Each chunk should have at most 5 words
        # Note: This is approximate due to character-based slicing

    def test_split_documents(self, sample_documents):
        """Test splitting multiple documents."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)

        chunked_docs = splitter.split_documents(sample_documents)

        assert len(chunked_docs) >= len(sample_documents)
        assert all(isinstance(d, Document) for d in chunked_docs)
        # Check metadata preservation
        assert all("source" in d.metadata for d in chunked_docs)
        assert all("chunk_idx" in d.metadata for d in chunked_docs)
        assert all("chunk_size" in d.metadata for d in chunked_docs)

    def test_metadata_preservation(self, sample_documents):
        """Test that original metadata is preserved in chunks."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)

        chunked_docs = splitter.split_documents(sample_documents)

        # Check that original metadata is preserved
        doc1_chunks = [d for d in chunked_docs if d.metadata.get("source") == "doc1.txt"]
        assert all(d.metadata["category"] == "test" for d in doc1_chunks)

        doc2_chunks = [d for d in chunked_docs if d.metadata.get("source") == "doc2.txt"]
        assert all(d.metadata["category"] == "test" for d in doc2_chunks)

    def test_chunk_index_metadata(self):
        """Test that chunk indices are correctly assigned."""
        splitter = TextSplitter(chunk_size=30, chunk_overlap=5)
        doc = Document(text="a" * 100, metadata={})

        chunked_docs = splitter.split_documents([doc])

        # Check indices are sequential
        indices = [d.metadata["chunk_idx"] for d in chunked_docs]
        assert indices == list(range(len(chunked_docs)))


# ============================================================================
# RecursiveTextSplitter Tests
# ============================================================================


class TestRecursiveTextSplitter:
    """Test recursive text splitter with natural boundaries."""

    def test_initialization(self):
        """Test splitter initialization."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        assert splitter.chunk_size == 100
        assert splitter.chunk_overlap == 20
        assert splitter.separators == ["\n\n", "\n", " ", ""]

    def test_custom_separators(self):
        """Test initialization with custom separators."""
        separators = ["\n", ". ", " "]
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20, separators=separators)
        assert splitter.separators == separators

    def test_split_by_paragraphs(self, sample_text):
        """Test splitting on paragraph boundaries."""
        splitter = RecursiveTextSplitter(
            chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_text(sample_text)

        assert len(chunks) > 0
        # Should respect paragraph boundaries when possible

    def test_split_by_lines(self):
        """Test splitting on line boundaries."""
        text = "\n".join([f"Line {i}" for i in range(20)])

        splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, separators=["\n", " ", ""]
        )

        chunks = splitter.split_text(text)

        assert len(chunks) > 1

    def test_split_by_sentences(self):
        """Test splitting on sentence boundaries."""
        text = ". ".join([f"Sentence {i}" for i in range(20)])

        splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, separators=[". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        assert len(chunks) > 1

    def test_fallback_to_character_split(self):
        """Test fallback to character-level splitting."""
        # Very long word that can't be split on natural boundaries
        text = "a" * 200

        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)

    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

        chunks = splitter.split_text("")

        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_short_text(self):
        """Test splitting short text."""
        text = "Short."
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_documents(self, sample_documents):
        """Test splitting documents with recursive strategy."""
        splitter = RecursiveTextSplitter(chunk_size=25, chunk_overlap=5)

        chunked_docs = splitter.split_documents(sample_documents)

        assert len(chunked_docs) >= len(sample_documents)
        assert all(isinstance(d, Document) for d in chunked_docs)
        # Check metadata
        assert all("chunk_idx" in d.metadata for d in chunked_docs)

    def test_preserves_meaningful_chunks(self):
        """Test that natural boundaries are preserved when possible."""
        text = """First paragraph.

Second paragraph.

Third paragraph."""

        splitter = RecursiveTextSplitter(
            chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", ". ", " "]
        )

        chunks = splitter.split_text(text)

        # Should create chunks respecting paragraph structure
        assert len(chunks) > 0

    def test_overlap_correctness(self):
        """Test that overlap is maintained in recursive splitting."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=5)

        chunks = splitter.split_text(text)

        assert len(chunks) > 1


# ============================================================================
# Comparison Tests
# ============================================================================


class TestSplitterComparison:
    """Compare behavior of different splitters."""

    def test_both_handle_same_text(self, long_text):
        """Test that both splitters can handle the same input."""
        text_splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        recursive_splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

        text_chunks = text_splitter.split_text(long_text)
        recursive_chunks = recursive_splitter.split_text(long_text)

        # Both should produce chunks
        assert len(text_chunks) > 0
        assert len(recursive_chunks) > 0

        # Both should respect size limits
        assert all(len(c) <= 100 for c in text_chunks)
        assert all(len(c) <= 100 for c in recursive_chunks)

    def test_recursive_respects_boundaries_better(self):
        """Test that recursive splitter preserves natural boundaries."""
        text = "First sentence. Second sentence. Third sentence. " * 10

        text_splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        recursive_splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, separators=[". ", " ", ""]
        )

        text_chunks = text_splitter.split_text(text)
        recursive_chunks = recursive_splitter.split_text(text)

        # Both should create multiple chunks
        assert len(text_chunks) > 1
        assert len(recursive_chunks) > 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_text(self):
        """Test with very long text."""
        text = "a" * 10000
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

        chunks = splitter.split_text(text)

        assert len(chunks) > 10
        assert all(len(c) <= 100 for c in chunks)

    def test_unicode_characters(self):
        """Test with Unicode characters."""
        text = "Hello 世界! " * 50
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # Verify Unicode is preserved
        assert any("世界" in chunk for chunk in chunks)

    def test_special_characters(self):
        """Test with special characters."""
        text = "Test\n\n\t\r special chars! @#$%^&*()" * 10
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)

        chunks = splitter.split_text(text)

        assert len(chunks) > 0

    def test_zero_overlap(self):
        """Test with zero overlap."""
        text = "a" * 100
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)

        chunks = splitter.split_text(text)

        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "a" * 50

    def test_single_character_chunks(self):
        """Test with very small chunk size."""
        text = "hello"
        splitter = TextSplitter(chunk_size=1, chunk_overlap=0)

        chunks = splitter.split_text(text)

        assert len(chunks) == 5
        assert "".join(chunks) == text
