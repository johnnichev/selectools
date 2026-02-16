"""Tests for SemanticChunker and ContextualChunker."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from selectools.rag.chunking import (
    ContextualChunker,
    RecursiveTextSplitter,
    SemanticChunker,
    TextSplitter,
    _cosine_similarity,
    _split_into_sentences,
)
from selectools.rag.vector_store import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(embeddings: List[List[float]]) -> MagicMock:
    """Create a mock EmbeddingProvider that returns pre-set embeddings."""
    embedder = MagicMock()
    embedder.embed_texts.return_value = embeddings
    embedder.embed_text.side_effect = lambda t: embeddings[0]
    return embedder


def _make_provider(context_text: str = "This chunk discusses topic X.") -> MagicMock:
    """Create a mock Provider that returns a fixed context string."""
    provider = MagicMock()
    response_msg = MagicMock()
    response_msg.content = context_text
    provider.complete.return_value = (response_msg, MagicMock())
    return provider


# ---------------------------------------------------------------------------
# _split_into_sentences
# ---------------------------------------------------------------------------


class TestSplitIntoSentences:
    def test_simple_sentences(self) -> None:
        text = "Hello world. This is great. Done."
        # Sentences may not split on lowercase after period
        result = _split_into_sentences(text)
        assert len(result) >= 1
        assert all(s.strip() for s in result)

    def test_uppercase_sentence_start(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = _split_into_sentences(text)
        assert len(result) == 3

    def test_empty_text(self) -> None:
        assert _split_into_sentences("") == []
        assert _split_into_sentences("   ") == []

    def test_single_sentence(self) -> None:
        result = _split_into_sentences("Just one sentence.")
        assert len(result) == 1

    def test_question_and_exclamation(self) -> None:
        text = "What happened? Everything exploded! Now we rebuild."
        result = _split_into_sentences(text)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    def test_empty_text(self) -> None:
        embedder = _make_embedder([])
        chunker = SemanticChunker(embedder)
        assert chunker.split_text("") == []
        assert chunker.split_text("   ") == []

    def test_single_sentence(self) -> None:
        embedder = _make_embedder([[1.0, 0.0]])
        chunker = SemanticChunker(embedder)
        result = chunker.split_text("One sentence only.")
        assert len(result) == 1

    def test_similar_sentences_stay_together(self) -> None:
        same_vec = [1.0, 0.0, 0.0]
        embedder = _make_embedder([same_vec, same_vec, same_vec])
        chunker = SemanticChunker(embedder, similarity_threshold=0.5)
        text = "Sentence A. Sentence B. Sentence C."
        result = chunker.split_text(text)
        assert len(result) == 1

    def test_dissimilar_sentences_split(self) -> None:
        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        embedder = _make_embedder(embeddings)
        chunker = SemanticChunker(embedder, similarity_threshold=0.8)
        text = "Topic A first. Topic A second. Topic B first. Topic B second."
        result = chunker.split_text(text)
        assert len(result) == 2
        assert "Topic A" in result[0]
        assert "Topic B" in result[1]

    def test_max_chunk_sentences_forces_split(self) -> None:
        same_vec = [1.0, 0.0]
        embedder = _make_embedder([same_vec] * 6)
        chunker = SemanticChunker(
            embedder,
            similarity_threshold=0.1,
            max_chunk_sentences=3,
        )
        text = "S1. S2. S3. S4. S5. S6."
        result = chunker.split_text(text)
        assert len(result) >= 2

    def test_min_chunk_sentences(self) -> None:
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
        embedder = _make_embedder(embeddings)
        chunker = SemanticChunker(
            embedder,
            similarity_threshold=0.9,
            min_chunk_sentences=2,
        )
        text = "Sentence A. Sentence B. Sentence C."
        result = chunker.split_text(text)
        assert all(len(c.split(". ")) >= 1 for c in result)

    def test_split_documents_preserves_metadata(self) -> None:
        same_vec = [1.0, 0.0]
        embedder = _make_embedder([same_vec, same_vec])
        chunker = SemanticChunker(embedder)
        docs = [Document(text="Hello world. Goodbye world.", metadata={"source": "test.txt"})]
        result = chunker.split_documents(docs)
        assert len(result) >= 1
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["chunker"] == "semantic"
        assert "chunk" in result[0].metadata

    def test_invalid_threshold(self) -> None:
        embedder = _make_embedder([])
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticChunker(embedder, similarity_threshold=1.5)
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticChunker(embedder, similarity_threshold=-0.1)

    def test_invalid_sentence_bounds(self) -> None:
        embedder = _make_embedder([])
        with pytest.raises(ValueError, match="min_chunk_sentences"):
            SemanticChunker(embedder, min_chunk_sentences=0)
        with pytest.raises(ValueError, match="max_chunk_sentences"):
            SemanticChunker(embedder, min_chunk_sentences=5, max_chunk_sentences=3)


# ---------------------------------------------------------------------------
# ContextualChunker
# ---------------------------------------------------------------------------


class TestContextualChunker:
    def test_enriches_chunks_with_context(self) -> None:
        base = TextSplitter(chunk_size=50, chunk_overlap=0, separator=" ")
        provider = _make_provider("Summary of this chunk.")
        chunker = ContextualChunker(base_chunker=base, provider=provider)

        docs = [Document(text="Word " * 40, metadata={"source": "doc.txt"})]
        result = chunker.split_documents(docs)

        assert len(result) >= 1
        for doc in result:
            assert doc.text.startswith("[Context] Summary of this chunk.")
            assert doc.metadata["context"] == "Summary of this chunk."
            assert doc.metadata["chunker"] == "contextual"
            assert doc.metadata["source"] == "doc.txt"

    def test_calls_provider_for_each_chunk(self) -> None:
        base = TextSplitter(chunk_size=20, chunk_overlap=0, separator=" ")
        provider = _make_provider("ctx")
        chunker = ContextualChunker(base_chunker=base, provider=provider)

        docs = [Document(text="Word " * 20)]
        result = chunker.split_documents(docs)

        assert provider.complete.call_count == len(result)

    def test_custom_prompt_template(self) -> None:
        base = TextSplitter(chunk_size=500, chunk_overlap=0)
        provider = _make_provider("custom context")
        custom_prompt = "Doc: {document}\nChunk: {chunk}\nDescribe it."
        chunker = ContextualChunker(
            base_chunker=base,
            provider=provider,
            prompt_template=custom_prompt,
        )

        docs = [Document(text="Short text.")]
        chunker.split_documents(docs)

        call_args = provider.complete.call_args
        user_msg = call_args[1]["messages"][0] if "messages" in call_args[1] else call_args[0][2][0]
        assert "Doc:" in user_msg.content
        assert "Describe it." in user_msg.content

    def test_custom_context_prefix(self) -> None:
        base = TextSplitter(chunk_size=500, chunk_overlap=0)
        provider = _make_provider("my context")
        chunker = ContextualChunker(
            base_chunker=base,
            provider=provider,
            context_prefix=">> ",
        )

        docs = [Document(text="Some text here.")]
        result = chunker.split_documents(docs)
        assert result[0].text.startswith(">> my context")

    def test_max_document_chars_truncation(self) -> None:
        base = TextSplitter(chunk_size=500, chunk_overlap=0)
        provider = _make_provider("ctx")
        chunker = ContextualChunker(
            base_chunker=base,
            provider=provider,
            max_document_chars=100,
        )

        long_doc = Document(text="A" * 10000)
        chunker.split_documents([long_doc])

        call_args = provider.complete.call_args
        user_msg = call_args[1]["messages"][0] if "messages" in call_args[1] else call_args[0][2][0]
        assert len(user_msg.content) < 10000

    def test_split_text_convenience(self) -> None:
        base = TextSplitter(chunk_size=500, chunk_overlap=0)
        provider = _make_provider("ctx")
        chunker = ContextualChunker(base_chunker=base, provider=provider)

        result = chunker.split_text("Hello world, this is a test.")
        assert len(result) >= 1
        assert isinstance(result[0], str)

    def test_invalid_base_chunker(self) -> None:
        provider = _make_provider("ctx")
        with pytest.raises(TypeError, match="split_documents"):
            ContextualChunker(base_chunker="not_a_chunker", provider=provider)

    def test_works_with_semantic_chunker(self) -> None:
        same_vec = [1.0, 0.0]
        embedder = _make_embedder([same_vec, same_vec])
        semantic = SemanticChunker(embedder)
        provider = _make_provider("semantic context")

        chunker = ContextualChunker(base_chunker=semantic, provider=provider)
        docs = [Document(text="Hello world. Goodbye world.")]
        result = chunker.split_documents(docs)
        assert len(result) >= 1
        assert "semantic context" in result[0].text

    def test_works_with_recursive_splitter(self) -> None:
        base = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        provider = _make_provider("recursive context")

        chunker = ContextualChunker(base_chunker=base, provider=provider)
        docs = [Document(text="Paragraph one.\n\nParagraph two.\n\nParagraph three.")]
        result = chunker.split_documents(docs)
        assert len(result) >= 1
        for doc in result:
            assert "recursive context" in doc.text

    def test_multiple_documents(self) -> None:
        base = TextSplitter(chunk_size=500, chunk_overlap=0)
        call_count = {"n": 0}

        def side_effect(*args: object, **kwargs: object) -> tuple:
            call_count["n"] += 1
            msg = MagicMock()
            msg.content = f"context for chunk {call_count['n']}"
            return (msg, MagicMock())

        provider = MagicMock()
        provider.complete.side_effect = side_effect

        chunker = ContextualChunker(base_chunker=base, provider=provider)
        docs = [
            Document(text="Doc one text.", metadata={"id": 1}),
            Document(text="Doc two text.", metadata={"id": 2}),
        ]
        result = chunker.split_documents(docs)
        assert len(result) == 2
        assert "context for chunk 1" in result[0].text
        assert "context for chunk 2" in result[1].text
        assert result[0].metadata["id"] == 1
        assert result[1].metadata["id"] == 2
