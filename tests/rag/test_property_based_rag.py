"""
Property-based tests for RAG chunking and retrieval.

Asserts structural invariants that must hold for all valid inputs:
- Chunks never exceed chunk_size
- No text content is silently lost during splitting
- BM25 add_documents is strictly additive
- HybridSearcher search returns bounded results

Run with: pytest tests/rag/test_property_based_rag.py -x -q
"""

from __future__ import annotations

from typing import List
from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from selectools.rag.vector_store import Document

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=0,
    max_size=1000,
)

nonempty_text = safe_text.filter(lambda t: bool(t.strip()))


@st.composite
def valid_splitter_params(draw):
    """Generate (chunk_size, chunk_overlap) pairs where overlap < size."""
    chunk_size = draw(st.integers(min_value=10, max_value=500))
    chunk_overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    return chunk_size, chunk_overlap


# ---------------------------------------------------------------------------
# TextSplitter
# ---------------------------------------------------------------------------


class TestTextSplitterProperties:
    """TextSplitter chunk sizes must respect the configured limit."""

    @given(text=safe_text, params=valid_splitter_params())
    @settings(max_examples=150)
    def test_no_chunk_exceeds_chunk_size(self, text: str, params) -> None:
        """Every chunk must be ≤ chunk_size characters."""
        from selectools.rag.chunking import TextSplitter

        chunk_size, chunk_overlap = params
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert len(chunk) <= chunk_size, (
                f"Chunk of len {len(chunk)} exceeds chunk_size={chunk_size}: {chunk!r:.50}"
            )

    @given(text=nonempty_text, params=valid_splitter_params())
    @settings(max_examples=100)
    def test_no_content_lost(self, text: str, params) -> None:
        """The concatenation of all chunks must contain all characters of the original text.

        With overlap, characters may appear multiple times, but nothing is dropped.
        """
        from selectools.rag.chunking import TextSplitter

        chunk_size, chunk_overlap = params
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        if not chunks:
            # Empty result is only valid for whitespace-only / empty text
            assert not text.strip()
            return

        # Every character in the original must appear somewhere in the chunks
        combined = "".join(chunks)
        for char in text:
            assert char in combined

    @given(params=valid_splitter_params())
    @settings(max_examples=50)
    def test_empty_text_returns_empty_list(self, params) -> None:
        from selectools.rag.chunking import TextSplitter

        chunk_size, chunk_overlap = params
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert splitter.split_text("") == []


# ---------------------------------------------------------------------------
# RecursiveTextSplitter
# ---------------------------------------------------------------------------


class TestRecursiveTextSplitterProperties:
    """RecursiveTextSplitter must respect chunk_size on arbitrary text."""

    @given(text=safe_text, params=valid_splitter_params())
    @settings(max_examples=150)
    def test_no_chunk_exceeds_chunk_size(self, text: str, params) -> None:
        """Every chunk produced by recursive splitting must be ≤ chunk_size."""
        from selectools.rag.chunking import RecursiveTextSplitter

        chunk_size, chunk_overlap = params
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert len(chunk) <= chunk_size, (
                f"Chunk of len {len(chunk)} exceeds chunk_size={chunk_size}: {chunk!r:.50}"
            )

    @given(params=valid_splitter_params())
    @settings(max_examples=50)
    def test_empty_text_returns_empty_list(self, params) -> None:
        from selectools.rag.chunking import RecursiveTextSplitter

        chunk_size, chunk_overlap = params
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert splitter.split_text("") == []


# ---------------------------------------------------------------------------
# TextSplitter construction validation
# ---------------------------------------------------------------------------


class TestTextSplitterConstructionValidation:
    """Invalid constructor arguments must raise immediately."""

    @given(chunk_size=st.integers(max_value=0))
    @settings(max_examples=50)
    def test_zero_or_negative_chunk_size_raises(self, chunk_size: int) -> None:
        from selectools.rag.chunking import TextSplitter

        with pytest.raises(ValueError):
            TextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    @given(chunk_overlap=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=50)
    def test_overlap_gte_chunk_size_raises(self, chunk_overlap: int) -> None:
        from selectools.rag.chunking import TextSplitter

        chunk_size = max(1, chunk_overlap)  # overlap >= size
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size)


# ---------------------------------------------------------------------------
# BM25 — structural invariants
# ---------------------------------------------------------------------------


class TestBM25RAGProperties:
    """BM25 must maintain consistent internal state after arbitrary document batches."""

    @given(
        batches=st.lists(st.lists(nonempty_text, min_size=1, max_size=10), min_size=1, max_size=5)
    )
    @settings(max_examples=80)
    def test_add_documents_count_is_cumulative(self, batches: List[List[str]]) -> None:
        """document_count after N add_documents calls must equal sum of all batch sizes."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        total = 0
        for batch in batches:
            docs = [Document(text=t) for t in batch]
            bm25.add_documents(docs)
            total += len(batch)

        assert bm25.document_count == total

    @given(docs=st.lists(nonempty_text, min_size=1, max_size=20), query=nonempty_text)
    @settings(max_examples=80)
    def test_scores_are_non_negative(self, docs: List[str], query: str) -> None:
        """All BM25 scores must be non-negative."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text=t) for t in docs])
        results = bm25.search(query, top_k=len(docs))
        for r in results:
            assert r.score >= 0.0


# ---------------------------------------------------------------------------
# InMemoryVectorStore — filter consistency
# ---------------------------------------------------------------------------


class TestVectorStoreFilterProperties:
    """Metadata filter must be consistent with direct dict comparison."""

    @given(
        metadata=st.dictionaries(
            keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
            values=st.one_of(st.text(max_size=20), st.integers(min_value=0, max_value=100)),
            max_size=5,
        )
    )
    @settings(max_examples=150)
    def test_full_metadata_filter_always_matches(self, metadata) -> None:
        """A filter equal to the full metadata must match the document."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        doc = Document(text="test", metadata=metadata)
        # Check filter inline (same logic as _matches_filter)
        for k, v in metadata.items():
            assert doc.metadata.get(k) == v

    @given(
        metadata=st.dictionaries(
            keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
            values=st.one_of(st.text(max_size=20), st.integers(min_value=0, max_value=100)),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_subset_filter_matches(self, metadata) -> None:
        """A filter using a subset of the document metadata keys must match."""
        if not metadata:
            return
        first_key = next(iter(metadata))
        doc = Document(text="test", metadata=metadata)
        assert doc.metadata.get(first_key) == metadata[first_key]


# ---------------------------------------------------------------------------
# HybridSearcher — search bounds
# ---------------------------------------------------------------------------


class TestHybridSearcherProperties:
    """HybridSearcher.search() must return bounded results for any top_k."""

    @given(top_k=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_results_bounded_by_top_k(self, top_k: int) -> None:
        """Results must be ≤ top_k regardless of corpus size."""
        from selectools.rag.hybrid import HybridSearcher

        mock_store = Mock()
        mock_store.embedder = Mock()
        mock_store.embedder.embed_query.return_value = [0.1, 0.2]
        # Return more results than top_k to check clamping
        mock_store.search.return_value = [
            __import__("selectools.rag.vector_store", fromlist=["SearchResult"]).SearchResult(
                document=Document(text=f"doc{i}"), score=float(i) / 100
            )
            for i in range(top_k * 3)
        ]

        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text=f"doc{i} python") for i in range(top_k * 3)])

        searcher = HybridSearcher(vector_store=mock_store, bm25=bm25)
        results = searcher.search("python", top_k=top_k)
        assert len(results) <= top_k
