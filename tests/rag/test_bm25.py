"""
Tests for BM25 keyword search engine.
"""

from __future__ import annotations

import pytest

from selectools.rag import Document
from selectools.rag.bm25 import BM25


class TestBM25Tokenization:
    """Test tokenization and text processing."""

    def test_basic_tokenization(self) -> None:
        bm25 = BM25(remove_stopwords=False)
        tokens = bm25.tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_stopword_removal(self) -> None:
        bm25 = BM25(remove_stopwords=True)
        tokens = bm25.tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "fast" in tokens

    def test_stopwords_disabled(self) -> None:
        bm25 = BM25(remove_stopwords=False)
        tokens = bm25.tokenize("the cat is here")
        assert "the" in tokens
        assert "is" in tokens

    def test_punctuation_removal(self) -> None:
        bm25 = BM25(remove_stopwords=False)
        tokens = bm25.tokenize("Hello, world! How's it going?")
        assert all("," not in t and "!" not in t and "?" not in t for t in tokens)

    def test_empty_string(self) -> None:
        bm25 = BM25()
        tokens = bm25.tokenize("")
        assert tokens == []

    def test_only_stopwords(self) -> None:
        bm25 = BM25(remove_stopwords=True)
        tokens = bm25.tokenize("the and or is are")
        assert tokens == []

    def test_numbers_preserved(self) -> None:
        bm25 = BM25(remove_stopwords=False)
        tokens = bm25.tokenize("Python 3.9 release")
        assert "python" in tokens
        assert "3" in tokens
        assert "9" in tokens
        assert "release" in tokens


class TestBM25Indexing:
    """Test document indexing."""

    def test_index_documents(self) -> None:
        bm25 = BM25()
        docs = [
            Document(text="Python programming language"),
            Document(text="Java programming language"),
        ]
        bm25.index_documents(docs)
        assert bm25.document_count == 2

    def test_index_replaces_previous(self) -> None:
        bm25 = BM25()
        bm25.index_documents([Document(text="First batch")])
        assert bm25.document_count == 1

        bm25.index_documents([Document(text="Second"), Document(text="Third")])
        assert bm25.document_count == 2

    def test_add_documents_incremental(self) -> None:
        bm25 = BM25()
        bm25.index_documents([Document(text="First document")])
        assert bm25.document_count == 1

        bm25.add_documents([Document(text="Second document")])
        assert bm25.document_count == 2

    def test_index_empty_list(self) -> None:
        bm25 = BM25()
        bm25.index_documents([])
        assert bm25.document_count == 0

    def test_clear(self) -> None:
        bm25 = BM25()
        bm25.index_documents([Document(text="Something")])
        assert bm25.document_count == 1

        bm25.clear()
        assert bm25.document_count == 0


class TestBM25Search:
    """Test search functionality."""

    @pytest.fixture()
    def bm25_with_docs(self) -> BM25:
        bm25 = BM25()
        docs = [
            Document(text="Python is a popular programming language for data science"),
            Document(text="Java is widely used in enterprise applications"),
            Document(text="Machine learning algorithms use Python extensively"),
            Document(text="JavaScript powers the modern web browsers"),
            Document(text="Python web frameworks include Django and Flask"),
        ]
        bm25.index_documents(docs)
        return bm25

    def test_basic_search(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("Python programming")
        assert len(results) > 0
        assert results[0].document.text.startswith("Python is a popular")

    def test_search_returns_search_results(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("Python")
        for r in results:
            assert hasattr(r, "document")
            assert hasattr(r, "score")
            assert isinstance(r.document, Document)
            assert isinstance(r.score, float)

    def test_search_sorted_by_score(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("Python", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limit(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("Python", top_k=2)
        assert len(results) <= 2

    def test_search_no_matches(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("quantum entanglement physics")
        assert len(results) == 0

    def test_search_empty_query(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("")
        assert len(results) == 0

    def test_search_empty_index(self) -> None:
        bm25 = BM25()
        results = bm25.search("anything")
        assert len(results) == 0

    def test_search_excludes_zero_scores(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("Python")
        for r in results:
            assert r.score > 0.0

    def test_search_with_metadata_filter(self) -> None:
        bm25 = BM25()
        docs = [
            Document(text="Python tutorial for beginners", metadata={"topic": "python"}),
            Document(text="Python advanced patterns", metadata={"topic": "python"}),
            Document(text="Java tutorial for beginners", metadata={"topic": "java"}),
        ]
        bm25.index_documents(docs)

        results = bm25.search("tutorial beginners", filter={"topic": "python"})
        assert len(results) == 1
        assert results[0].document.metadata["topic"] == "python"

    def test_search_filter_no_matches(self) -> None:
        bm25 = BM25()
        docs = [
            Document(text="Python programming", metadata={"lang": "python"}),
        ]
        bm25.index_documents(docs)

        results = bm25.search("Python", filter={"lang": "rust"})
        assert len(results) == 0

    def test_exact_term_match_scores_higher(self) -> None:
        bm25 = BM25()
        docs = [
            Document(text="selectools library for Python agent tool calling"),
            Document(text="Python programming is great for building tools"),
        ]
        bm25.index_documents(docs)

        results = bm25.search("selectools")
        assert len(results) >= 1
        assert "selectools" in results[0].document.text

    def test_query_only_stopwords(self, bm25_with_docs: BM25) -> None:
        results = bm25_with_docs.search("the is and or")
        assert len(results) == 0


class TestBM25Parameters:
    """Test BM25 parameter tuning."""

    def test_custom_k1(self) -> None:
        bm25 = BM25(k1=2.0)
        bm25.index_documents([Document(text="test document")])
        results = bm25.search("test")
        assert len(results) == 1

    def test_custom_b(self) -> None:
        bm25 = BM25(b=0.0)
        docs = [
            Document(text="short"),
            Document(text="this is a much longer document with more words"),
        ]
        bm25.index_documents(docs)
        results = bm25.search("document words")
        assert len(results) >= 1

    def test_k1_zero_reduces_tf_influence(self) -> None:
        docs = [
            Document(text="python python python programming"),
            Document(text="python programming language"),
        ]

        bm25_high_k1 = BM25(k1=2.0)
        bm25_high_k1.index_documents(docs)
        results_high = bm25_high_k1.search("python")

        bm25_low_k1 = BM25(k1=0.0)
        bm25_low_k1.index_documents(docs)
        results_low = bm25_low_k1.search("python")

        assert len(results_high) >= 2
        assert len(results_low) >= 2
        score_diff_high = results_high[0].score - results_high[1].score
        score_diff_low = results_low[0].score - results_low[1].score
        assert score_diff_high >= score_diff_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
