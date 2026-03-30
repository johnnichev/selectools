"""
BM25 (Best Matching 25) keyword search engine.

Pure-Python implementation of Okapi BM25 for keyword-based document retrieval.
Zero external dependencies -- uses only the Python standard library.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .vector_store import Document, SearchResult

_SPLIT_RE = re.compile(r"[^a-z0-9]+")

_STOP_WORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "not",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "because",
    "but",
    "if",
    "about",
    "up",
    "it",
    "its",
    "this",
    "that",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
}


@dataclass
class _IndexedDoc:
    """Internal representation of a document in the BM25 index."""

    document: Document
    term_freqs: Dict[str, int] = field(default_factory=dict)
    doc_len: int = 0


class BM25:
    """
    Okapi BM25 keyword search engine.

    Indexes documents and retrieves them by keyword relevance using the
    standard BM25 scoring function. Pure Python, zero external dependencies.

    Args:
        k1: Term frequency saturation parameter (default: 1.5).
            Higher values increase the influence of term frequency.
        b: Length normalisation parameter (default: 0.75).
            0 = no length normalisation, 1 = full normalisation.
        remove_stopwords: Whether to filter English stop words (default: True).

    Example:
        >>> from selectools.rag import Document
        >>> from selectools.rag.bm25 import BM25
        >>>
        >>> bm25 = BM25()
        >>> docs = [
        ...     Document(text="Python programming language"),
        ...     Document(text="Java programming language"),
        ...     Document(text="Machine learning with Python"),
        ... ]
        >>> bm25.index_documents(docs)
        >>> results = bm25.search("Python programming", top_k=2)
        >>> print(results[0].document.text)
        Python programming language
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        self._lock = threading.Lock()
        self._docs: List[_IndexedDoc] = []
        self._doc_count: int = 0
        self._avg_doc_len: float = 0.0
        self._df: Dict[str, int] = {}  # document frequency per term

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return self._doc_count

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase terms, optionally removing stop words.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of normalised tokens.
        """
        tokens = _SPLIT_RE.split(text.lower())
        tokens = [t for t in tokens if t]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOP_WORDS]
        return tokens

    def index_documents(self, documents: List[Document]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of documents.

        Calling this method replaces any previously indexed documents.

        Args:
            documents: Documents to index.
        """
        new_docs: List[_IndexedDoc] = []
        new_df: Dict[str, int] = {}
        total_len = 0

        for doc in documents:
            tokens = self.tokenize(doc.text)
            tf = Counter(tokens)
            idx_doc = _IndexedDoc(
                document=doc,
                term_freqs=dict(tf),
                doc_len=len(tokens),
            )
            new_docs.append(idx_doc)
            total_len += idx_doc.doc_len
            for term in tf:
                new_df[term] = new_df.get(term, 0) + 1

        with self._lock:
            self._docs = new_docs
            self._df = new_df
            self._doc_count = len(self._docs)
            self._avg_doc_len = total_len / self._doc_count if self._doc_count else 0.0

    def add_documents(self, documents: List[Document]) -> None:
        """
        Incrementally add documents to an existing BM25 index.

        Updates IDF values and average document length accordingly.

        Args:
            documents: Documents to add.
        """
        new_docs: List[_IndexedDoc] = []
        new_df_delta: Dict[str, int] = {}
        total_new_len = 0

        for doc in documents:
            tokens = self.tokenize(doc.text)
            tf = Counter(tokens)
            idx_doc = _IndexedDoc(
                document=doc,
                term_freqs=dict(tf),
                doc_len=len(tokens),
            )
            new_docs.append(idx_doc)
            total_new_len += idx_doc.doc_len
            for term in tf:
                new_df_delta[term] = new_df_delta.get(term, 0) + 1

        with self._lock:
            self._docs.extend(new_docs)
            for term, delta in new_df_delta.items():
                self._df[term] = self._df.get(term, 0) + delta
            self._doc_count = len(self._docs)
            total_len = self._avg_doc_len * (self._doc_count - len(new_docs)) + total_new_len
            self._avg_doc_len = total_len / self._doc_count if self._doc_count else 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for documents matching the query by keyword relevance.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return (default: 5).
            filter: Optional metadata filter dict. Only documents whose metadata
                    contains all key-value pairs are included.

        Returns:
            List of ``SearchResult`` objects sorted by BM25 score (highest first).
            Scores are non-negative; documents with zero relevance are excluded.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Take an atomic snapshot of all shared index state so that concurrent
        # index_documents / add_documents / clear calls cannot race with the
        # scoring loop below (which must not hold the lock during CPU work).
        with self._lock:
            if self._doc_count == 0:
                return []
            docs_snapshot = list(self._docs)
            doc_count_snapshot = self._doc_count
            avg_doc_len_snapshot = self._avg_doc_len
            df_snapshot = dict(self._df)

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        scores: List[float] = []

        for idx_doc in docs_snapshot:
            if filter and not self._matches_filter(idx_doc.document, filter):
                scores.append(0.0)
                continue

            score = self._score_document_snapshot(
                idx_doc, query_terms, doc_count_snapshot, avg_doc_len_snapshot, df_snapshot
            )
            scores.append(score)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[SearchResult] = []
        for i in ranked:
            if scores[i] <= 0.0:
                break
            results.append(
                SearchResult(
                    document=docs_snapshot[i].document,
                    score=scores[i],
                )
            )
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Remove all documents from the index."""
        with self._lock:
            self._docs = []
            self._doc_count = 0
            self._avg_doc_len = 0.0
            self._df = {}

    def _score_document(self, idx_doc: _IndexedDoc, query_terms: List[str]) -> float:
        """Compute BM25 score using current instance state (not thread-safe — use snapshot variant)."""
        return self._score_document_snapshot(
            idx_doc, query_terms, self._doc_count, self._avg_doc_len, self._df
        )

    def _score_document_snapshot(
        self,
        idx_doc: _IndexedDoc,
        query_terms: List[str],
        doc_count: int,
        avg_doc_len: float,
        df: Dict[str, int],
    ) -> float:
        """Compute BM25 score against explicit (snapshot) index parameters.

        Accepts pre-snapshotted ``doc_count``, ``avg_doc_len``, and ``df`` so
        the caller can operate on a consistent view of the index without holding
        the lock during CPU-bound scoring.
        """
        score = 0.0
        for term in query_terms:
            if term not in idx_doc.term_freqs:
                continue

            tf = idx_doc.term_freqs[term]
            term_df = df.get(term, 0)

            idf = math.log((doc_count - term_df + 0.5) / (term_df + 0.5) + 1.0)

            numerator = tf * (self.k1 + 1)
            avg_len = avg_doc_len if avg_doc_len > 0 else 1e-8
            denominator = tf + self.k1 * (1 - self.b + self.b * idx_doc.doc_len / avg_len)
            score += idf * numerator / denominator

        return score

    @staticmethod
    def _matches_filter(doc: Document, filter: Dict[str, Any]) -> bool:
        """Check if document metadata matches all filter criteria."""
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True


__all__ = ["BM25"]
