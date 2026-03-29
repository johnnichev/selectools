"""
Regression tests for Phase 3 (second-pass) RAG bug fixes.

Each test is labelled with the bug ID it guards against. These tests target
edge cases that were invisible to the existing test suite because they require
concurrent execution, adversarial inputs, or unusual argument combinations.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from selectools.rag.vector_store import Document, SearchResult

# ============================================================================
# C1 — InMemoryVectorStore.delete() duplicate-ID list/numpy desync
# ============================================================================


class TestInMemoryVectorStoreDeleteDedup:
    """Regression: deleting the same ID twice must not corrupt list/numpy alignment."""

    def _make_store(self):
        """Build a store with 3 documents without a real embedder."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        mock_embedder = Mock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        store = InMemoryVectorStore(embedder=mock_embedder)

        docs = [
            Document(text="alpha"),
            Document(text="beta"),
            Document(text="gamma"),
        ]
        ids = store.add_documents(docs, embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        return store, ids

    def test_delete_duplicate_ids_does_not_corrupt_alignment(self):
        """Passing the same ID twice must remove it only once (C1)."""
        store, ids = self._make_store()
        assert len(store.documents) == 3

        # Delete the same ID twice — pre-fix this would remove two list entries
        # but only one numpy row, causing index mismatches.
        store.delete([ids[1], ids[1]])

        assert len(store.documents) == 2
        assert store.embeddings.shape[0] == 2  # numpy rows must match doc list
        assert ids[1] not in store.ids

    def test_delete_duplicate_ids_remaining_docs_still_searchable(self):
        """After dedup-delete the surviving docs must search correctly (C1)."""
        store, ids = self._make_store()
        store.delete([ids[0], ids[0]])

        # Search with a vector close to the second document
        results = store.search(query_embedding=[0.3, 0.4], top_k=2)
        texts = [r.document.text for r in results]
        assert "beta" in texts
        assert "gamma" in texts
        assert "alpha" not in texts

    def test_delete_all_via_duplicates(self):
        """Deleting all IDs (with duplicates) must result in an empty store (C1)."""
        store, ids = self._make_store()
        # Pass every ID twice
        store.delete(ids + ids)
        assert len(store.documents) == 0
        assert store.embeddings is None or store.embeddings.shape[0] == 0


# ============================================================================
# L1 — InMemoryVectorStore max_documents=0 / None guard
# ============================================================================


class TestInMemoryVectorStoreMaxDocumentsGuard:
    """Regression: max_documents=0 must not be treated as 'no limit' (L1)."""

    def test_max_documents_none_never_warns(self, recwarn):
        """max_documents=None (default) must never emit a warning (L1)."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        mock_embedder = Mock()
        store = InMemoryVectorStore(embedder=mock_embedder, max_documents=None)
        docs = [Document(text=f"doc{i}") for i in range(10)]
        store.add_documents(docs, embeddings=[[float(i)] * 2 for i in range(10)])

        capacity_warnings = [w for w in recwarn.list if "max_documents" in str(w.message)]
        assert len(capacity_warnings) == 0

    def test_max_documents_zero_is_not_skipped(self, recwarn):
        """max_documents=0 is falsy but must still trigger the warning (L1).

        A store with limit 0 is unusual but must not silently skip the check.
        """
        import warnings

        from selectools.rag.stores.memory import InMemoryVectorStore

        mock_embedder = Mock()
        store = InMemoryVectorStore(embedder=mock_embedder, max_documents=0)
        docs = [Document(text="overflow")]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.add_documents(docs, embeddings=[[0.1, 0.2]])

        capacity_warnings = [w for w in caught if "max_documents" in str(w.message)]
        assert len(capacity_warnings) == 1


# ============================================================================
# H1 — BM25 thread-safety
# ============================================================================


class TestBM25ThreadSafety:
    """Regression: concurrent add_documents must not corrupt BM25 state (H1)."""

    def test_concurrent_add_documents_count_consistent(self):
        """All documents added from N threads must be reflected in doc_count (H1)."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        errors: List[Exception] = []
        n_threads = 10
        docs_per_thread = 20

        def add_batch(thread_id: int):
            try:
                docs = [Document(text=f"thread{thread_id} doc{i}") for i in range(docs_per_thread)]
                bm25.add_documents(docs)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent add_documents raised: {errors}"
        assert bm25.document_count == n_threads * docs_per_thread

    def test_concurrent_index_and_search_no_crash(self):
        """Calling search while index_documents runs concurrently must not crash (H1)."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        initial = [Document(text="initial document about python") for _ in range(50)]
        bm25.index_documents(initial)

        errors: List[Exception] = []

        def reindex():
            for _ in range(5):
                docs = [Document(text=f"concurrent doc {i}") for i in range(20)]
                try:
                    bm25.index_documents(docs)
                except Exception as e:
                    errors.append(e)

        def search():
            for _ in range(20):
                try:
                    bm25.search("python doc", top_k=3)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reindex), threading.Thread(target=search)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent index/search raised: {errors}"


# ============================================================================
# L3 — BM25.search() top_k validation
# ============================================================================


class TestBM25TopKValidation:
    """Regression: top_k < 1 must raise ValueError immediately (L3)."""

    def test_top_k_zero_raises(self):
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="hello")])
        with pytest.raises(ValueError, match="top_k"):
            bm25.search("hello", top_k=0)

    def test_top_k_negative_raises(self):
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="hello")])
        with pytest.raises(ValueError, match="top_k"):
            bm25.search("hello", top_k=-5)

    def test_top_k_one_works(self):
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="hello world")])
        results = bm25.search("hello", top_k=1)
        assert len(results) <= 1


# ============================================================================
# H2 — ChromaVectorStore.search() n_results clamping
# ============================================================================


class TestChromaSearchNResultsClamping:
    """Regression: requesting more results than stored docs must not error (H2)."""

    def _make_chroma_store(self, collection_size: int):
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = collection_size
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.EphemeralClient.return_value = mock_client
        return mock_chroma, mock_collection

    def test_top_k_larger_than_collection_clamps_n_results(self):
        """search(top_k=100) on a 3-doc collection must pass n_results=3 to Chroma (H2)."""
        mock_chroma, mock_collection = self._make_chroma_store(collection_size=3)
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["a", "b", "c"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.3]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=Mock())
            store.search([0.1] * 4, top_k=100)

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 3

    def test_empty_collection_returns_empty_without_query(self):
        """search on an empty collection must return [] without calling query (H2)."""
        mock_chroma, mock_collection = self._make_chroma_store(collection_size=0)

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=Mock())
            results = store.search([0.1] * 4, top_k=5)

        assert results == []
        mock_collection.query.assert_not_called()


# ============================================================================
# M4 — ChromaVectorStore.add_documents() uses upsert (idempotency)
# ============================================================================


class TestChromaUpsertIdempotency:
    """Regression: add_documents must call upsert, not add (M4)."""

    def test_add_documents_calls_upsert(self):
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.EphemeralClient.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=Mock())
            store.add_documents([Document(text="hello")], embeddings=[[0.1, 0.2]])

        mock_collection.upsert.assert_called_once()
        mock_collection.add.assert_not_called()


# ============================================================================
# M1 — HybridSearcher explicit vector_top_k=0 / keyword_top_k=0
# ============================================================================


class TestHybridSearcherExplicitTopK:
    """Regression: vector_top_k=0 must NOT fall back to candidate_k (M1)."""

    def test_explicit_vector_top_k_zero_is_respected(self):
        """vector_top_k=0 must be forwarded as-is, not replaced by candidate_k."""
        from selectools.rag.hybrid import HybridSearcher

        mock_store = Mock()
        mock_store.embedder = Mock()
        mock_store.embedder.embed_query.return_value = [0.1, 0.2]
        mock_store.search.return_value = []

        mock_bm25 = Mock()
        mock_bm25.search.return_value = []

        searcher = HybridSearcher(vector_store=mock_store, bm25=mock_bm25)
        searcher.search("test", top_k=5, vector_top_k=0)

        # vector store must have been called with top_k=0, not top_k=10
        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs["top_k"] == 0

    def test_explicit_keyword_top_k_zero_is_respected(self):
        """keyword_top_k=0 must be forwarded as-is, not replaced by candidate_k."""
        from selectools.rag.hybrid import HybridSearcher

        mock_store = Mock()
        mock_store.embedder = Mock()
        mock_store.embedder.embed_query.return_value = [0.1, 0.2]
        mock_store.search.return_value = []

        mock_bm25 = Mock()
        mock_bm25.search.return_value = []

        searcher = HybridSearcher(vector_store=mock_store, bm25=mock_bm25)
        searcher.search("test", top_k=5, keyword_top_k=0)

        call_kwargs = mock_bm25.search.call_args[1]
        assert call_kwargs["top_k"] == 0

    def test_none_vector_top_k_defaults_to_candidate_k(self):
        """vector_top_k=None (default) must fall back to top_k * 2."""
        from selectools.rag.hybrid import HybridSearcher

        mock_store = Mock()
        mock_store.embedder = Mock()
        mock_store.embedder.embed_query.return_value = [0.1, 0.2]
        mock_store.search.return_value = []

        mock_bm25 = Mock()
        mock_bm25.search.return_value = []

        searcher = HybridSearcher(vector_store=mock_store, bm25=mock_bm25)
        searcher.search("test", top_k=5)

        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs["top_k"] == 10  # 5 * 2


# ============================================================================
# M2 — ContextualChunker provider error fallback
# ============================================================================


class TestContextualChunkerProviderErrorFallback:
    """Regression: provider failure must not abort chunking pipeline (M2)."""

    def test_provider_exception_produces_empty_context_not_crash(self):
        """When provider.complete() raises, chunk must still be returned with empty context."""
        from selectools.rag.chunking import ContextualChunker, TextSplitter

        base = TextSplitter(chunk_size=50, chunk_overlap=0)
        mock_provider = Mock()
        mock_provider.complete.side_effect = RuntimeError("network error")

        chunker = ContextualChunker(base_chunker=base, provider=mock_provider)
        docs = [Document(text="A" * 100)]  # 2 chunks of 50

        result = chunker.split_documents(docs)
        assert len(result) == 2
        for doc in result:
            # Context should be empty string when provider fails
            assert doc.metadata.get("context") == ""

    def test_partial_provider_failure_continues(self):
        """Provider failure on chunk 1 must not prevent chunk 2 from being processed."""
        from selectools.rag.chunking import ContextualChunker, TextSplitter
        from selectools.types import Message, Role

        base = TextSplitter(chunk_size=50, chunk_overlap=0)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            good_msg = Mock()
            good_msg.content = "context for chunk 2"
            return good_msg, {}

        mock_provider = Mock()
        mock_provider.complete.side_effect = side_effect

        chunker = ContextualChunker(base_chunker=base, provider=mock_provider)
        docs = [Document(text="A" * 100)]
        result = chunker.split_documents(docs)

        assert len(result) == 2
        assert result[0].metadata["context"] == ""
        assert result[1].metadata["context"] == "context for chunk 2"


# ============================================================================
# M3 — ContextualChunker template validation at construction time
# ============================================================================


class TestContextualChunkerTemplateValidation:
    """Regression: invalid prompt templates must fail at construction, not at use (M3)."""

    def test_missing_document_placeholder_raises(self):
        from selectools.rag.chunking import ContextualChunker, TextSplitter

        base = TextSplitter(chunk_size=100, chunk_overlap=10)
        with pytest.raises(ValueError, match=r"\{document\}"):
            ContextualChunker(
                base_chunker=base,
                provider=Mock(),
                prompt_template="Only has {chunk} placeholder",
            )

    def test_missing_chunk_placeholder_raises(self):
        from selectools.rag.chunking import ContextualChunker, TextSplitter

        base = TextSplitter(chunk_size=100, chunk_overlap=10)
        with pytest.raises(ValueError, match=r"\{chunk\}"):
            ContextualChunker(
                base_chunker=base,
                provider=Mock(),
                prompt_template="Only has {document} placeholder",
            )

    def test_valid_template_accepted(self):
        from selectools.rag.chunking import ContextualChunker, TextSplitter

        base = TextSplitter(chunk_size=100, chunk_overlap=10)
        chunker = ContextualChunker(
            base_chunker=base,
            provider=Mock(),
            prompt_template="Document: {document}\nChunk: {chunk}",
        )
        assert "{document}" in chunker.prompt_template
        assert "{chunk}" in chunker.prompt_template

    def test_default_template_is_valid(self):
        """The built-in default template must pass its own validation (M3)."""
        from selectools.rag.chunking import ContextualChunker, TextSplitter

        base = TextSplitter(chunk_size=100, chunk_overlap=10)
        # Should not raise
        ContextualChunker(base_chunker=base, provider=Mock())


# ============================================================================
# M5 — DocumentLoader.from_pdf() encrypted PDF error handling
# ============================================================================


class TestDocumentLoaderEncryptedPDF:
    """Regression: encrypted PDFs must raise ValueError, not PdfReadError (M5)."""

    def test_encrypted_pdf_raises_value_error(self, tmp_path):
        """PdfReadError from pypdf must be caught and re-raised as ValueError (M5)."""
        from selectools.rag.loaders import DocumentLoader

        fake_pdf = tmp_path / "encrypted.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")  # not a valid PDF

        # Patch PdfReader to simulate encrypted file error
        mock_pypdf = MagicMock()

        try:
            from pypdf.errors import PdfReadError as _RealPdfReadError

            PdfReadError = _RealPdfReadError
        except ImportError:
            PdfReadError = Exception

        mock_pypdf.PdfReader.side_effect = PdfReadError("file has not been decrypted")
        mock_pypdf.errors.PdfReadError = PdfReadError

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
            with pytest.raises(ValueError, match="encrypted or corrupt"):
                DocumentLoader.from_pdf(str(fake_pdf))

    def test_valid_pdf_still_works(self, tmp_path):
        """Non-encrypted PDF must continue to work after the error-handling addition (M5)."""
        from selectools.rag.loaders import DocumentLoader

        mock_pypdf = MagicMock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Hello from PDF"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pypdf.PdfReader.return_value = mock_reader
        mock_pypdf.errors.PdfReadError = Exception

        fake_pdf = tmp_path / "good.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
            docs = DocumentLoader.from_pdf(str(fake_pdf))

        assert len(docs) == 1
        assert docs[0].text == "Hello from PDF"


# ============================================================================
# L2 — SemanticSearchTool conditional "..." suffix
# ============================================================================


class TestSemanticSearchToolTruncation:
    """Regression: short texts must not get a spurious trailing '...' (L2)."""

    def _make_tool(self, text: str) -> "str":
        from selectools.rag.tools import SemanticSearchTool

        mock_store = Mock()
        mock_store.embedder = Mock()
        mock_store.embedder.embed_query.return_value = [0.1, 0.2]
        mock_store.search.return_value = [
            SearchResult(
                document=Document(text=text, metadata={"source": "test.txt"}),
                score=0.9,
            )
        ]

        tool_obj = SemanticSearchTool(vector_store=mock_store, top_k=1)
        # semantic_search is a Tool object; call the underlying function directly.
        return tool_obj.semantic_search.function(tool_obj, "query")

    def test_short_text_no_ellipsis(self):
        """Text under 200 chars must NOT end with '...' (L2)."""
        result = self._make_tool("Short text")
        assert "Short text..." not in result
        assert "Short text" in result

    def test_long_text_gets_ellipsis(self):
        """Text over 200 chars must be truncated with '...' appended (L2)."""
        long_text = "x" * 300
        result = self._make_tool(long_text)
        assert "x" * 200 + "..." in result

    def test_exactly_200_chars_no_ellipsis(self):
        """Text of exactly 200 chars is not truncated — no '...' (L2)."""
        text = "a" * 200
        result = self._make_tool(text)
        # The preview is text[:200] = full text; no truncation occurred
        assert text + "..." not in result


# ============================================================================
# Phase 2 regression: BM25 ZeroDivisionError on empty avg_doc_len
# ============================================================================


class TestBM25ZeroDivisionGuard:
    """Regression: single-token documents must not trigger ZeroDivisionError."""

    def test_single_doc_single_token_no_zero_division(self):
        from selectools.rag.bm25 import BM25

        bm25 = BM25(remove_stopwords=False)
        bm25.index_documents([Document(text="a")])
        # Should not raise ZeroDivisionError
        results = bm25.search("a", top_k=1)
        assert len(results) == 1
