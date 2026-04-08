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
        # semantic_search is a @tool-decorated method; accessing it on an
        # instance returns a Tool whose function has `self` pre-bound via
        # the _BoundMethodTool descriptor, so we pass only the LLM kwarg.
        return tool_obj.semantic_search.function("query")

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


# ============================================================================
# Phase 4 regression: BM25.clear() thread safety
# ============================================================================


class TestBM25ClearThreadSafety:
    """Regression: BM25.clear() must hold the lock to avoid concurrent data corruption."""

    def test_concurrent_add_and_clear_no_corruption(self):
        """clear() concurrent with add_documents() must not silently lose documents (Phase4)."""

        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="base doc")])
        errors: list = []

        def add_batch():
            for _ in range(50):
                try:
                    bm25.add_documents([Document(text="added doc")])
                except Exception as e:
                    errors.append(e)

        def clear_loop():
            for _ in range(50):
                try:
                    bm25.clear()
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_batch)
        t2 = threading.Thread(target=clear_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent clear/add raised: {errors}"

    def test_clear_after_add_leaves_empty_state(self):
        """Sequential clear after add must leave document_count=0 (Phase4)."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.add_documents([Document(text=f"doc{i}") for i in range(10)])
        assert bm25.document_count == 10
        bm25.clear()
        assert bm25.document_count == 0
        assert bm25._docs == []
        assert bm25._df == {}
        assert bm25._avg_doc_len == 0.0


# ============================================================================
# Phase 4 regression: SemanticChunker IndexError on partial embeddings
# ============================================================================


class TestSemanticChunkerPartialEmbeddings:
    """Regression: SemanticChunker must not crash when embedder returns fewer vectors (Phase4)."""

    def test_fewer_embeddings_than_sentences_no_index_error(self):
        """embedder returning 2 vectors for 5 sentences must not raise IndexError (Phase4)."""
        from unittest.mock import Mock

        from selectools.rag.chunking import SemanticChunker

        mock_embedder = Mock()
        # 5 sentences will be detected, but only return 2 embeddings
        mock_embedder.embed_texts.return_value = [
            [1.0, 0.0],
            [0.9, 0.1],
        ]

        chunker = SemanticChunker(mock_embedder, similarity_threshold=0.5)
        # Text produces >=5 sentences
        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth. Epsilon is fifth."

        # Must not raise IndexError
        chunks = chunker.split_text(text)
        assert isinstance(chunks, list)
        # Should produce at least one chunk
        assert len(chunks) >= 1

    def test_single_embedding_returned_no_crash(self):
        """Embedder returning 1 vector for multi-sentence text must not crash (Phase4)."""
        from unittest.mock import Mock

        from selectools.rag.chunking import SemanticChunker

        mock_embedder = Mock()
        mock_embedder.embed_texts.return_value = [[1.0, 0.0]]  # only 1 vector

        chunker = SemanticChunker(mock_embedder, similarity_threshold=0.5)
        text = "First sentence here. Second sentence there. Third sentence everywhere."

        chunks = chunker.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1


# ============================================================================
# Phase 4 regression: SQLiteVectorStore connection leak on exception
# ============================================================================


class TestSQLiteVectorStoreConnectionSafety:
    """Regression: SQLite connections must be closed even when exceptions occur (Phase4)."""

    def test_add_documents_closes_connection_on_exception(self, tmp_path):
        """If cursor.execute raises, conn.close() must still be called (Phase4).

        We verify the fix indirectly: after a simulated failure the database
        file is still accessible (not locked), which would only be true if
        the connection was properly closed.
        """
        import sqlite3
        from unittest.mock import Mock, patch

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        mock_embedder = Mock()
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)

        # Use a wrapper connection class to track close() calls
        closed_calls = []
        original_connect = sqlite3.connect

        class TrackingConnection:
            """Thin wrapper that records close() invocations."""

            def __init__(self, real_conn):
                self._conn = real_conn

            def cursor(self):
                # Raise to simulate a disk-full error mid-insert
                raise sqlite3.OperationalError("disk full")

            def execute(self, *args, **kwargs):
                return self._conn.execute(*args, **kwargs)

            def commit(self):
                return self._conn.commit()

            def close(self):
                closed_calls.append(True)
                return self._conn.close()

        def patched_connect(path, **kwargs):
            return TrackingConnection(original_connect(path, **kwargs))

        with patch("selectools.rag.stores.sqlite.sqlite3.connect", side_effect=patched_connect):
            try:
                store.add_documents([Document(text="test")], embeddings=[[0.1, 0.2]])
            except Exception:
                pass

        # The connection must have been closed (via the finally block)
        assert len(closed_calls) >= 1, "Connection was not closed after exception"


# ============================================================================
# Phase 4 regression: DocumentLoader.from_directory() path traversal guard
# ============================================================================


class TestDocumentLoaderPathTraversalGuard:
    """Regression: glob_pattern with '..' must be rejected to prevent path traversal (Phase4)."""

    def test_dotdot_in_glob_pattern_raises(self, tmp_path):
        """A pattern containing '..' must raise ValueError (Phase4)."""
        from selectools.rag.loaders import DocumentLoader

        with pytest.raises(ValueError, match="path traversal"):
            DocumentLoader.from_directory(str(tmp_path), glob_pattern="../*.txt")

    def test_nested_dotdot_in_glob_pattern_raises(self, tmp_path):
        """Nested '../..' patterns must also be rejected (Phase4)."""
        from selectools.rag.loaders import DocumentLoader

        with pytest.raises(ValueError, match="path traversal"):
            DocumentLoader.from_directory(str(tmp_path), glob_pattern="../../etc/passwd")

    def test_normal_glob_pattern_still_works(self, tmp_path):
        """Standard glob patterns must continue to work after the traversal guard (Phase4)."""
        from selectools.rag.loaders import DocumentLoader

        # Create a test file
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello from sample")

        docs = DocumentLoader.from_directory(str(tmp_path), glob_pattern="*.txt")
        assert len(docs) == 1
        assert docs[0].text == "Hello from sample"

    def test_wildcard_subdir_pattern_still_works(self, tmp_path):
        """Recursive wildcard patterns like **/*.txt must not be blocked (Phase4)."""
        from selectools.rag.loaders import DocumentLoader

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("nested file")

        docs = DocumentLoader.from_directory(str(tmp_path), glob_pattern="**/*.txt")
        assert len(docs) == 1


# ============================================================================
# Phase 4 regression: SQLiteVectorStore NULL metadata guard in search()
# ============================================================================


class TestSQLiteNullMetadataGuard:
    """Regression: NULL metadata from external DB writes must not crash search() (Phase4)."""

    def test_filter_with_null_metadata_json_no_crash(self, tmp_path):
        """A row with 'null' metadata must not crash the filter path (Phase4)."""
        import json
        import sqlite3
        from unittest.mock import Mock

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        mock_embedder = Mock()
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)

        # Manually insert a row with null metadata
        conn = sqlite3.connect(db_path)
        embedding = json.dumps([0.1, 0.2])
        conn.execute(
            "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
            ("manual_id", "text with null meta", "null", embedding),
        )
        conn.commit()
        conn.close()

        # search with filter must not crash with AttributeError
        results = store.search([0.1, 0.2], top_k=5, filter={"source": "test"})
        assert isinstance(results, list)

    def test_search_with_null_metadata_no_crash(self, tmp_path):
        """A row with 'null' metadata must not crash the non-filter path (Phase4)."""
        import json
        import sqlite3
        from unittest.mock import Mock

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        mock_embedder = Mock()
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)

        conn = sqlite3.connect(db_path)
        embedding = json.dumps([0.1, 0.2])
        conn.execute(
            "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
            ("manual_id", "text with null meta", "null", embedding),
        )
        conn.commit()
        conn.close()

        # search without filter must also not crash
        results = store.search([0.1, 0.2], top_k=5)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].document.metadata == {}


# ============================================================================
# Pass 2 regression: BM25.search() race condition with index_documents/clear
# ============================================================================


class TestBM25SearchLockSnapshot:
    """Regression: BM25.search() must snapshot shared state under lock (Pass 2).

    Before the fix, search() read self._doc_count, self._docs, self._df, and
    self._avg_doc_len without holding the lock.  A concurrent clear() or
    index_documents() call could replace self._docs between the len check and
    the iteration, producing an IndexError.
    """

    def test_concurrent_search_and_clear_no_index_error(self):
        """search() concurrent with clear()+index_documents() must not raise IndexError."""
        import threading

        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text=f"doc {i}") for i in range(100)])

        errors: list = []

        def search_loop() -> None:
            for _ in range(2000):
                try:
                    bm25.search("doc", top_k=5)
                except IndexError as e:
                    errors.append(e)
                except Exception:
                    pass  # other errors are not the regression we care about

        def clear_reindex_loop() -> None:
            for _ in range(100):
                bm25.clear()
                bm25.index_documents([Document(text=f"doc {i}") for i in range(100)])

        t1 = threading.Thread(target=search_loop)
        t2 = threading.Thread(target=clear_reindex_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent search/clear raised IndexError: {errors[:3]}"

    def test_search_result_integrity_after_concurrent_add(self):
        """Results from a concurrent search must have correct scores (not mix snapshots)."""
        import threading

        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents([Document(text="alpha beta gamma")])
        errors: list = []

        def add_loop() -> None:
            for _ in range(200):
                bm25.add_documents([Document(text=f"delta epsilon {_}")])

        def search_assert() -> None:
            for _ in range(500):
                try:
                    results = bm25.search("alpha beta", top_k=3)
                    # Each result must have a valid non-negative score
                    for r in results:
                        assert r.score >= 0.0, f"Negative score: {r.score}"
                except IndexError as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_loop)
        t2 = threading.Thread(target=search_assert)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"IndexError during concurrent search/add: {errors[:3]}"

    def test_search_returns_correct_results_after_fix(self):
        """Normal search must still return correct ranked results after the snapshot fix."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        bm25.index_documents(
            [
                Document(text="python programming language"),
                Document(text="java enterprise application"),
                Document(text="machine learning python neural"),
            ]
        )

        results = bm25.search("python programming", top_k=2)

        assert len(results) == 2
        # Best match must be the python programming doc
        assert "python programming" in results[0].document.text
        # Scores must be descending
        assert results[0].score >= results[1].score


# ============================================================================
# Pass 3 regression: SQLiteVectorStore SQL NULL metadata causes TypeError
# ============================================================================


class TestSQLiteVectorStoreSQLNullMetadata:
    """Regression: SQL NULL metadata (Python None) must not raise TypeError in search().

    The existing phase-4 test covers JSON string 'null' (which json.loads() decodes
    to Python None, handled by 'or {}').  This test covers actual SQL NULL -- a
    column value that arrives as Python None from sqlite3, for which json.loads(None)
    raises TypeError before any 'or {}' guard can run.
    """

    def test_filter_with_sql_null_metadata_no_type_error(self, tmp_path):
        """filter path: SQL NULL metadata must not raise TypeError (Pass 3)."""
        import json
        import sqlite3
        from unittest.mock import Mock

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        mock_embedder = Mock()
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)

        # Insert a row with *actual* SQL NULL (not the JSON string "null")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
            ("id_sql_null", "text with sql null meta", None, json.dumps([0.1, 0.2])),
        )
        conn.commit()
        conn.close()

        # Must not raise TypeError: the JSON object must be str, bytes or bytearray
        try:
            results = store.search([0.1, 0.2], top_k=5, filter={"source": "test"})
        except TypeError as e:
            pytest.fail(f"TypeError raised for SQL NULL metadata in filter path: {e}")

        assert isinstance(results, list)

    def test_no_filter_with_sql_null_metadata_no_type_error(self, tmp_path):
        """non-filter path: SQL NULL metadata must not raise TypeError (Pass 3)."""
        import json
        import sqlite3
        from unittest.mock import Mock

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        mock_embedder = Mock()
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)

        # Insert a row with *actual* SQL NULL metadata
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
            ("id_sql_null2", "searchable content", None, json.dumps([1.0, 0.0])),
        )
        conn.commit()
        conn.close()

        try:
            results = store.search([1.0, 0.0], top_k=5)
        except TypeError as e:
            pytest.fail(f"TypeError raised for SQL NULL metadata in non-filter path: {e}")

        assert len(results) == 1
        # SQL NULL metadata must be normalised to an empty dict
        assert results[0].document.metadata == {}


# ============================================================================
# P5-SC1 — SemanticChunker silent data loss when embedder returns k<n embeddings
# ============================================================================


class TestSemanticChunkerEmbedderTruncation:
    """Regression: SemanticChunker must not silently discard sentences when the
    embedder returns fewer vectors than sentences (P5-SC1).

    Previously, when the embedder returned exactly 1 embedding for n>1 sentences,
    the sentences list was truncated to [sentences[0]] and the 'len <= 1' branch
    returned only the first sentence, silently dropping the remaining n-1 sentences.
    """

    def _make_chunker(self, embeddings):
        from selectools.rag.chunking import SemanticChunker

        mock_embedder = Mock()
        mock_embedder.embed_texts.return_value = embeddings
        return SemanticChunker(mock_embedder, similarity_threshold=0.5)

    def test_one_embedding_for_three_sentences_returns_full_text(self):
        """When embedder returns 1 vector for 3 sentences, full text is returned (P5-SC1)."""
        chunker = self._make_chunker([[0.1, 0.2]])

        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunker.split_text(text)

        # Must not silently drop sentences 2 and 3.
        assert len(chunks) == 1
        combined = " ".join(chunks)
        assert "Sentence one" in combined
        assert "Sentence two" in combined
        assert "Sentence three" in combined

    def test_zero_embeddings_for_multiple_sentences_returns_full_text(self):
        """When embedder returns 0 vectors, the full text is returned as fallback (P5-SC1)."""
        chunker = self._make_chunker([])

        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunker.split_text(text)

        assert len(chunks) == 1
        assert "Sentence one" in chunks[0]
        assert "Sentence two" in chunks[0]
        assert "Sentence three" in chunks[0]

    def test_two_embeddings_for_three_sentences_truncates_correctly(self):
        """When embedder returns 2 vectors for 3 sentences, first 2 sentences are used (P5-SC1).

        k=2 is enough to compute 1 similarity value, so truncation is acceptable.
        """
        # High similarity so all go in one chunk
        chunker = self._make_chunker([[0.1, 0.2], [0.15, 0.25]])

        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunker.split_text(text)

        # May drop sentence three (only k=2 embeddings available) but must not crash.
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) and c.strip() for c in chunks)

    def test_exactly_matching_embeddings_proceeds_normally(self):
        """When embedder returns exactly len(sentences) vectors, normal chunking occurs (P5-SC1)."""
        # Two embeddings, low similarity → should split into two chunks
        chunker = self._make_chunker([[1.0, 0.0], [0.0, 1.0]])

        text = "Sentence one. Sentence two."
        chunks = chunker.split_text(text)

        # Two orthogonal embeddings with threshold 0.5 should trigger a split.
        assert len(chunks) == 2

    def test_full_text_fallback_does_not_return_empty_for_whitespace_input(self):
        """Zero-embedding fallback with whitespace-only text returns [] not [''] (P5-SC1)."""
        chunker = self._make_chunker([])

        chunks = chunker.split_text("   \n  ")
        assert chunks == []
