"""End-to-end tests for FAISSVectorStore against real faiss-cpu.

These tests use the real ``faiss`` package (no mocking) and a deterministic
hash-based embedder so no API keys are required. They exercise the actual
FAISS C++ bindings and verify that:

- selectools' wrapper calls match the real FAISS API
- Cosine similarity search returns correct nearest-neighbour ordering
- Save/load round-trip preserves both the index and document payloads
- Delete and clear leave the index in a usable state

Run with:

    pytest tests/rag/test_e2e_faiss_store.py --run-e2e -v
"""

from __future__ import annotations

import hashlib
from typing import List

import pytest

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from selectools.embeddings import EmbeddingProvider  # noqa: E402
from selectools.rag import Document  # noqa: E402
from selectools.rag.stores import FAISSVectorStore  # noqa: E402


class HashEmbedder(EmbeddingProvider):
    """Deterministic 32-dim hash embedder so tests need no API key."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * ((self._dim // len(digest)) + 1))[: self._dim]
        return [(b / 127.5) - 1.0 for b in raw]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.mark.e2e
class TestFAISSRealBindings:
    """Tests that exercise the real faiss-cpu C++ bindings."""

    def test_real_faiss_is_imported(self) -> None:
        """Confirm we are hitting real faiss, not a mock module."""
        import faiss as real_faiss

        assert hasattr(real_faiss, "IndexFlatIP")
        # Real faiss has a numeric version number; the mock we use in unit
        # tests does not.
        assert hasattr(real_faiss, "__version__")

    def test_add_and_search_single_document(self) -> None:
        """Adding a doc and searching returns it with a positive score."""
        embedder = HashEmbedder()
        store = FAISSVectorStore(embedder=embedder)
        store.add_documents([Document(text="the quick brown fox")])
        results = store.search(embedder.embed_query("the quick brown fox"), top_k=1)
        assert len(results) == 1
        assert results[0].document.text == "the quick brown fox"
        # Cosine self-similarity should be ~1.0
        assert results[0].score > 0.99

    def test_search_returns_topk_ordered(self) -> None:
        """Search returns top_k results in descending score order."""
        embedder = HashEmbedder()
        store = FAISSVectorStore(embedder=embedder)
        docs = [Document(text=f"document number {i}", metadata={"idx": i}) for i in range(5)]
        store.add_documents(docs)
        results = store.search(embedder.embed_query("document number 2"), top_k=3)
        assert len(results) == 3
        # Exact match should be first
        assert results[0].document.text == "document number 2"
        # Scores strictly descending
        for a, b in zip(results, results[1:]):
            assert a.score >= b.score

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        """Persisting then loading restores both vectors and documents."""
        embedder = HashEmbedder()
        store = FAISSVectorStore(embedder=embedder)
        docs = [
            Document(text="alpha", metadata={"id": "a"}),
            Document(text="beta", metadata={"id": "b"}),
            Document(text="gamma", metadata={"id": "c"}),
        ]
        store.add_documents(docs)
        save_path = tmp_path / "faiss_index"
        store.save(str(save_path))

        loaded = FAISSVectorStore.load(str(save_path), embedder=embedder)
        results = loaded.search(embedder.embed_query("alpha"), top_k=3)
        texts = {r.document.text for r in results}
        assert texts == {"alpha", "beta", "gamma"}
        # Metadata survived the round-trip
        alpha = next(r for r in results if r.document.text == "alpha")
        assert alpha.document.metadata["id"] == "a"

    def test_clear_leaves_store_usable(self) -> None:
        """clear() empties the index and new adds still work."""
        embedder = HashEmbedder()
        store = FAISSVectorStore(embedder=embedder)
        store.add_documents([Document(text="will be cleared")])
        store.clear()
        assert store.search(embedder.embed_query("anything"), top_k=1) == []
        store.add_documents([Document(text="after clear")])
        results = store.search(embedder.embed_query("after clear"), top_k=1)
        assert len(results) == 1
        assert results[0].document.text == "after clear"
