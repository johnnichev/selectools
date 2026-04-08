"""End-to-end tests for QdrantVectorStore against a real Qdrant instance.

``test_qdrant_store.py`` mocks the ``qdrant_client`` module. This file
requires a running Qdrant server and exercises the real client.

To run:

    # Start Qdrant locally:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

    # Then:
    pytest tests/rag/test_e2e_qdrant_store.py --run-e2e -v

Or point at Qdrant Cloud:

    QDRANT_URL=https://xxx.cloud.qdrant.io \
    QDRANT_API_KEY=... \
    pytest tests/rag/test_e2e_qdrant_store.py --run-e2e -v

Tests skip automatically if no Qdrant is reachable.
"""

from __future__ import annotations

import hashlib
import os
import socket
import uuid
from typing import List
from urllib.parse import urlparse

import pytest

pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from selectools.embeddings import EmbeddingProvider  # noqa: E402
from selectools.rag import Document  # noqa: E402
from selectools.rag.stores import QdrantVectorStore  # noqa: E402

pytestmark = pytest.mark.e2e


def _qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable() -> bool:
    url = urlparse(_qdrant_url())
    host = url.hostname or "localhost"
    port = url.port or (443 if url.scheme == "https" else 6333)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def qdrant_or_skip() -> None:
    if not _qdrant_reachable():
        pytest.skip(f"Qdrant not reachable at {_qdrant_url()}")


class HashEmbedder(EmbeddingProvider):
    """Deterministic 32-dim hash embedder so tests need no API key."""

    @property
    def dimension(self) -> int:
        return 32

    def embed_query(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * 2)[:32]
        return [(b / 127.5) - 1.0 for b in raw]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def qdrant_store(qdrant_or_skip: None) -> QdrantVectorStore:
    """Create a QdrantVectorStore with a unique collection per test."""
    collection = f"selectools_e2e_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        embedder=HashEmbedder(),
        collection_name=collection,
        url=_qdrant_url(),
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False,  # REST is more reliable for e2e
    )
    yield store
    # Cleanup: drop the collection
    try:
        store.clear()
    except Exception:
        pass


class TestQdrantRealServer:
    def test_add_and_search(self, qdrant_store: QdrantVectorStore) -> None:
        """Real add + search round-trip against a real Qdrant instance."""
        docs = [
            Document(text="the first document", metadata={"id": "a"}),
            Document(text="the second document", metadata={"id": "b"}),
            Document(text="another unrelated text", metadata={"id": "c"}),
        ]
        qdrant_store.add_documents(docs)
        query_vec = qdrant_store.embedder.embed_query("the first document")
        results = qdrant_store.search(query_vec, top_k=3)
        assert len(results) == 3
        # Exact-match doc should be first
        assert results[0].document.text == "the first document"

    def test_clear_empties_collection(self, qdrant_store: QdrantVectorStore) -> None:
        """clear() removes all documents from the real collection."""
        qdrant_store.add_documents([Document(text="temporary")])
        qdrant_store.clear()
        query_vec = qdrant_store.embedder.embed_query("temporary")
        results = qdrant_store.search(query_vec, top_k=1)
        assert results == []
