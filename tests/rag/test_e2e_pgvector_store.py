"""End-to-end tests for PgVectorStore against a real PostgreSQL instance.

``test_pgvector_store.py`` mocks psycopg2. This file requires a real
PostgreSQL server with the ``pgvector`` extension installed.

To run:

    # Start Postgres + pgvector locally:
    docker run -d --name pgvector \
      -e POSTGRES_PASSWORD=selectools -p 5432:5432 \
      pgvector/pgvector:pg16

    docker exec pgvector psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector"

    # Then:
    POSTGRES_URL="postgresql://postgres:selectools@localhost:5432/postgres" \
    pytest tests/rag/test_e2e_pgvector_store.py --run-e2e -v

Tests skip automatically if POSTGRES_URL is not set.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import List

import pytest

pytest.importorskip("psycopg2", reason="psycopg2-binary not installed")

from selectools.embeddings import EmbeddingProvider  # noqa: E402
from selectools.rag import Document  # noqa: E402
from selectools.rag.stores import PgVectorStore  # noqa: E402

pytestmark = pytest.mark.e2e


def _postgres_url() -> str | None:
    return os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def postgres_or_skip() -> str:
    url = _postgres_url()
    if not url:
        pytest.skip("POSTGRES_URL / DATABASE_URL not set — skipping pgvector e2e")
    return url


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
def pg_store(postgres_or_skip: str) -> PgVectorStore:
    """Create a PgVectorStore with a unique table per test (auto-cleaned)."""
    table = f"selectools_e2e_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(
        embedder=HashEmbedder(),
        connection_string=postgres_or_skip,
        table_name=table,
        dimensions=32,
    )
    yield store
    # Cleanup: drop the table
    try:
        import psycopg2

        conn = psycopg2.connect(postgres_or_skip)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")  # nosec B608
        conn.close()
    except Exception:
        pass


class TestPgVectorRealServer:
    def test_add_and_search(self, pg_store: PgVectorStore) -> None:
        """Real add + search round-trip against a real Postgres+pgvector."""
        docs = [
            Document(text="alpha document", metadata={"id": "a"}),
            Document(text="beta document", metadata={"id": "b"}),
            Document(text="gamma document", metadata={"id": "c"}),
        ]
        pg_store.add_documents(docs)
        query_vec = pg_store.embedder.embed_query("alpha document")
        results = pg_store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert results[0].document.text == "alpha document"

    def test_clear_truncates_table(self, pg_store: PgVectorStore) -> None:
        """clear() removes all rows from the real pgvector table."""
        pg_store.add_documents([Document(text="to be cleared")])
        pg_store.clear()
        results = pg_store.search(pg_store.embedder.embed_query("to be cleared"), top_k=1)
        assert results == []
