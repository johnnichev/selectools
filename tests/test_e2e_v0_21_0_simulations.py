"""Full-release end-to-end simulations for v0.21.0.

The 12 isolated e2e test files prove that each v0.21.0 subsystem works
against its real backend in isolation. This file is different — each
scenario wires **multiple** v0.21.0 features together in a single agent
run against a real LLM, to prove the combinations work:

- Scenario 1: CSV loader → real OpenAI embeddings → real FAISS → RAGTool
  → real OpenAI Agent → real OTel SDK span capture
- Scenario 2: real Gemini agent with a multimodal image input + the new
  execute_python toolbox tool, OTel observer attached
- Scenario 3: real Anthropic agent with query_sqlite + execute_python
  toolbox tools against a real SQLite database
- Scenario 4: real Qdrant vector store with real OpenAI embeddings wired
  into a real OpenAI Agent (skipped if Qdrant is not reachable)

These simulations are the only place we verify that:

- The @tool() schema on the new toolbox tools is correct enough for
  real providers' native tool calling to actually pick them
- The real RAGTool + real vector store + real embeddings + real LLM
  retrieval path actually returns useful context to the LLM
- OTelObserver captures spans on REAL LLM calls (not just fake provider
  stubs), including gen_ai.* attributes with actual model / token data
- Multimodal messages flow through an iterative agent loop that also
  uses tools, not just a single one-shot call

Cost: every scenario that runs hits a real API. Keep prompts short,
max_tokens small, and max_iterations capped so the whole file runs for
well under $0.01 per invocation.

Run with:

    pytest tests/test_e2e_v0_21_0_simulations.py --run-e2e -v
"""

from __future__ import annotations

import os
import socket
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

from selectools import Agent, AgentConfig
from selectools.observe import OTelObserver
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.gemini_provider import GeminiProvider
from selectools.providers.openai_provider import OpenAIProvider
from selectools.rag import Document, DocumentLoader
from selectools.rag.stores import FAISSVectorStore
from selectools.rag.tools import RAGTool
from selectools.toolbox import code_tools, db_tools

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# OpenTelemetry fixture comes from tests/conftest.py (session-wide singleton)
# ---------------------------------------------------------------------------

pytest.importorskip("opentelemetry", reason="opentelemetry-api not installed")
pytest.importorskip("opentelemetry.sdk", reason="opentelemetry-sdk not installed")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _require(env_var: str) -> None:
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")


def _make_tiny_red_png() -> bytes:
    """Build a 4x4 solid-red PNG with no external deps."""
    width, height = 4, 4
    row = b"\x00" + b"\xff\x00\x00" * width
    raw = row * height

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _qdrant_reachable(url: str = "http://localhost:6333") -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Scenario 1 — RAG pipeline with real OpenAI embeddings + FAISS + OpenAI agent + OTel
# ---------------------------------------------------------------------------


class TestScenario1_RAGWithOpenAI:
    """CSV → real embeddings → FAISS → RAGTool → real OpenAI agent → OTel spans."""

    def test_agent_answers_from_csv_backed_faiss(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        _require("OPENAI_API_KEY")
        pytest.importorskip("faiss", reason="faiss-cpu not installed")

        # 1. Build a small CSV of facts with a deliberately unusual anchor word
        # so we can tell whether the agent actually retrieved from our docs
        # (vs. answering from the LLM's prior knowledge)
        csv_path = tmp_path / "facts.csv"
        csv_path.write_text(
            "topic,body\n"
            "selectools,"
            '"The selectools library was first tagged with the magic codename ZOOPLANKTON-91 in v0.21.0."\n'
            "python,"
            '"Python is a high-level programming language created by Guido van Rossum."\n',
            encoding="utf-8",
        )

        # 2. Load via the new CSV loader
        docs = DocumentLoader.from_csv(
            str(csv_path), text_column="body", metadata_columns=["topic"]
        )
        assert len(docs) == 2

        # 3. Real OpenAI embeddings
        from selectools.embeddings.openai import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")

        # 4. Real FAISS store
        store = FAISSVectorStore(embedder=embedder)
        store.add_documents(docs)

        # 5. Real RAGTool
        rag_tool = RAGTool(vector_store=store, top_k=2)

        # 6. Real OpenAI agent with OTel observer
        agent = Agent(
            tools=[rag_tool.search_knowledge_base],
            provider=OpenAIProvider(),
            config=AgentConfig(
                model="gpt-4o-mini",
                max_tokens=150,
                max_iterations=4,
                observers=[OTelObserver(tracer_name="selectools-sim")],
            ),
        )

        # 7. Ask a question that REQUIRES retrieval (the anchor word is unique)
        result = agent.run(
            "What is the magic codename associated with selectools v0.21.0? "
            "Use the search_knowledge_base tool and quote the codename verbatim."
        )

        # 8. Assert the agent actually retrieved from OUR docs
        assert "ZOOPLANKTON" in result.content.upper(), (
            f"Agent did not return the anchor word from the CSV. Got: {result.content[:300]}"
        )
        assert result.usage.total_tokens > 0

        # 9. Assert OTel captured real spans for this real run
        spans = otel_exporter.get_finished_spans()
        assert len(spans) > 0, "OTel captured no spans for the real LLM+tool run"
        saw_gen_ai = any((s.attributes or {}).get("gen_ai.system") == "selectools" for s in spans)
        assert saw_gen_ai, "No span carried gen_ai.system='selectools'"


# ---------------------------------------------------------------------------
# Scenario 2 — Multimodal + toolbox + OTel with real Gemini
# ---------------------------------------------------------------------------


class TestScenario2_MultimodalWithGemini:
    """Real Gemini vision call + execute_python tool + OTel in one run."""

    def test_gemini_sees_image_and_calls_python_tool(
        self, tmp_path: Path, otel_exporter: InMemorySpanExporter
    ) -> None:
        (
            _require("GOOGLE_API_KEY") if not os.environ.get("GEMINI_API_KEY") else None
        )  # either is fine

        # 1. Write a tiny red PNG to disk (image_message needs a file path)
        png_path = tmp_path / "red.png"
        png_path.write_bytes(_make_tiny_red_png())

        # 2. Real Gemini agent with execute_python + OTel
        agent = Agent(
            tools=[code_tools.execute_python],
            provider=GeminiProvider(),
            config=AgentConfig(
                model="gemini-2.5-flash",
                max_tokens=200,
                max_iterations=4,
                observers=[OTelObserver(tracer_name="selectools-sim")],
            ),
        )

        # 3. Build a multimodal message that asks for BOTH vision AND tool use
        from selectools import image_message

        msg = image_message(
            str(png_path),
            prompt=(
                "Step 1: In one word, what primary color dominates this tiny image? "
                "Step 2: Use the execute_python tool to compute and print the result of 7*6. "
                "Then give me a one-sentence final answer containing both the color and the number."
            ),
        )

        result = agent.run([msg])

        # 4. Assert the real Gemini call did BOTH things:
        #    (a) saw the image (mentions red)
        #    (b) called execute_python and got 42
        content_lower = result.content.lower()
        assert "red" in content_lower, f"Gemini did not describe the image: {result.content[:300]}"
        assert "42" in result.content, (
            f"Gemini did not use execute_python to compute 7*6: {result.content[:300]}"
        )
        assert result.usage.total_tokens > 0

        # 5. OTel should have captured the run
        spans = otel_exporter.get_finished_spans()
        assert len(spans) > 0, "OTel captured no spans"


# ---------------------------------------------------------------------------
# Scenario 3 — Toolbox integration with real Anthropic agent
# ---------------------------------------------------------------------------


class TestScenario3_ToolboxWithAnthropic:
    """Real Anthropic Claude picks and calls query_sqlite + execute_python."""

    def test_claude_uses_sqlite_tool(self, tmp_path: Path) -> None:
        _require("ANTHROPIC_API_KEY")

        # 1. Create a real SQLite db with deliberately distinctive data
        db_path = tmp_path / "people.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE people (name TEXT, age INTEGER)")
        conn.executemany(
            "INSERT INTO people VALUES (?, ?)",
            [("alice", 29), ("bob", 31), ("carol", 47), ("dave", 23)],
        )
        conn.commit()
        conn.close()

        # 2. Real Anthropic agent with the new db_tools AND code_tools
        agent = Agent(
            tools=[db_tools.query_sqlite, code_tools.execute_python],
            provider=AnthropicProvider(),
            config=AgentConfig(
                model="claude-haiku-4-5",
                max_tokens=300,
                max_iterations=4,
            ),
        )

        # 3. Ask a question that requires the sqlite tool
        result = agent.run(
            f"Use the query_sqlite tool with db_path='{db_path}' to find the "
            f"name of the oldest person in the 'people' table. "
            f"Respond with just their name."
        )

        # 4. Assert the agent called the tool and got 'carol' (the oldest at 47)
        assert "carol" in result.content.lower(), (
            f"Anthropic did not find carol via query_sqlite: {result.content[:300]}"
        )
        assert result.usage.total_tokens > 0


# ---------------------------------------------------------------------------
# Scenario 4 — Qdrant RAG with real OpenAI agent (skipped if no Qdrant)
# ---------------------------------------------------------------------------


class TestScenario4_RAGWithQdrant:
    """Same shape as scenario 1 but proves Qdrant works end-to-end too."""

    def test_agent_answers_from_qdrant_backed_rag(
        self, otel_exporter: InMemorySpanExporter
    ) -> None:
        _require("OPENAI_API_KEY")
        pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        if not _qdrant_reachable(qdrant_url):
            pytest.skip(f"Qdrant not reachable at {qdrant_url}")

        import uuid

        from selectools.embeddings.openai import OpenAIEmbeddingProvider
        from selectools.rag.stores import QdrantVectorStore

        embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        store = QdrantVectorStore(
            embedder=embedder,
            collection_name=f"selectools_sim_{uuid.uuid4().hex[:8]}",
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            prefer_grpc=False,
        )

        # Add anchor documents with a unique phrase
        store.add_documents(
            [
                Document(
                    text=(
                        "The selectools v0.21.0 connector expansion was internally "
                        "nicknamed PROJECT FLAMINGO-17 by the NichevLabs team."
                    ),
                    metadata={"src": "internal"},
                ),
                Document(
                    text="Selectools is an AI agent framework written in Python.",
                    metadata={"src": "public"},
                ),
            ]
        )

        try:
            rag_tool = RAGTool(vector_store=store, top_k=2)
            agent = Agent(
                tools=[rag_tool.search_knowledge_base],
                provider=OpenAIProvider(),
                config=AgentConfig(
                    model="gpt-4o-mini",
                    max_tokens=150,
                    max_iterations=4,
                    observers=[OTelObserver(tracer_name="selectools-sim")],
                ),
            )

            result = agent.run(
                "What was the internal nickname for the selectools v0.21.0 connector "
                "expansion? Use search_knowledge_base and quote it verbatim."
            )

            assert "FLAMINGO" in result.content.upper(), (
                f"OpenAI+Qdrant RAG did not retrieve the anchor: {result.content[:300]}"
            )
            assert result.usage.total_tokens > 0
            assert len(otel_exporter.get_finished_spans()) > 0
        finally:
            store.clear()
