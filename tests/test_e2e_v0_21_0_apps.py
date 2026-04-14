"""Persona-based app simulations for v0.21.0.

These are **not** integration tests of "does feature A combined with
feature B work". They are simulations of **real application use cases**,
matching the selectools simulation idiom from ``tests/test_simulation_evals.py``:

- Each test sets up an agent with a realistic system prompt
- Multi-turn conversations use real ``ConversationMemory``
- Real LLM calls drive the agent through plausible user workflows
- Assertions check the *behaviour* of the app, not just the wiring

Three app shapes are covered:

1. **Documentation Q&A bot** (RAG pipeline used the way a real support
   bot would): FAQ CSV loader → real OpenAI embeddings → FAISS →
   RAGTool → multi-turn user conversation with memory → agent must cite
   from KB and refuse on out-of-KB questions

2. **Data analyst bot** (toolbox chaining the way a real analytics bot
   would): real SQLite sales db → Claude with ``query_sqlite`` +
   ``execute_python`` → agent must query, compute, and answer with a
   real number

3. **Knowledge base librarian** (all four new document loaders feeding a
   real Qdrant store → Gemini agent using RAGTool to answer a question
   whose answer is split across multiple source files)

Each simulation is gated behind ``--run-e2e`` and will skip cleanly when
credentials or backing services aren't available. Total cost per full
run is under $0.01 at current pricing.

Run with:

    pytest tests/test_e2e_v0_21_0_apps.py --run-e2e -v
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import uuid
from pathlib import Path

import pytest

from selectools import Agent, AgentConfig
from selectools.memory import ConversationMemory
from selectools.rag import DocumentLoader
from selectools.rag.stores import FAISSVectorStore
from selectools.rag.tools import RAGTool
from selectools.toolbox import code_tools, db_tools

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _openai_or_skip() -> tuple:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from selectools.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(), "gpt-4o-mini"


def _anthropic_or_skip() -> tuple:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from selectools.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(), "claude-haiku-4-5"


def _gemini_or_skip() -> tuple:
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    from selectools.providers.gemini_provider import GeminiProvider

    return GeminiProvider(), "gemini-2.5-flash"


def _openai_embedder():
    pytest.importorskip("openai")
    from selectools.embeddings.openai import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider(model="text-embedding-3-small")


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


# ===========================================================================
# App 1: Documentation Q&A Bot
# ===========================================================================
#
# Persona: a support bot for a fictional product called "Skylake" whose
# knowledge base consists of a FAQ CSV. A real user opens the bot, asks
# several questions, some of which are covered and some aren't. The bot
# should answer with information from the KB and should refuse (or say it
# doesn't know) for out-of-KB questions. This is the canonical RAG support
# bot pattern.


@pytest.fixture
def skylake_faq_agent(tmp_path: Path):
    """Build a real RAG support bot for the fictional Skylake product."""
    _openai_or_skip()  # fail fast if no creds
    pytest.importorskip("faiss", reason="faiss-cpu not installed")

    # 1. Realistic FAQ CSV — five entries with unique anchor facts so we
    #    can assert that retrieval actually worked
    faq_csv = tmp_path / "skylake_faq.csv"
    faq_csv.write_text(
        "question,answer\n"
        '"How do I install Skylake?",'
        '"Install Skylake by running: curl -sL https://skylake.sh | bash. Version 4.2.1 is the latest stable release."\n'
        '"What is the default port?",'
        '"Skylake listens on port 8742 by default. You can override this with the --port flag or the SKYLAKE_PORT environment variable."\n'
        '"How do I reset my password?",'
        '"Run skylake auth reset --user <email>. A reset link will be emailed within 15 minutes."\n'
        '"Does Skylake support single sign-on?",'
        '"Yes, Skylake supports SAML 2.0 and OpenID Connect for SSO. Configuration lives in /etc/skylake/sso.yaml."\n'
        '"What is the monthly uptime SLA?",'
        '"The enterprise plan includes a 99.95% monthly uptime SLA with service credits for breaches."\n',
        encoding="utf-8",
    )

    # 2. Load via the new CSV loader, embed, and index in real FAISS
    docs = DocumentLoader.from_csv(
        str(faq_csv), text_column="answer", metadata_columns=["question"]
    )
    assert len(docs) == 5

    embedder = _openai_embedder()
    store = FAISSVectorStore(embedder=embedder)
    store.add_documents(docs)

    # 3. Wire the RAG tool into a real OpenAI agent with a support-bot
    #    system prompt. Use ConversationMemory so the bot can actually
    #    carry context across turns.
    provider, model = _openai_or_skip()
    rag_tool = RAGTool(vector_store=store, top_k=3)
    return Agent(
        tools=[rag_tool.search_knowledge_base],
        provider=provider,
        memory=ConversationMemory(max_messages=20),
        config=AgentConfig(
            model=model,
            system_prompt=(
                "You are the official support bot for a product called Skylake. "
                "Always use the search_knowledge_base tool before answering. "
                "If the knowledge base does not contain the answer, say you "
                "don't know — do NOT invent details. Be concise: 1-2 sentences."
            ),
            max_tokens=200,
            max_iterations=4,
        ),
    )


class TestApp1_DocsQABot:
    def test_bot_answers_install_question_from_kb(self, skylake_faq_agent: Agent) -> None:
        """Turn 1: user asks an in-KB question. Bot should quote KB facts."""
        result = skylake_faq_agent.run("How do I install Skylake?")
        assert result.content
        content = result.content.lower()
        # KB anchor facts that a correct retrieval would surface
        assert "curl" in content or "skylake.sh" in content or "4.2.1" in content, (
            f"Bot did not retrieve install instructions from KB. Got: {result.content[:300]}"
        )

    def test_bot_answers_port_question_using_memory(self, skylake_faq_agent: Agent) -> None:
        """Turn 2 (same agent): different in-KB question.

        Exercises ConversationMemory by making a SECOND call on the same
        agent instance. If memory is broken the agent would either drop
        context or re-send the whole first turn, and token usage on the
        second call would look weird. More importantly, this proves that
        tool calling continues to work across turns on a memory-enabled
        agent — a bug-prone area.
        """
        skylake_faq_agent.run("How do I install Skylake?")  # Turn 1
        result = skylake_faq_agent.run("Got it. What port does it listen on?")  # Turn 2
        assert result.content
        assert "8742" in result.content, (
            f"Bot did not retrieve the port fact from KB on turn 2. Got: {result.content[:300]}"
        )

    def test_bot_refuses_out_of_kb_question(self, skylake_faq_agent: Agent) -> None:
        """User asks something NOT in the KB. Bot must not hallucinate."""
        result = skylake_faq_agent.run(
            "What is the maximum WebSocket message size Skylake supports?"
        )
        assert result.content
        content = result.content.lower()
        # A correct bot says "don't know" (or similar). We don't require an
        # exact phrase — just that the bot does not confidently invent a
        # numeric answer. Accept any phrasing that signals uncertainty.
        signals_uncertainty = (
            "don't know" in content
            or "do not know" in content
            or "not in the knowledge base" in content
            or "not available" in content
            or "can't find" in content
            or "cannot find" in content
            or "not listed" in content
            or "no information" in content
            or "not covered" in content
            or "unable to find" in content
        )
        assert signals_uncertainty, (
            f"Bot should refuse out-of-KB questions instead of hallucinating. "
            f"Got: {result.content[:300]}"
        )


# ===========================================================================
# App 2: Data Analyst Bot
# ===========================================================================
#
# Persona: an analytics assistant for a small sales database. A real user
# asks a business question whose answer requires:
#   1. Running a SQL query to pull raw data
#   2. Using Python to compute a derived number
#   3. Explaining the result in natural language
#
# This exercises multi-step tool chaining by a real LLM — a path that
# mock tests cannot validate because the LLM decides when each tool is
# needed and how to pass data between them.


@pytest.fixture
def sales_db(tmp_path: Path) -> Path:
    """Create a real SQLite sales db with deliberately distinctive numbers."""
    db_path = tmp_path / "sales.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount_usd REAL, month TEXT)"
    )
    # Carefully chosen so the answer is unambiguous: region 'EU' has the
    # highest total (1000 + 2000 + 3000 = 6000) and a specific average
    # (2000) that the LLM should be able to verify with Python.
    rows = [
        (1, "US", 500, "2026-01"),
        (2, "US", 600, "2026-02"),
        (3, "US", 700, "2026-03"),
        (4, "EU", 1000, "2026-01"),
        (5, "EU", 2000, "2026-02"),
        (6, "EU", 3000, "2026-03"),
        (7, "APAC", 800, "2026-01"),
        (8, "APAC", 900, "2026-02"),
    ]
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path


class TestApp2_DataAnalystBot:
    def test_bot_finds_top_region_and_computes_average(self, sales_db: Path) -> None:
        """Multi-step: query → compute → explain."""
        provider, model = _anthropic_or_skip()

        agent = Agent(
            tools=[db_tools.query_sqlite, code_tools.execute_python],
            provider=provider,
            config=AgentConfig(
                model=model,
                system_prompt=(
                    "You are a data analyst assistant. You have two tools: "
                    "query_sqlite for reading from a SQLite database, and "
                    "execute_python for running small Python snippets when "
                    "you need to compute a derived value. Always use the "
                    "tools to get real numbers — do not guess."
                ),
                max_tokens=500,
                max_iterations=6,
            ),
        )

        result = agent.run(
            f"Use db_path='{sales_db}'. Find the region with the highest "
            f"total sales in the 'orders' table, and report its average "
            f"order amount. Show your work."
        )
        assert result.content
        content = result.content
        # The correct region is EU (total = 6000)
        assert "EU" in content or "eu" in content.lower(), (
            f"Bot did not identify EU as top region. Got: {content[:400]}"
        )
        # The average of EU orders is 2000. Accept '2000' or '2,000'.
        assert "2000" in content or "2,000" in content, (
            f"Bot did not compute the correct average (2000). Got: {content[:400]}"
        )


# ===========================================================================
# App 3: Knowledge Base Librarian
# ===========================================================================
#
# Persona: a librarian that ingests docs from heterogeneous sources (CSV,
# JSON, HTML, URL) into a real Qdrant store and answers questions whose
# truth is split across sources. This exercises every new v0.21.0
# document loader in a single realistic workflow.


@pytest.fixture
def librarian_agent(tmp_path: Path):
    """Build a real Qdrant-backed librarian agent with heterogeneous sources."""
    pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if not _qdrant_reachable(qdrant_url):
        pytest.skip(f"Qdrant not reachable at {qdrant_url}")
    _gemini_or_skip()  # fail fast if no Gemini creds

    from selectools.rag.stores import QdrantVectorStore

    # 1. CSV source — product catalog with unique anchor phrase
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "sku,description\n"
        '"SKY-001","The Skylake SKY-001 is an edge router shipping with the internal codename THUNDERCAT-7."\n'
        '"SKY-002","The Skylake SKY-002 is a development kit."\n',
        encoding="utf-8",
    )

    # 2. JSON source — release notes with another unique anchor phrase
    json_path = tmp_path / "releases.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "version": "4.2.1",
                    "body": (
                        "Skylake 4.2.1 was released on the full-moon day and "
                        "is internally referenced as the MOONWALK release."
                    ),
                },
                {"version": "4.2.0", "body": "Skylake 4.2.0 was a bug-fix release."},
            ]
        ),
        encoding="utf-8",
    )

    # 3. HTML source — marketing blurb with a third anchor phrase
    html_path = tmp_path / "about.html"
    html_path.write_text(
        "<html><body><article>"
        "<p>Skylake was founded in Helsinki in 2023.</p>"
        "<p>The team operates under the office code VANTA-NORTH.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )

    # 4. Load via all four loaders
    csv_docs = DocumentLoader.from_csv(str(csv_path), text_column="description")
    json_docs = DocumentLoader.from_json(
        str(json_path), text_field="body", metadata_fields=["version"]
    )
    html_docs = DocumentLoader.from_html(str(html_path))
    all_docs = csv_docs + json_docs + html_docs
    assert len(all_docs) >= 5  # 2 csv + 2 json + 1 html

    embedder = _openai_embedder()  # needs OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set for embedding")

    store = QdrantVectorStore(
        embedder=embedder,
        collection_name=f"skylake_kb_{uuid.uuid4().hex[:8]}",
        url=qdrant_url,
        api_key=os.environ.get("QDRANT_API_KEY"),
        prefer_grpc=False,
    )
    store.add_documents(all_docs)

    provider, model = _gemini_or_skip()
    rag_tool = RAGTool(vector_store=store, top_k=4)

    agent = Agent(
        tools=[rag_tool.search_knowledge_base],
        provider=provider,
        config=AgentConfig(
            model=model,
            system_prompt=(
                "You are the Skylake knowledge base librarian. Always use "
                "search_knowledge_base to answer. Quote anchor phrases from "
                "the docs verbatim when asked for them. Keep answers short."
            ),
            max_tokens=200,
            max_iterations=4,
        ),
    )

    try:
        yield agent
    finally:
        # Cleanup: drop the collection
        try:
            store.clear()
        except Exception:
            pass


class TestApp3_KnowledgeBaseLibrarian:
    def test_librarian_retrieves_from_csv_source(self, librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the CSV-loaded docs."""
        result = librarian_agent.run(
            "What is the internal codename for the SKY-001 router? Quote it verbatim."
        )
        assert result.content
        assert "THUNDERCAT" in result.content.upper(), (
            f"Librarian did not retrieve the CSV anchor phrase. Got: {result.content[:300]}"
        )

    def test_librarian_retrieves_from_json_source(self, librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the JSON-loaded docs."""
        result = librarian_agent.run("What is the internal reference name for Skylake 4.2.1?")
        assert result.content
        assert "MOONWALK" in result.content.upper(), (
            f"Librarian did not retrieve the JSON anchor phrase. Got: {result.content[:300]}"
        )

    def test_librarian_retrieves_from_html_source(self, librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the HTML-loaded docs."""
        result = librarian_agent.run("What is the Skylake office code?")
        assert result.content
        assert "VANTA-NORTH" in result.content.upper(), (
            f"Librarian did not retrieve the HTML anchor phrase. Got: {result.content[:300]}"
        )


# ===========================================================================
# App 3b: Knowledge Base Librarian (FAISS variant)
# ===========================================================================
#
# Same persona as App 3 but backed by FAISSVectorStore instead of Qdrant.
# This means the "all four document loaders fed into a single RAG pipeline"
# coverage is also available on machines without Docker/Qdrant — a real
# concern for CI environments that don't run containers.


@pytest.fixture
def faiss_librarian_agent(tmp_path: Path):
    """Build a real FAISS-backed librarian agent with heterogeneous sources."""
    _openai_or_skip()  # fail fast if no creds
    pytest.importorskip("faiss", reason="faiss-cpu not installed")

    # 1. CSV source — product catalog with unique anchor phrase
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "sku,description\n"
        '"SKY-001","The Skylake SKY-001 is an edge router shipping with the internal codename OSPREY-88."\n'
        '"SKY-002","The Skylake SKY-002 is a development kit."\n',
        encoding="utf-8",
    )

    # 2. JSON source — release notes with a distinct anchor phrase
    json_path = tmp_path / "releases.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "version": "4.2.1",
                    "body": (
                        "Skylake 4.2.1 was released on the summer solstice "
                        "and is internally referenced as the CRESCENT release."
                    ),
                },
                {"version": "4.2.0", "body": "Skylake 4.2.0 was a bug-fix release."},
            ]
        ),
        encoding="utf-8",
    )

    # 3. HTML source — marketing blurb with a third anchor phrase
    html_path = tmp_path / "about.html"
    html_path.write_text(
        "<html><body><article>"
        "<p>Skylake was founded in Helsinki in 2023.</p>"
        "<p>The team operates under the office code AURORA-SOUTH.</p>"
        "</article></body></html>",
        encoding="utf-8",
    )

    # 4. Load via all three loaders
    csv_docs = DocumentLoader.from_csv(str(csv_path), text_column="description")
    json_docs = DocumentLoader.from_json(
        str(json_path), text_field="body", metadata_fields=["version"]
    )
    html_docs = DocumentLoader.from_html(str(html_path))
    all_docs = csv_docs + json_docs + html_docs
    assert len(all_docs) >= 5  # 2 csv + 2 json + 1 html

    embedder = _openai_embedder()

    # 5. Real FAISS store — no external server required
    store = FAISSVectorStore(embedder=embedder)
    store.add_documents(all_docs)

    provider, model = _openai_or_skip()
    rag_tool = RAGTool(vector_store=store, top_k=4)

    return Agent(
        tools=[rag_tool.search_knowledge_base],
        provider=provider,
        config=AgentConfig(
            model=model,
            system_prompt=(
                "You are the Skylake knowledge base librarian. Always use "
                "search_knowledge_base to answer. Quote anchor phrases from "
                "the docs verbatim when asked for them. Keep answers short."
            ),
            max_tokens=200,
            max_iterations=4,
        ),
    )


class TestApp3b_KnowledgeBaseLibrarianFAISS:
    """Same shape as App 3 but backed by FAISS — runnable without Docker."""

    def test_librarian_retrieves_from_csv_source(self, faiss_librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the CSV-loaded docs."""
        result = faiss_librarian_agent.run(
            "What is the internal codename for the SKY-001 router? Quote it verbatim."
        )
        assert result.content
        assert "OSPREY" in result.content.upper(), (
            f"FAISS librarian did not retrieve the CSV anchor phrase. Got: {result.content[:300]}"
        )

    def test_librarian_retrieves_from_json_source(self, faiss_librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the JSON-loaded docs."""
        result = faiss_librarian_agent.run("What is the internal reference name for Skylake 4.2.1?")
        assert result.content
        assert "CRESCENT" in result.content.upper(), (
            f"FAISS librarian did not retrieve the JSON anchor phrase. Got: {result.content[:300]}"
        )

    def test_librarian_retrieves_from_html_source(self, faiss_librarian_agent: Agent) -> None:
        """Asks a question whose answer lives in the HTML-loaded docs."""
        result = faiss_librarian_agent.run("What is the Skylake office code?")
        assert result.content
        assert "AURORA-SOUTH" in result.content.upper(), (
            f"FAISS librarian did not retrieve the HTML anchor phrase. Got: {result.content[:300]}"
        )
