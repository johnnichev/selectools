# Selectools

[![PyPI version](https://badge.fury.io/py/selectools.svg)](https://badge.fury.io/py/selectools)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://johnnichev.github.io/selectools)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Evaluators](https://img.shields.io/badge/evaluators-50-06b6d4.svg)](https://johnnichev.github.io/selectools/modules/EVALS/)

An open-source project from **[NichevLabs](https://nichevlabs.com)**.

**Multi-agent orchestration in plain Python.** Build agent graphs, compose pipelines with `|`, deploy with one command. No DSL, no compile step, no paid debugger. Works with OpenAI, Anthropic, Gemini, and Ollama.

### 3 Ways to Build

```python
# 1. Single agent — 5 lines
agent = Agent(tools=[search, calculate], provider=OpenAIProvider())
result = agent.run("What is 15 * 7?")

# 2. Multi-agent graph — 1 line
result = AgentGraph.chain(planner, writer, reviewer).run("Write a blog post")

# 3. Deploy — 1 command
# selectools serve agent.yaml
```

## What's New in v0.19

### v0.19.2 — Enterprise Hardening

```python
from selectools.stability import stable, beta, deprecated
from selectools import trace_to_html

# Mark your own extensions with stability levels
@stable
class MyProductionAgent: ...

@beta
class MyExperimentalFeature: ...

@deprecated(since="0.19", replacement="MyProductionAgent")
class MyOldAgent: ...

# Visualise any trace as a waterfall HTML timeline
Path("trace.html").write_text(trace_to_html(result.trace))
```

- **Stability markers** — `@stable`, `@beta`, `@deprecated(since, replacement)` for public API signalling
- **Trace HTML viewer** — `trace_to_html(trace)` renders a standalone waterfall timeline
- **Deprecation policy** — 2-minor-version window, programmatic introspection via `.__stability__`
- **Security audit** — all 41 `# nosec` annotations reviewed and published in `docs/SECURITY.md`
- **Quality infrastructure** — property-based tests (Hypothesis), thread-safety smoke suite, 5 new production simulations (3135 tests total)

### v0.19.1 — Advanced Agent Patterns

```python
from selectools.patterns import PlanAndExecuteAgent, ReflectiveAgent, DebateAgent, TeamLeadAgent

# PlanAndExecute — planner generates typed steps, executor runs them sequentially
agent = PlanAndExecuteAgent(planner=planner, executor=executor, provider=provider)
result = agent.run("Research and write a blog post about LLM safety")

# ReflectiveAgent — actor drafts, critic reviews, actor revises until approved
agent = ReflectiveAgent(actor=actor, critic=critic, provider=provider, max_reflections=3)
result = agent.run("Draft a product announcement email")

# DebateAgent — multiple agents argue, judge synthesizes conclusion
agent = DebateAgent(agents={"optimist": opt, "skeptic": skep}, judge=judge, provider=provider)
result = agent.run("Should we migrate our infrastructure to microservices?")

# TeamLeadAgent — lead delegates subtasks, team executes in parallel or sequentially
agent = TeamLeadAgent(lead=lead, team={"researcher": r, "writer": w}, provider=provider)
result = agent.run("Produce a competitive analysis report")
```

- **PlanAndExecuteAgent** — Typed `PlanStep` list; optional replanning on step failure
- **ReflectiveAgent** — Actor–critic loop with `ReflectionRound` records per revision
- **DebateAgent** — N-agent debate with transcript, judge synthesis, `DebateResult`
- **TeamLeadAgent** — `sequential`, `parallel`, or `dynamic` delegation strategies

### v0.19.0 — Serve, Deploy & Complete Composition

```python
# One command deploys your agent over HTTP with SSE streaming
# selectools serve agent.yaml

# Compose tools into a single callable
from selectools import compose
search_and_summarize = compose(search_web, summarize)

# Streaming composition
async for chunk in pipeline.astream("input"):
    print(chunk)
```

- **`selectools serve`** — HTTP deployment with SSE streaming, Playground UI, `/health`, `/schema`
- **YAML config** — `AgentConfig.from_yaml("agent.yaml")`, 5 built-in templates
- **`compose()`** — Chain tools into composite tool; `retry()` and `cache_step()` wrappers
- **PostgresCheckpointStore** — Durable graph checkpointing backed by PostgreSQL

<details>
<summary><strong>v0.18.x highlights</strong></summary>

### v0.18.0 — Multi-Agent Orchestration

```python
from selectools import AgentGraph, SupervisorAgent, AgentConfig, OpenAIProvider, tool

# Build a multi-agent graph in plain Python — no DSL, no compile step
graph = AgentGraph()
graph.add_node("planner", planner_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)
graph.add_edge("planner", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", AgentGraph.END)
graph.set_entry("planner")
result = graph.run("Write a blog post about AI safety")

# Or use SupervisorAgent for automatic coordination
supervisor = SupervisorAgent(
    agents={"researcher": researcher, "writer": writer},
    provider=OpenAIProvider(),
    strategy="plan_and_execute",  # also: round_robin, dynamic, magentic
)
result = supervisor.run("Write a comprehensive report on LLM safety")
```

- **AgentGraph** — Directed graph of agent nodes with plain Python routing
- **4 Supervisor Strategies** — plan_and_execute, round_robin, dynamic, magentic (Magentic-One pattern)
- **Human-in-the-Loop** — Generator nodes with `yield InterruptRequest()` — resumes at exact yield point (LangGraph restarts the whole node)
- **Parallel Execution** — `add_parallel_nodes()` with 3 merge policies (LAST_WINS, FIRST_WINS, APPEND)
- **Checkpointing** — 3 backends (InMemory, File, SQLite) for durable mid-graph persistence
- **Subgraph Composition** — Nest graphs inside graphs with explicit state mapping
- **ModelSplit** — Separate planner/executor models for 70-90% cost reduction
- **Loop & Stall Detection** — State hash tracking with observer events
- **10 New StepTypes** — Full trace visibility into graph execution
- **13 New Observer Events** — on_graph_start/end, on_node_start/end, on_graph_interrupt/resume, and more

### v0.18.0 — Composable Pipelines

```python
from selectools import Pipeline, step, parallel, branch

@step
def summarize(text: str) -> str:
    return agent.run(f"Summarize: {text}").content

@step
def translate(text: str, lang: str = "es") -> str:
    return agent.run(f"Translate to {lang}: {text}").content

# Compose with | operator
pipeline = summarize | translate
result = pipeline.run("Long article text here...")

# Fan-out to multiple steps, merge results
research = parallel(search_web, search_docs, search_db)

# Conditional branching
route = branch(
    lambda x: "technical" if "code" in x else "general",
    technical=code_review_pipeline,
    general=summarize_pipeline,
)
```

- **Pipeline** — Chain steps sequentially with `|` operator or `Pipeline(steps=[...])`
- **@step decorator** — Wrap any sync/async callable into a composable pipeline step
- **parallel()** — Fan-out to multiple steps and merge results
- **branch()** — Conditional routing based on input data

</details>

<details>
<summary><strong>v0.17.x highlights</strong></summary>

### v0.17.7 — Caching & Context

```python
from selectools.cache_semantic import SemanticCache
from selectools.embeddings.openai import OpenAIEmbeddingProvider

# Semantic cache — cache hits for paraphrased queries
cache = SemanticCache(
    embedding_provider=OpenAIEmbeddingProvider(),
    similarity_threshold=0.92,
)
config = AgentConfig(cache=cache)
# "Weather in NYC?" hits cache for "What's the weather in New York City?"

# Prompt compression — prevent context-window overflow
config = AgentConfig(
    compress_context=True,
    compress_threshold=0.75,   # trigger at 75 % context fill
    compress_keep_recent=4,    # keep last 4 turns verbatim
)

# Conversation branching — fork history for A/B exploration
branch = agent.memory.branch()   # independent snapshot
store.branch("main", "experiment")  # fork a persisted session
```

### v0.17.6 — Quick Wins

```python
from selectools import AgentConfig, REASONING_STRATEGIES, tool

# Reasoning strategies — guide the LLM's thought process
config = AgentConfig(reasoning_strategy="react")   # Thought → Action → Observation
config = AgentConfig(reasoning_strategy="cot")      # Chain-of-Thought step-by-step
config = AgentConfig(reasoning_strategy="plan_then_act")  # Plan first, then execute

# Tool result caching — skip re-execution for identical calls
@tool(description="Search the web", cacheable=True, cache_ttl=60)
def web_search(query: str) -> str:
    return expensive_api_call(query)
```

Also: Python 3.9–3.13 CI matrix (verified zero compatibility issues).

</details>

<details>
<summary><strong>v0.17.4 and earlier</strong></summary>

### v0.17.4 — Agent Intelligence

```python
from selectools import AgentConfig, estimate_run_tokens, KnowledgeMemory, SQLiteKnowledgeStore

# Pre-execution token estimation
estimate = estimate_run_tokens(messages, tools, system_prompt, model="gpt-4o")
print(f"{estimate.total_tokens} tokens, {estimate.remaining_tokens} remaining")

# Model switching — cheap for tools, expensive for reasoning
config = AgentConfig(
    model="claude-haiku-4-5",
    model_selector=lambda i, tc, u: "claude-sonnet-4-6" if i > 2 else "claude-haiku-4-5",
)

# Knowledge memory with pluggable stores and importance scoring
memory = KnowledgeMemory(store=SQLiteKnowledgeStore("knowledge.db"), max_entries=50)
memory.remember("User prefers dark mode", category="preference", importance=0.9, ttl_days=30)
```

### v0.17.3 — Agent Runtime Controls

```python
from selectools import AgentConfig, CancellationToken, SimpleStepObserver
from selectools.tools import tool

# Token/cost budget — stop before burning money
config = AgentConfig(max_total_tokens=50000, max_cost_usd=0.20)

# Cooperative cancellation from any thread
token = CancellationToken()
result = await agent.arun("long task", cancel_token=token)
# token.cancel()  ← from UI handler, supervisor, timeout manager

# Per-tool approval gate
@tool(requires_approval=True, description="Send email to customer")
def send_email(to: str, subject: str, body: str) -> str: ...

# Single-callback observer for SSE streaming
config = AgentConfig(observers=[SimpleStepObserver(
    lambda event, run_id, **data: sse_send({"type": event, **data})
)])
```

### v0.17.1 — MCP Client/Server

```python
from selectools.mcp import mcp_tools, MCPServerConfig

with mcp_tools(MCPServerConfig(command="python", args=["server.py"])) as tools:
    agent = Agent(provider=provider, tools=tools, config=config)
```

- **MCPClient** — stdio + HTTP transport, circuit breaker, retry, tool caching
- **MultiMCPClient** — multiple servers, graceful degradation, name prefixing
- **MCPServer** — expose `@tool` functions as MCP server

### v0.17.0 — Built-in Eval Framework

```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(agent=agent, cases=[
    TestCase(input="Cancel account", expect_tool="cancel_sub", expect_no_pii=True),
    TestCase(input="Balance?", expect_contains="balance", expect_latency_ms_lte=500),
])
report = suite.run()
report.to_html("report.html")
```

- **50 Evaluators** — 30 deterministic + 21 LLM-as-judge
- **A/B Testing**, regression detection, snapshot testing
- **HTML reports**, JUnit XML, CLI, GitHub Action integration

</details>

> Full changelog: [CHANGELOG.md](https://github.com/johnnichev/selectools/blob/main/CHANGELOG.md)

<details>
<summary><strong>v0.16.x highlights</strong></summary>

- **v0.16.6**: Gemini 3.x thought_signature crash fix — base64 round-trip for non-UTF-8 binary signatures
- **v0.16.5**: Design Patterns & Code Quality — terminal actions, async observers, Gemini 3.x thought signatures, agent decomposition, hooks deprecated
- **v0.16.4**: Parallel execution safety — coherence + screening in parallel, guardrail immutability, streaming usage tracking
- **v0.16.0**: Memory & Persistence — persistent sessions (3 backends), summarize-on-trim, entity memory, knowledge graph

</details>

<details>
<summary><strong>v0.15.x highlights</strong></summary>

- **v0.15.0**: Enterprise Reliability — Guardrails engine (5 built-in), audit logging (4 privacy levels), tool output screening (15 patterns), coherence checking

</details>

<details>
<summary><strong>v0.14.x highlights</strong></summary>

- **v0.14.1**: Critical streaming fix — 13 bugs fixed across all providers; 141 new tests (total: 1100)
- **v0.14.0**: AgentObserver Protocol (25 events), 145 models with March 2026 pricing, OpenAI `max_completion_tokens` auto-detection, 11 bug fixes

</details>

## Coming from LangChain?

| LangChain/LangGraph | selectools |
|---|---|
| `StateGraph` + `add_node` + `add_edge` + `compile()` | `AgentGraph.chain(a, b, c).run(prompt)` |
| LCEL `prompt \| llm \| parser` with Runnable protocol | `@step` + `\|` on plain functions |
| `interrupt()` restarts the whole node on resume | `yield InterruptRequest()` resumes at yield point |
| LangSmith (paid) for tracing and evals | Built-in: 50 evaluators + traces, zero cost |
| 5+ packages (`langchain-core`, `langgraph`, `langsmith`...) | 1 package: `pip install selectools` |
| `langserve` for deployment | `selectools serve agent.yaml` |

> Full migration guide with code examples: **[Coming from LangChain](docs/MIGRATION.md)**

## Why Selectools

| Capability | What You Get |
|---|---|
| **Provider Agnostic** | Switch between OpenAI, Anthropic, Gemini, Ollama with one line. Your tools stay identical. |
| **Structured Output** | Pydantic or JSON Schema `response_format` with auto-retry on validation failure. |
| **Execution Traces** | Every `run()` returns `result.trace` — structured timeline of LLM calls, tool picks, and executions. |
| **Reasoning Visibility** | `result.reasoning` surfaces *why* the agent chose a tool, extracted from LLM responses. |
| **Provider Fallback** | `FallbackProvider` tries providers in priority order with circuit breaker on failure. |
| **Batch Processing** | `agent.batch()` / `agent.abatch()` for concurrent multi-prompt classification. |
| **Tool Policy Engine** | Declarative allow/review/deny rules with glob patterns. Human-in-the-loop approval callbacks. |
| **Hybrid Search** | BM25 keyword + vector semantic search with RRF/weighted fusion and cross-encoder reranking. |
| **Advanced Chunking** | Fixed, recursive, semantic (embedding-based), and contextual (LLM-enriched) chunking strategies. |
| **E2E Streaming** | Token-level `astream()` with native tool call support. Parallel tool execution via `asyncio.gather`. |
| **Dynamic Tools** | Load tools from files/directories at runtime. Add, remove, replace tools without restarting. |
| **Response Caching** | LRU + TTL in-memory cache and Redis backend. Avoid redundant LLM calls for identical requests. |
| **Routing Mode** | Agent selects a tool without executing it. Use for intent classification and request routing. |
| **Guardrails Engine** | Input/output validation pipeline with PII redaction, topic blocking, toxicity detection, and format enforcement. |
| **Audit Logging** | JSONL audit trail with privacy controls (redact, hash, omit) and daily rotation. |
| **Tool Output Screening** | Prompt injection detection with 15 built-in patterns. Per-tool or global. |
| **Coherence Checking** | LLM-based verification that tool calls match user intent — catches injection-driven tool misuse. |
| **Persistent Sessions** | `SessionStore` with JSON file, SQLite, and Redis backends. Auto-save/load with TTL expiry. |
| **Entity Memory** | LLM-based entity extraction with deduplication, LRU pruning, and system prompt injection. |
| **Knowledge Graph** | Relationship triple extraction with in-memory and SQLite storage and keyword-based querying. |
| **Cross-Session Knowledge** | Daily logs + persistent facts with auto-registered `remember` tool. |
| **MCP Integration** | Connect to any MCP tool server (stdio + HTTP). MCPClient, MultiMCPClient, MCPServer. Circuit breaker, retry, graceful degradation. |
| **Eval Framework** | 50 built-in evaluators (30 deterministic + 21 LLM-as-judge). A/B testing, regression detection, snapshot testing, HTML reports, JUnit XML, CI integration. |
| **Multi-Agent Orchestration** | `AgentGraph` for directed agent graphs, `SupervisorAgent` with 4 strategies, HITL via generator nodes, parallel execution, checkpointing, subgraph composition. |
| **Composable Pipelines** | `Pipeline` + `@step` + `|` operator + `parallel()` + `branch()` — chain agents, tools, and transforms with plain Python. |
| **AgentObserver Protocol** | 45-event lifecycle observer with `run_id`/`call_id` correlation. Built-in `LoggingObserver` + `SimpleStepObserver`. |
| **Runtime Controls** | Token/cost budget limits, cooperative cancellation, per-tool approval gates, model switching per iteration. |
| **Production Hardened** | Retries with backoff, per-tool timeouts, iteration caps, cost warnings, observability hooks + observers. |
| **Library-First** | Not a framework. No magic globals, no hidden state. Use as much or as little as you need. |

## What's Included

- **5 LLM Providers**: OpenAI, Anthropic, Gemini, Ollama + FallbackProvider (auto-failover)
- **Structured Output**: Pydantic / JSON Schema `response_format` with auto-retry
- **Execution Traces**: `result.trace` with typed timeline of every agent step
- **Reasoning Visibility**: `result.reasoning` explains *why* the agent chose a tool
- **Batch Processing**: `agent.batch()` / `agent.abatch()` for concurrent classification
- **Tool Policy Engine**: Declarative allow/review/deny rules with human-in-the-loop
- **4 Embedding Providers**: OpenAI, Anthropic/Voyage, Gemini (free!), Cohere
- **4 Vector Stores**: In-memory, SQLite, Chroma, Pinecone
- **Hybrid Search**: BM25 + vector fusion with Cohere/Jina reranking
- **Advanced Chunking**: Semantic + contextual chunking for better retrieval
- **Dynamic Tool Loading**: Plugin system with hot-reload support
- **Response Caching**: InMemoryCache and RedisCache with stats tracking
- **152 Model Registry**: Type-safe constants with pricing and metadata
- **Pre-built Toolbox**: 24 tools for files, data, text, datetime, web
- **Persistent Sessions**: 3 backends (JSON file, SQLite, Redis) with TTL
- **Entity Memory**: LLM-based named entity extraction and tracking
- **Knowledge Graph**: Triple extraction with in-memory and SQLite storage
- **Cross-Session Knowledge**: Daily logs + persistent memory with `remember` tool, pluggable stores (File, SQLite), importance scoring, TTL
- **Token Budget & Cancellation**: `max_total_tokens`, `max_cost_usd` hard limits; `CancellationToken` for cooperative stopping
- **Token Estimation**: `estimate_run_tokens()` for pre-execution budget checks
- **Model Switching**: `model_selector` callback for per-iteration model selection
- **Semantic Cache**: `SemanticCache` — embedding-based cache hits for paraphrased queries (cosine similarity, LRU + TTL)
- **Prompt Compression**: Auto-summarise old history when context window fills up; `compress_context`, `compress_threshold`, `compress_keep_recent`
- **Conversation Branching**: `ConversationMemory.branch()` and `SessionStore.branch()` for A/B exploration and checkpointing
- **Multi-Agent Orchestration**: `AgentGraph` with routing, parallel execution, HITL, checkpointing; `SupervisorAgent` with 4 strategies (plan_and_execute, round_robin, dynamic, magentic)
- **Composable Pipelines**: `Pipeline` + `@step` + `|` operator + `parallel()` + `branch()` — chain agents, tools, and transforms
- **75 Examples**: Multi-agent graphs, RAG, hybrid search, streaming, structured output, traces, batch, policy, observer, guardrails, audit, sessions, entity memory, knowledge graph, eval framework, advanced agent patterns, stability markers, HTML trace viewer, and more
- **Built-in Eval Framework**: 50 evaluators (30 deterministic + 21 LLM-as-judge), A/B testing, regression detection, HTML reports, JUnit XML, snapshot testing
- **AgentObserver Protocol**: 45 lifecycle events with `run_id` correlation, `LoggingObserver`, `SimpleStepObserver`, OTel export
- **3135 Tests**: Unit, integration, regression, and E2E with real API calls

## Install

```bash
pip install selectools                    # Core + basic RAG
pip install selectools[rag]               # + Chroma, Pinecone, Voyage, Cohere, PyPDF
pip install selectools[cache]             # + Redis cache
pip install selectools[mcp]               # + MCP client/server
pip install selectools[rag,cache,mcp]    # Everything
```

Add your provider's API key to a `.env` file in your project root:

```
OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY, GEMINI_API_KEY — whichever provider you use
```

## Quick Start

> **New to Selectools?** Follow the [5-minute Quickstart tutorial](docs/QUICKSTART.md) — no API key needed.

### Tool Calling Agent (No API Key)

```python
from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider

@tool(description="Look up the price of a product")
def get_price(product: str) -> str:
    prices = {"laptop": "$999", "phone": "$699", "headphones": "$149"}
    return prices.get(product.lower(), f"No price found for {product}")

agent = Agent(
    tools=[get_price],
    provider=LocalProvider(),
    config=AgentConfig(max_iterations=3),
)

result = agent.ask("How much is a laptop?")
print(result.content)
```

### Tool Calling Agent (OpenAI)

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.models import OpenAI

@tool(description="Search the web for information")
def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    tools=[search],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    config=AgentConfig(max_iterations=5),
)

result = agent.ask("Search for Python tutorials")
print(result.content)
```

### RAG Agent

```python
from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import RAGAgent, VectorStore

embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
store = VectorStore.create("memory", embedder=embedder)

agent = RAGAgent.from_directory(
    directory="./docs",
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    vector_store=store,
    chunk_size=500, top_k=3,
)

result = agent.ask("What are the main features?")
print(result.content)
print(agent.get_usage_summary())  # LLM + embedding costs
```

### Hybrid Search (Keyword + Semantic)

```python
from selectools.rag import BM25, HybridSearcher, FusionMethod, HybridSearchTool, VectorStore

store = VectorStore.create("memory", embedder=embedder)
store.add_documents(chunked_docs)

searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.6,
    keyword_weight=0.4,
    fusion=FusionMethod.RRF,
)
searcher.add_documents(chunked_docs)

# Use with agent
hybrid_tool = HybridSearchTool(searcher=searcher, top_k=5)
agent = Agent(tools=[hybrid_tool.search_knowledge_base], provider=provider)
```

### Streaming with Parallel Tools

```python
import asyncio
from selectools import Agent, AgentConfig
from selectools.types import StreamChunk, AgentResult

agent = Agent(
    tools=[tool_a, tool_b, tool_c],
    provider=provider,
    config=AgentConfig(parallel_tool_execution=True),  # Default: enabled
)

async for item in agent.astream("Run all tasks"):
    if isinstance(item, StreamChunk):
        print(item.content, end="", flush=True)
    elif isinstance(item, AgentResult):
        print(f"\nDone in {item.iterations} iterations")
```

## Key Features

### Hybrid Search & Reranking

Combine semantic search with BM25 keyword matching for better recall on exact terms, names, and acronyms:

```python
from selectools.rag import BM25, HybridSearcher, CohereReranker, FusionMethod

searcher = HybridSearcher(
    vector_store=store,
    fusion=FusionMethod.RRF,
    reranker=CohereReranker(),  # Optional cross-encoder reranking
)
results = searcher.search("GDPR compliance", top_k=5)
```

See [docs/modules/HYBRID_SEARCH.md](docs/modules/HYBRID_SEARCH.md) for full documentation.

### Advanced Chunking

Go beyond fixed-size splitting with embedding-aware and LLM-enriched chunking:

```python
from selectools.rag import SemanticChunker, ContextualChunker

# Split at topic boundaries using embedding similarity
semantic = SemanticChunker(embedder=embedder, similarity_threshold=0.75)

# Enrich each chunk with LLM-generated context (Anthropic-style contextual retrieval)
contextual = ContextualChunker(base_chunker=semantic, provider=provider)
enriched_docs = contextual.split_documents(documents)
```

See [docs/modules/ADVANCED_CHUNKING.md](docs/modules/ADVANCED_CHUNKING.md) for full documentation.

### Dynamic Tool Loading

Discover and load `@tool` functions from files and directories at runtime:

```python
from selectools.tools import ToolLoader

# Load tools from a plugin directory
tools = ToolLoader.from_directory("./plugins", recursive=True)
agent.add_tools(tools)

# Hot-reload after editing a plugin
updated = ToolLoader.reload_file("./plugins/search.py")
agent.replace_tool(updated[0])

# Remove tools the agent no longer needs
agent.remove_tool("deprecated_search")
```

See [docs/modules/DYNAMIC_TOOLS.md](docs/modules/DYNAMIC_TOOLS.md) for full documentation.

### Response Caching

Avoid redundant LLM calls with pluggable caching:

```python
from selectools import Agent, AgentConfig, InMemoryCache

cache = InMemoryCache(max_size=1000, default_ttl=300)
agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(cache=cache),
)

# Same question twice -> second call is instant (cache hit)
agent.ask("What is Python?")
agent.reset()
agent.ask("What is Python?")

print(cache.stats)  # CacheStats(hits=1, misses=1, hit_rate=50.00%)
```

For distributed setups: `from selectools.cache_redis import RedisCache`

### Routing Mode

Agent selects a tool without executing it -- use for intent classification:

```python
config = AgentConfig(routing_only=True)
agent = Agent(tools=[send_email, schedule_meeting, search_kb], provider=provider, config=config)

result = agent.ask("Book a meeting with Alice tomorrow")
print(result.tool_name)  # "schedule_meeting"
print(result.tool_args)  # {"attendee": "Alice", "date": "tomorrow"}
```

### Structured Output

Get typed, validated results from the LLM:

```python
from pydantic import BaseModel
from typing import Literal

class Classification(BaseModel):
    intent: Literal["billing", "support", "sales", "cancel"]
    confidence: float
    priority: Literal["low", "medium", "high"]

result = agent.ask("I want to cancel my account", response_format=Classification)
print(result.parsed)  # Classification(intent="cancel", confidence=0.95, priority="high")
```

Auto-retries with error feedback when validation fails.

### Execution Traces & Reasoning

See exactly what your agent did and why:

```python
result = agent.run("Classify this ticket")

# Structured timeline of every step
for step in result.trace:
    print(f"{step.type} | {step.duration_ms:.0f}ms | {step.summary}")

# Why the agent chose a tool
print(result.reasoning)  # "Customer is asking about billing, routing to billing_support"

# Export for dashboards
result.trace.to_json("trace.json")
```

### Provider Fallback

Automatic failover with circuit breaker:

```python
from selectools import FallbackProvider, OpenAIProvider, AnthropicProvider

provider = FallbackProvider([
    OpenAIProvider(default_model="gpt-4o-mini"),
    AnthropicProvider(default_model="claude-haiku"),
])
agent = Agent(tools=[...], provider=provider)
# If OpenAI is down → tries Anthropic automatically
```

### Batch Processing

Classify multiple requests concurrently:

```python
results = await agent.abatch(
    ["Cancel my subscription", "How do I upgrade?", "My payment failed"],
    max_concurrency=10,
)
```

### Tool Policy & Human-in-the-Loop

Declarative safety rules with approval callbacks:

```python
from selectools import ToolPolicy

policy = ToolPolicy(
    allow=["search_*", "read_*"],
    review=["send_*", "create_*"],
    deny=["delete_*"],
)

async def confirm(tool_name, tool_args, reason):
    return await get_user_approval(tool_name, tool_args)

config = AgentConfig(tool_policy=policy, confirm_action=confirm)
```

### AgentObserver Protocol

Class-based observability with `run_id` correlation for Langfuse, OpenTelemetry, Datadog, or custom integrations:

```python
from selectools import Agent, AgentConfig, AgentObserver, LoggingObserver

class MyObserver(AgentObserver):
    def on_tool_end(self, run_id, call_id, tool_name, result, duration_ms):
        print(f"[{run_id}] {tool_name} finished in {duration_ms:.1f}ms")

    def on_provider_fallback(self, run_id, failed_provider, next_provider, error):
        print(f"[{run_id}] {failed_provider} failed, falling back to {next_provider}")

agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(observers=[MyObserver(), LoggingObserver()]),
)
```

45 lifecycle events: run, LLM, tool, iteration, batch, policy, structured output, fallback, retry, memory trim, guardrail, coherence, screening, session, entity, KG, budget exceeded, cancelled, prompt compressed, plus 13 graph events (graph start/end, node start/end, routing, interrupt, resume, parallel, stall, loop, supervisor replan). See `observer.py` for full reference.

### E2E Streaming & Parallel Execution

- `agent.astream()` yields `StreamChunk` (text deltas) then `AgentResult` (final)
- Multiple tool calls execute concurrently via `asyncio.gather()` (3 tools @ 0.15s each = ~0.15s total)
- Fallback chain: `astream` -> `acomplete` -> `complete` via executor
- Context propagation with `contextvars` for tracing/auth

See [docs/modules/STREAMING.md](docs/modules/STREAMING.md) for full documentation.

## Providers

| Provider | Streaming | Vision | Native Tools | Cost |
|---|---|---|---|---|
| **OpenAI** | Yes | Yes | Yes | Paid |
| **Anthropic** | Yes | Yes | Yes | Paid |
| **Gemini** | Yes | Yes | Yes | Free tier |
| **Ollama** | Yes | No | No | Free (local) |
| **Fallback** | Yes | Yes | Yes | Varies (wraps others) |
| **Local** | No | No | No | Free (testing) |

```python
from selectools.models import OpenAI, Anthropic, Gemini, Ollama

# IDE autocomplete for all 152 models with pricing metadata
model = OpenAI.GPT_4O_MINI
print(f"Cost: ${model.prompt_cost}/${model.completion_cost} per 1M tokens")
print(f"Context: {model.context_window:,} tokens")
```

## Embedding Providers

```python
from selectools.embeddings import (
    OpenAIEmbeddingProvider,     # text-embedding-3-small/large
    AnthropicEmbeddingProvider,  # Voyage AI (voyage-3, voyage-3-lite)
    GeminiEmbeddingProvider,     # FREE (text-embedding-001/004)
    CohereEmbeddingProvider,     # embed-english-v3.0
)
```

## Vector Stores

```python
from selectools.rag import VectorStore

store = VectorStore.create("memory", embedder=embedder)           # Fast, no persistence
store = VectorStore.create("sqlite", embedder=embedder, db_path="docs.db")  # Persistent
store = VectorStore.create("chroma", embedder=embedder, persist_directory="./chroma")
store = VectorStore.create("pinecone", embedder=embedder, index_name="my-index")
```

## Agent Configuration

```python
config = AgentConfig(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=2000,
    max_iterations=6,
    max_retries=3,
    retry_backoff_seconds=2.0,
    request_timeout=60.0,
    tool_timeout_seconds=30.0,
    cost_warning_threshold=0.50,
    parallel_tool_execution=True,
    routing_only=False,
    stream=False,
    cache=None,                  # InMemoryCache or RedisCache
    tool_policy=None,            # ToolPolicy with allow/review/deny rules
    confirm_action=None,         # Human-in-the-loop approval callback
    approval_timeout=60.0,       # Seconds before auto-deny
    enable_analytics=True,
    verbose=False,
    observers=[LoggingObserver()],  # Lifecycle observer (replaces deprecated hooks)
    system_prompt="You are a helpful assistant...",
)
```

## Tool Definition

### `@tool` Decorator (Recommended)

```python
from selectools import tool

@tool(description="Calculate compound interest")
def calculate_interest(principal: float, rate: float, years: int) -> str:
    amount = principal * (1 + rate / 100) ** years
    return f"After {years} years: ${amount:.2f}"
```

### Tool Registry

```python
from selectools import ToolRegistry

registry = ToolRegistry()

@registry.tool(description="Search the knowledge base")
def search_kb(query: str, max_results: int = 5) -> str:
    return f"Results for: {query}"

agent = Agent(tools=registry.all(), provider=provider)
```

### Injected Parameters

Keep secrets out of the LLM's view:

```python
db_tool = Tool(
    name="query_db",
    description="Execute SQL query",
    parameters=[ToolParameter(name="sql", param_type=str, description="SQL query")],
    function=query_database,
    injected_kwargs={"db_connection": db_conn}  # Hidden from LLM
)
```

### Streaming Tools

```python
from typing import Generator

@tool(description="Process large file", streaming=True)
def process_file(filepath: str) -> Generator[str, None, None]:
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            yield f"[Line {i}] {line.strip()}\n"

config = AgentConfig(observers=[SimpleStepObserver(lambda event, run_id, **kw: print(kw.get("chunk", ""), end=""))])
```

## Conversation Memory

```python
from selectools import Agent, ConversationMemory

memory = ConversationMemory(max_messages=20)
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.ask("My name is Alice")
agent.ask("What's my name?")  # Remembers "Alice"
```

## Cost Tracking

```python
result = agent.ask("Search and summarize")

print(f"Total cost: ${agent.total_cost:.6f}")
print(f"Total tokens: {agent.total_tokens:,}")
print(agent.get_usage_summary())
# Includes LLM + embedding costs, per-tool breakdown
```

## Examples

Examples are numbered by difficulty. Start from 01 and work your way up.

| # | Example | Features | API Key? |
|---|---|---|---|
| 01 | `01_hello_world.py` | First agent, `@tool`, `ask()` | No |
| 02 | `02_search_weather.py` | ToolRegistry, multiple tools | No |
| 03 | `03_toolbox.py` | 24 pre-built tools (file, data, text, datetime, web) | No |
| 04 | `04_conversation_memory.py` | Multi-turn memory | Yes |
| 05 | `05_cost_tracking.py` | Token counting, cost warnings | Yes |
| 06 | `06_async_agent.py` | `arun()`, concurrent agents, FastAPI | Yes |
| 07 | `07_streaming_tools.py` | Generator-based streaming | Yes |
| 08 | `08_streaming_parallel.py` | `astream()`, parallel execution, StreamChunk | Yes |
| 09 | `09_caching.py` | InMemoryCache, RedisCache, cache stats | Yes |
| 10 | `10_routing_mode.py` | Routing mode, intent classification | Yes |
| 11 | `11_tool_analytics.py` | Call counts, success rates, timing | Yes |
| 12 | `12_observability_hooks.py` | Lifecycle hooks, tool validation | Yes |
| 13 | `13_dynamic_tools.py` | ToolLoader, plugins, hot-reload | Yes |
| 14 | `14_rag_basic.py` | RAG pipeline, document loading, vector search | Yes + `[rag]` |
| 15 | `15_semantic_search.py` | Pure semantic search, metadata filtering | Yes + `[rag]` |
| 16 | `16_rag_advanced.py` | PDFs, SQLite persistence, custom chunking | Yes + `[rag]` |
| 17 | `17_rag_multi_provider.py` | Embedding/store/chunk-size comparisons | Yes + `[rag]` |
| 18 | `18_hybrid_search.py` | BM25 + vector fusion, RRF, reranking | Yes + `[rag]` |
| 19 | `19_advanced_chunking.py` | Semantic and contextual chunking | Yes + `[rag]` |
| 20 | `20_customer_support_bot.py` | Multi-tool customer support workflow | Yes |
| 21 | `21_data_analysis_agent.py` | Data exploration and analysis | Yes |
| 22 | `22_ollama_local.py` | Fully local LLM via Ollama | No (Ollama) |
| 23 | `23_structured_output.py` | Pydantic response_format, auto-retry, JSON extraction | No |
| 24 | `24_traces_and_reasoning.py` | AgentTrace timeline, reasoning visibility, JSON export | No |
| 25 | `25_provider_fallback.py` | FallbackProvider, circuit breaker, failover chain | No |
| 26 | `26_batch_processing.py` | batch(), abatch(), structured batch, error isolation | No |
| 27 | `27_tool_policy.py` | ToolPolicy, deny_when, HITL approval, memory trimming | No |
| 28 | `28_agent_observer.py` | AgentObserver, LoggingObserver, multiple observers, OTel export | No |
| 29 | `29_guardrails.py` | Input/output guardrails, PII redaction, topic blocking | No |
| 30 | `30_audit_logging.py` | JSONL audit logging, privacy controls, daily rotation | No |
| 31 | `31_tool_output_screening.py` | Prompt injection detection in tool outputs | No |
| 32 | `32_coherence_checking.py` | LLM-based intent verification for injection defense | Yes |
| 33 | `33_persistent_sessions.py` | JsonFileSessionStore, cross-restart persistence | No |
| 34 | `34_summarize_on_trim.py` | Summarize trimmed messages for context preservation | No |
| 35 | `35_entity_memory.py` | Named entity extraction and tracking | No |
| 36 | `36_knowledge_graph.py` | Triple extraction, in-memory and SQLite storage | No |
| 37 | `37_knowledge_memory.py` | Cross-session facts, daily logs, `remember` tool | No |
| 38 | `38_terminal_tools.py` | `@tool(terminal=True)`, `stop_condition` callback | No |
| 39 | `39_eval_framework.py` | EvalSuite, TestCase, evaluators, HTML reports | No |
| 40 | `40_eval_advanced.py` | Pairwise A/B, regression detection, snapshots | No |
| 41 | `41_mcp_client.py` | MCPClient, mcp_tools(), tool interop | No |
| 42 | `42_mcp_server.py` | MCPServer, expose tools as MCP endpoints | No |
| 43 | `43_token_budget.py` | `max_total_tokens`, `max_cost_usd` budget limits | No |
| 44 | `44_cancellation.py` | CancellationToken, cooperative stopping | No |
| 45 | `45_approval_gate.py` | `@tool(requires_approval=True)`, confirm_action | No |
| 46 | `46_simple_observer.py` | SimpleStepObserver, single-callback integration | No |
| 47 | `47_token_estimation.py` | `estimate_run_tokens()`, pre-flight cost checks | No |
| 48 | `48_model_switching.py` | `model_selector` callback, per-iteration model | No |
| 49 | `49_knowledge_stores.py` | SQLite, Redis, Supabase knowledge stores | No |
| 50 | `50_reasoning_strategies.py` | ReAct, Chain-of-Thought, Plan-then-Act | No |
| 51 | `51_tool_result_caching.py` | `@tool(cacheable=True, cache_ttl=300)` | No |
| 52 | `52_semantic_cache.py` | SemanticCache with embedding similarity | Yes |
| 53 | `53_prompt_compression.py` | Auto-summarize old history on context fill | No |
| 54 | `54_conversation_branching.py` | `memory.branch()`, `store.branch()` | No |
| 55 | `55_agent_graph_linear.py` | Linear AgentGraph pipeline | No |
| 56 | `56_agent_graph_parallel.py` | Parallel fan-out with merge policies | No |
| 57 | `57_agent_graph_conditional.py` | Conditional routing with plain Python | No |
| 58 | `58_agent_graph_hitl.py` | Human-in-the-loop with generator nodes | No |
| 59 | `59_agent_graph_checkpointing.py` | Checkpoint, interrupt, resume | No |
| 60 | `60_supervisor_agent.py` | SupervisorAgent with 4 strategies | No |
| 61 | `61_agent_graph_subgraph.py` | Nested subgraph composition | No |
| 62 | `62_yaml_config.py` | Load AgentConfig from YAML | No |
| 63 | `63_agent_templates.py` | Built-in agent templates | No |
| 64 | `64_selectools_serve.py` | Serve agent over HTTP with `selectools serve` | No |
| 65 | `65_tool_composition.py` | `compose()` tool chaining | No |
| 66 | `66_streaming_pipeline.py` | `pipeline.astream()` streaming composition | No |
| 67 | `67_type_safe_pipeline.py` | Type-safe step contracts | No |
| 68 | `68_postgres_checkpoints.py` | PostgresCheckpointStore for AgentGraph | Yes + `[postgres]` |
| 69 | `69_trace_store.py` | Trace storage and querying | No |
| 70 | `70_plan_and_execute.py` | PlanAndExecuteAgent with typed steps | No |
| 71 | `71_reflective_agent.py` | ReflectiveAgent actor–critic loop | No |
| 72 | `72_debate_agent.py` | DebateAgent with optimist/skeptic/judge | No |
| 73 | `73_team_lead_agent.py` | TeamLeadAgent with all 3 delegation strategies | No |

Run any example:

```bash
python examples/01_hello_world.py   # No API key needed
python examples/14_rag_basic.py     # Needs OPENAI_API_KEY
```

## Documentation

**[Read the full documentation](https://johnnichev.github.io/selectools)** — hosted on GitHub Pages with search, dark mode, and easy navigation.

Also available in [`docs/`](docs/README.md):

| Module | Description |
|---|---|
| [AGENT](docs/modules/AGENT.md) | Agent loop, structured output, traces, reasoning, batch, policy |
| [STREAMING](docs/modules/STREAMING.md) | E2E streaming, parallel execution, routing |
| [TOOLS](docs/modules/TOOLS.md) | Tool definition, validation, registry |
| [DYNAMIC_TOOLS](docs/modules/DYNAMIC_TOOLS.md) | ToolLoader, plugins, hot-reload |
| [HYBRID_SEARCH](docs/modules/HYBRID_SEARCH.md) | BM25, fusion, reranking |
| [ADVANCED_CHUNKING](docs/modules/ADVANCED_CHUNKING.md) | Semantic & contextual chunking |
| [RAG](docs/modules/RAG.md) | Complete RAG pipeline |
| [EMBEDDINGS](docs/modules/EMBEDDINGS.md) | Embedding providers |
| [VECTOR_STORES](docs/modules/VECTOR_STORES.md) | Storage backends |
| [PROVIDERS](docs/modules/PROVIDERS.md) | LLM provider adapters + FallbackProvider |
| [MEMORY](docs/modules/MEMORY.md) | Conversation memory + tool-pair trimming |
| [USAGE](docs/modules/USAGE.md) | Cost tracking & analytics |
| [MODELS](docs/modules/MODELS.md) | Model registry & pricing |
| [SESSIONS](docs/modules/SESSIONS.md) | Persistent session stores (JSON, SQLite, Redis) |
| [ENTITY_MEMORY](docs/modules/ENTITY_MEMORY.md) | Entity extraction and tracking |
| [KNOWLEDGE_GRAPH](docs/modules/KNOWLEDGE_GRAPH.md) | Triple extraction and storage |
| [KNOWLEDGE](docs/modules/KNOWLEDGE.md) | Cross-session knowledge memory |
| [GUARDRAILS](docs/modules/GUARDRAILS.md) | Input/output validation pipeline |
| [AUDIT](docs/modules/AUDIT.md) | JSONL audit logging |
| [SECURITY](docs/modules/SECURITY.md) | Screening & coherence checking |
| [EVALS](docs/modules/EVALS.md) | 50 evaluators, A/B testing, regression |
| [MCP](docs/modules/MCP.md) | MCP client/server integration |
| [BUDGET](docs/modules/BUDGET.md) | Token/cost budget limits |
| [CANCELLATION](docs/modules/CANCELLATION.md) | Cooperative cancellation |
| [ORCHESTRATION](docs/modules/ORCHESTRATION.md) | AgentGraph, routing, parallel, HITL |
| [SUPERVISOR](docs/modules/SUPERVISOR.md) | SupervisorAgent, 4 strategies |
| [PATTERNS](docs/modules/PATTERNS.md) | PlanAndExecute, Reflective, Debate, TeamLead |
| [PARSER](docs/modules/PARSER.md) | Tool call parsing |
| [PROMPT](docs/modules/PROMPT.md) | System prompt generation |

## Tests

```bash
pytest tests/ -x -q          # All tests
pytest tests/ -k "not e2e"   # Skip E2E (no API keys needed)
```

3135 tests covering parsing, agent loop, providers, RAG pipeline, hybrid search, advanced chunking, dynamic tools, caching, streaming, guardrails, sessions, memory, eval framework, budget/cancellation, knowledge stores, orchestration, pipelines, agent patterns, stability markers, trace viewer, and E2E integration with real API calls.

## License

**Apache-2.0** — Use freely in commercial applications. No copyleft restrictions. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome contributions for new tools, providers, vector stores, examples, and documentation.

---

[Roadmap](ROADMAP.md) | [Changelog](CHANGELOG.md) | [Documentation](docs/README.md)
