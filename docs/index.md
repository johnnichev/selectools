---
hide:
  - navigation
---

# Selectools

**Production-ready AI agents with tool calling, RAG, and hybrid search.**

Connect LLMs to your Python functions, embed and search your documents with vector + keyword fusion, stream responses in real time, and dynamically manage tools at runtime.

[:octicons-download-16: Install](#install){ .md-button .md-button--primary }
[:octicons-book-16: Quickstart](QUICKSTART.md){ .md-button }
[:octicons-mark-github-16: GitHub](https://github.com/johnnichev/selectools){ .md-button }

---

## Why Selectools?

<div class="grid cards" markdown>

-   :material-swap-horizontal:{ .lg .middle } **Provider Agnostic**

    ---

    Switch between OpenAI, Anthropic, Gemini, and Ollama with one line. Your tools stay identical.

-   :material-code-json:{ .lg .middle } **Structured Output**

    ---

    Pydantic or JSON Schema `response_format` with auto-retry on validation failure.

-   :material-chart-timeline:{ .lg .middle } **Execution Traces**

    ---

    Every `run()` returns `result.trace` — a structured timeline of LLM calls, tool picks, and executions.

-   :material-shield-check:{ .lg .middle } **Enterprise Security**

    ---

    Guardrails, PII redaction, prompt injection screening, coherence checking, and JSONL audit logging.

-   :material-magnify:{ .lg .middle } **Hybrid RAG Search**

    ---

    BM25 keyword + vector semantic search with RRF/weighted fusion and cross-encoder reranking.

-   :material-lightning-bolt:{ .lg .middle } **E2E Streaming**

    ---

    Token-level `astream()` with native tool call support. Parallel tool execution via `asyncio.gather`.

</div>

---

## Install

=== "Core"

    ```bash
    pip install selectools
    ```

=== "Core + RAG"

    ```bash
    pip install selectools[rag]
    ```

=== "Everything"

    ```bash
    pip install selectools[rag,cache]
    ```

---

## Minimal Agent

```python
from selectools import Agent, AgentConfig, tool
from selectools.providers import OpenAIProvider

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 72°F and sunny in {city}"

agent = Agent(
    config=AgentConfig(model="gpt-4.1-mini"),
    provider=OpenAIProvider(),
    tools=[get_weather],
)

result = agent.run("What's the weather in NYC?")
print(result.content)         # "It's 72°F and sunny in NYC"
print(result.trace.timeline)  # Structured execution trace
print(result.reasoning)       # Why the agent chose get_weather
```

!!! tip "No API key? No problem"
    Use the built-in `LocalProvider` stub to test tool calling and agent logic without any API key.
    See the [Quickstart](QUICKSTART.md) for details.

---

## What's Included

| Category | Highlights |
|---|---|
| **5 LLM Providers** | OpenAI, Anthropic, Gemini, Ollama + FallbackProvider (auto-failover with circuit breaker) |
| **152 Models** | Type-safe constants with March 2026 pricing for all providers |
| **Structured Output** | Pydantic / JSON Schema `response_format` with auto-retry on validation failure |
| **Execution Traces** | `result.trace` — typed timeline of every LLM call, tool pick, and execution |
| **Reasoning Visibility** | `result.reasoning` explains *why* the agent chose each tool |
| **Batch Processing** | `agent.batch()` / `agent.abatch()` for concurrent multi-prompt classification |
| **Tool Policy Engine** | Declarative allow/review/deny rules with human-in-the-loop callbacks |
| **E2E Streaming** | Token-level `astream()` with native tool call support and parallel execution |
| **Dynamic Tool Loading** | Plugin system with hot-reload, `add_tools()`, `remove_tool()`, `replace_tool()` |
| **Response Caching** | InMemoryCache and RedisCache with LRU + TTL and stats tracking |
| **24 Pre-built Tools** | Files, data, text, datetime, web — ready to use |
| **Hybrid Search** | BM25 + vector fusion with Cohere/Jina reranking |
| **Advanced Chunking** | Semantic + contextual chunking for better RAG retrieval |
| **Guardrails Engine** | Input/output validation with PII redaction, topic blocking, toxicity detection |
| **Audit Logging** | JSONL audit trail with privacy controls and daily rotation |
| **Tool Output Screening** | Prompt injection detection with 15 built-in patterns |
| **Coherence Checking** | LLM-based verification that tool calls match user intent |
| **Persistent Sessions** | SessionStore protocol with JSON file, SQLite, and Redis backends with TTL |
| **Summarize-on-Trim** | LLM-generated summaries of trimmed messages for context preservation |
| **Entity Memory** | Auto-extract named entities with LRU-pruned registry and context injection |
| **Knowledge Graph** | Relationship triple extraction with in-memory and SQLite storage |
| **Cross-Session Knowledge** | Daily logs + persistent facts with auto-registered `remember` tool |
| **AgentObserver Protocol** | 45-event lifecycle observer with run/call ID correlation, `SimpleStepObserver`, and OTel export |
| **Runtime Controls** | Token/cost budget limits, cooperative cancellation, per-tool approval gates, model switching per iteration |
| **Reasoning Strategies** | Built-in ReAct, Chain-of-Thought, and Plan-Then-Act via `reasoning_strategy` config |
| **Tool Result Caching** | `@tool(cacheable=True, cache_ttl=60)` — skip re-execution for identical tool calls |
| **Semantic Cache** | `SemanticCache` — embedding-based cache hits for paraphrased queries via cosine similarity |
| **Prompt Compression** | Proactive context-window management — summarises old messages when fill-rate exceeds threshold |
| **Conversation Branching** | `memory.branch()` and `store.branch()` — fork history for A/B exploration and checkpointing |
| **Multi-Agent Orchestration** | `AgentGraph` for directed graphs, `SupervisorAgent` with 4 strategies, HITL via generator nodes, parallel execution, 3 checkpoint backends |
| **Advanced Agent Patterns** | `PlanAndExecuteAgent`, `ReflectiveAgent`, `DebateAgent`, `TeamLeadAgent` — high-level coordination patterns built on AgentGraph |
| **Composable Pipelines** | `Pipeline` + `@step` + `|` operator + `parallel()` + `branch()` — chain agents, tools, and transforms with plain Python |
| **Eval Framework** | 50 built-in evaluators, A/B testing, regression detection, HTML reports, JUnit XML |
| **2918 Tests** | Unit, integration, regression, and E2E |

---

## Learning Path

!!! abstract "Beginner"
    1. **[Quickstart](QUICKSTART.md)** — Build your first agent in 5 minutes
    2. **[Architecture](ARCHITECTURE.md)** — Big-picture overview
    3. **[Agent](modules/AGENT.md)** — Core agent loop
    4. **[Tools](modules/TOOLS.md)** — Creating custom tools

!!! abstract "Intermediate"
    5. **[Providers](modules/PROVIDERS.md)** — Switch between OpenAI, Anthropic, Gemini, Ollama
    6. **[Memory](modules/MEMORY.md)** — Conversation persistence and sliding windows
    7. **[Streaming](modules/STREAMING.md)** — Real-time token streaming
    8. **[Models & Pricing](modules/MODELS.md)** — 152 models with cost data

!!! abstract "Advanced"
    9. **[RAG Pipeline](modules/RAG.md)** — Document search and retrieval
    10. **[Hybrid Search](modules/HYBRID_SEARCH.md)** — BM25 + vector fusion
    11. **[Dynamic Tools](modules/DYNAMIC_TOOLS.md)** — Plugin systems and hot-reload

!!! abstract "Production / Enterprise"
    12. **[Guardrails](modules/GUARDRAILS.md)** — Input/output validation pipeline
    13. **[Audit Logging](modules/AUDIT.md)** — Compliance and privacy-aware logging
    14. **[Security](modules/SECURITY.md)** — Tool output screening and coherence checking
    15. **[Error Handling](modules/EXCEPTIONS.md)** — Custom exception hierarchy

!!! abstract "Evaluation"
    16. **[Eval Framework](modules/EVALS.md)** — 50 built-in evaluators, A/B testing, regression detection, snapshot testing

!!! abstract "Memory & Persistence"
    17. **[Sessions](modules/SESSIONS.md)** — Persistent session storage with 3 backends
    18. **[Entity Memory](modules/ENTITY_MEMORY.md)** — Named entity extraction and tracking
    19. **[Knowledge Graph](modules/KNOWLEDGE_GRAPH.md)** — Relationship triple extraction
    20. **[Knowledge Memory](modules/KNOWLEDGE.md)** — Cross-session durable memory

!!! abstract "Multi-Agent & Composition"
    21. **[Orchestration](modules/ORCHESTRATION.md)** — Agent graphs, routing, parallel execution, HITL
    22. **[Supervisor](modules/SUPERVISOR.md)** — 4 coordination strategies for multi-agent teams
    23. **[Patterns](modules/PATTERNS.md)** — PlanAndExecute, Reflective, Debate, TeamLead
    24. **Pipeline** — Composable pipelines with `@step`, `|` operator, `parallel()`, `branch()`

---

## Architecture at a Glance

```
User Query
  ↓
INPUT GUARDRAILS → validate / redact (PII, topic, toxicity)
  ↓
AGENT → loads history (MEMORY) → calls PROVIDER (or FALLBACK chain)
  ↓
CACHE → checked → hit? Return cached response
  ↓
PROVIDER → formats prompt → calls LLM → CACHE stores result
  ↓
OUTPUT GUARDRAILS → validate response (format, length, toxicity)
  ↓
PARSER → extracts TOOL_CALL → REASONING extracted
  ↓
POLICY ENGINE → allow / review / deny → HITL callback if review
  ↓
COHERENCE CHECK → verifies tool call matches user intent
  ↓
TOOL EXECUTION → parallel if multiple → OUTPUT SCREENING
  ↓
TRACE records step → AUDIT LOGGER writes JSONL → USAGE tracks costs
  ↓
Loop continues or returns AgentResult (.parsed, .trace, .reasoning)

Multi-Agent layer (optional):
  AGENT_GRAPH → directed graph of agents with routing, parallel fan-out,
                and checkpoint-backed state
  SUPERVISOR  → coordinates agent teams via round-robin, priority,
                adaptive, or LLM-based strategy selection
```

---

## Links

[:fontawesome-brands-python: PyPI Package](https://pypi.org/project/selectools/){ .md-button }
[:fontawesome-brands-github: GitHub Repository](https://github.com/johnnichev/selectools){ .md-button }
[:material-notebook: Getting Started Notebook](https://github.com/johnnichev/selectools/blob/main/notebooks/getting_started.ipynb){ .md-button }
[:material-code-tags: 73 Example Scripts](https://github.com/johnnichev/selectools/tree/main/examples){ .md-button }

---

*An open-source project from [NichevLabs](https://nichevlabs.com).*
