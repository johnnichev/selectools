# Selectools Development Roadmap

An open-source project from [NichevLabs](https://nichevlabs.com).

> **Status Legend**
>
> - ✅ **Implemented** - Merged and available in latest release
> - 🔵 **In Progress** - Actively being worked on
> - 🟡 **Planned** - Scheduled for implementation
> - ⏸️ **Deferred** - Postponed to later release
> - ❌ **Cancelled** - No longer planned

---

v0.17.0 ✅ Eval Framework
39 evaluators → A/B testing → snapshots → regression → HTML/JUnit → CI → templates

v0.17.1 ✅ MCP Client/Server
MCPClient → mcp_tools() → MCPServer → MultiMCPClient → tool interop

v0.17.3 ✅ Agent Runtime Controls
Token budget → Cancellation → Cost attribution → Structured results → Approval gate → SimpleStepObserver

v0.17.4 ✅ Agent Intelligence
Token estimation → Model switching → Knowledge memory enhancement (4 store backends)

v0.17.5 ✅ Bug Hunt & Async Guardrails
91 validated fixes (13 critical, 26 high, 52 medium+low) → Async guardrails
→ 40 regression tests → 5 new Common Pitfalls

v0.17.6 ✅ Quick Wins
ReAct/CoT reasoning strategies → Tool result caching → Python 3.9–3.13 CI matrix

v0.17.7 ✅ Caching & Context
Semantic caching → Prompt compression → Conversation branching
(55 tests, 3 examples)

v0.18.0 ✅ Multi-Agent Orchestration
AgentGraph → GraphState → Typed reducers → Resume-from-yield interrupts
→ Scatter fan-out → Checkpointing → SupervisorAgent → Graph visualization

v0.18.x 🟡 Composability Layer
Pipeline → @step decorator → | operator → parallel/branch/retry
→ Tool composition (@compose) → Type-safe composition (LCEL answer)

v0.19.0 🟡 Serve & Deploy
Structured AgentConfig refactor (41 fields → nested dataclasses)
→ selectools serve CLI → FastAPI AgentRouter → Flask AgentBlueprint
→ Playground UI → YAML agent config → Agent templates (5 built-in)

v0.19.x 🟡 Enterprise Hardening
Security audit → Stability markers (Production/Stable)
→ Deprecation policy → Compatibility matrix → Postgres checkpoint backend

v0.20.0 🟡 Advanced Agent Patterns
PlanAndExecute → ReflectiveAgent → Debate → TeamLead
→ 50+ evaluators (semantic similarity, rubric, multi-turn coherence)

v0.20.x 🟡 Connector Expansion + Performance
AWS Bedrock provider → Azure OpenAI → FAISS → Qdrant → pgvector
→ CSV/JSON/HTML/URL loaders → Published benchmarks

v0.21.0 🟡 Polish & Community
Tool marketplace → Visual agent builder → Enhanced trace viewer
→ LangChain migration guide → Cookbook → Documentation generation

---

## v0.17.0: Eval Framework ✅

Focus: Built-in evaluation and testing for AI agents — a capability no other framework ships as a library feature.

### EvalSuite

```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(agent=agent, cases=[
    TestCase(input="Cancel my account", expect_tool="cancel_subscription"),
    TestCase(input="Check my balance", expect_contains="balance"),
    TestCase(input="What's 2+2?", expect_output="4"),
])
report = suite.run()
print(report.accuracy)       # 0.95
print(report.latency_p50)    # 1.2s
print(report.total_cost)     # $0.003
```

### Built-in Evaluators

- `ToolUseEvaluator` — did the agent pick the right tool?
- `CorrectnessEvaluator` — LLM-as-judge: is the output correct?
- `RelevanceEvaluator` — LLM-as-judge: is the output relevant?
- `FaithfulnessEvaluator` — RAG-specific: is the output grounded in retrieved documents?
- `LatencyEvaluator` — did the agent respond within the time budget?
- `CostEvaluator` — did the agent stay within the cost budget?

### Module Structure

```
src/selectools/evals/
    __init__.py          # Public exports
    types.py             # TestCase, CaseResult, CaseVerdict, EvalFailure
    suite.py             # EvalSuite orchestration (run/arun, batch dispatch)
    evaluators.py        # Evaluator protocol + built-in evaluators
    report.py            # EvalReport with accuracy, latency, cost, regressions
    dataset.py           # DatasetLoader (JSON/YAML/dict -> List[TestCase])
    regression.py        # BaselineStore, RegressionDetector
    html.py              # Self-contained HTML report renderer
    junit.py             # JUnit XML output for CI
```

### Key Differentiator

Every team building agents needs evaluation. Today they either build it from scratch or pay for a SaaS product. Selectools ships it as a library — zero dependencies, runs in your test suite, produces structured reports.

| Feature                        | Status  | Impact | Effort |
| ------------------------------ | ------- | ------ | ------ |
| **EvalSuite + TestCase**       | ✅ Done | High   | Medium |
| **Built-in Evaluators (39)**   | ✅ Done | High   | Medium |
| **EvalReport**                 | ✅ Done | Medium | Small  |
| **Dataset Loader (JSON/YAML)** | ✅ Done | Medium | Small  |
| **Regression Detection**       | ✅ Done | High   | Medium |
| **HTML Report Export**         | ✅ Done | Medium | Small  |
| **JUnit XML for CI**           | ✅ Done | Medium | Small  |
| **Pairwise A/B eval**          | ✅ Done | Medium | Small  |
| **Snapshot testing**           | ✅ Done | Medium | Small  |
| **Live eval dashboard**        | ✅ Done | Medium | Medium |
| **SVG badge generator**        | ✅ Done | Low    | Small  |

---

## v0.17.1: MCP Client/Server ✅

Focus: Model Context Protocol integration for tool interoperability.

### MCP Client

Discover and call tools from any MCP-compliant server:

```python
from selectools.mcp import MCPClient, mcp_tools

client = MCPClient(server_url="http://localhost:8080")
tools = mcp_tools(client)   # Returns List[Tool] that proxy to MCP server

agent = Agent(tools=tools + local_tools, provider=provider)
```

### MCP Server

Expose `@tool` functions as MCP-compliant endpoints:

```python
from selectools.mcp import MCPServer

server = MCPServer(tools=[search_tool, calculator_tool])
server.serve(host="0.0.0.0", port=8080)
```

### Key Differentiator

First-class MCP support lets Selectools agents use any MCP-compatible tool server — massive integration surface area without building individual connectors.

| Feature                 | Status  | Impact | Effort |
| ----------------------- | ------- | ------ | ------ |
| **MCPClient**           | ✅ Done | High   | Medium |
| **mcp_tools() adapter** | ✅ Done | High   | Small  |
| **MCPServer**           | ✅ Done | Medium | Medium |
| **MultiMCPClient**      | ✅ Done | Medium | Small  |

---

## v0.17.7: Caching & Context ✅

Focus: Smarter token management and memory exploration.

### Semantic Caching

Drop-in replacement for `InMemoryCache` that embeds cache keys and uses cosine similarity to serve semantically equivalent queries from cache — even if the exact prompt wording differs.

```python
from selectools.cache_semantic import SemanticCache
from selectools.providers.openai_provider import OpenAIProvider

cache = SemanticCache(
    embedding_provider=OpenAIProvider(),
    similarity_threshold=0.92,
    max_size=1000,
    ttl=3600,
)
agent = Agent(tools=[...], provider=provider, config=AgentConfig(cache=cache))
# "What's the weather in NYC?" hits cache for "Weather in New York City?"
```

Same `Cache` protocol as `InMemoryCache` — zero migration cost.

### Prompt Compression

Before each LLM call, if the estimated token count exceeds a configurable threshold of the context window, proactively summarize old messages rather than letting the window overflow. Connects `estimate_run_tokens()` + `summarize_on_trim` into a proactive system.

```python
config = AgentConfig(
    compress_context=True,
    compress_threshold=0.75,   # start compressing at 75% of context window
    compress_keep_recent=4,    # always keep the last N turns verbatim
)
```

### Conversation Branching

Deep-copy conversation state for exploration, what-if analysis, or agent evaluation.

```python
branch = agent.memory.branch()             # snapshot current state
branch_result = branch_agent.run("...")    # explore without affecting main

# Also works with session stores:
session_store.branch(source_id="conv-123", new_id="conv-123-alt")
```

| Feature                    | Status    | Impact | Effort |
| -------------------------- | --------- | ------ | ------ |
| **SemanticCache**          | 🟡 High   | High   | Medium |
| **Prompt compression**     | 🟡 High   | Medium | Medium |
| **Conversation branching** | 🟡 Medium | Medium | Small  |

---

## v0.18.0: Multi-Agent Orchestration ✅

**Status: ✅ Implemented in v0.18.0**

Focus: DAG-based multi-agent workflows that are simpler and more Pythonic than LangGraph.

### Design Philosophy

LangGraph requires learning StateGraph, MessageAnnotation, Pregel channels, and a custom checkpointing API before building anything. Selectools takes the opposite approach: **agents are the primitive, composition is plain Python**. An `AgentGraph` should feel like writing normal Python with `async/await`, not configuring a data pipeline.

**Core principles**:

1. **Agents are nodes, not functions** — each node is a full `Agent` instance with its own tools, provider, and config, reusing all existing infrastructure (traces, observers, guardrails, policies)
2. **Edges are just Python functions** — no special `ConditionalEdge` class; a routing function takes the result and returns the next node name via plain `if/elif/else`
3. **State is a typed dataclass** — no Pydantic models for state; just a `@dataclass` that gets passed between nodes
4. **Checkpointing is serialization** — the state is JSON-serializable; checkpoint stores implement a 3-method protocol
5. **HITL reuses existing patterns** — the existing `ToolPolicy` + `confirm_action` pattern already handles human-in-the-loop

### Module Structure

```
src/selectools/orchestration/
    __init__.py           # Public exports: AgentGraph, GraphNode, GraphState, GraphResult
    graph.py              # AgentGraph: the DAG-based orchestration engine
    node.py               # GraphNode: wraps Agent with input/output transforms
    state.py              # GraphState: typed state container with merge semantics
    checkpoint.py         # CheckpointStore protocol + InMemory, File, SQLite backends
    supervisor.py         # SupervisorAgent: meta-agent for task decomposition
```

### Core Abstractions

#### GraphState

```python
@dataclass
class GraphState:
    messages: List[Message]                    # Accumulated messages across nodes
    data: Dict[str, Any]                       # Arbitrary key-value store for inter-node communication
    current_node: str                          # Name of the currently executing node
    history: List[Tuple[str, AgentResult]]     # Ordered list of (node_name, result) pairs
    metadata: Dict[str, Any]                   # User-attached metadata (carried through checkpoints)
```

Intentionally flat and JSON-serializable. No Pydantic, no custom descriptors, no annotation magic.

#### AgentGraph

```python
from selectools.orchestration import AgentGraph

graph = AgentGraph()
graph.add_node("planner", planner_agent)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)

graph.add_edge("planner", "researcher")
graph.add_conditional_edge("researcher", lambda state: "writer" if state.data.get("ready") else "researcher")
graph.add_edge("writer", AgentGraph.END)

graph.set_entry("planner")
result = graph.run("Write a blog post about AI agents")
```

### How It Beats LangGraph

| LangGraph                                                | Selectools AgentGraph                            | Why better                                                       |
| -------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| Custom `StateGraph` with `Annotated[list, add_messages]` | Plain `GraphState` dataclass                     | No custom type system to learn                                   |
| `conditionalEdges` with special return constants         | Plain Python function returning a string         | Debuggable, testable, IDE-friendly                               |
| Pregel channels for state management                     | `Dict[str, Any]` with merge functions            | Standard Python data structures                                  |
| Separate `compile()` step before execution               | Validate + run in one step                       | No compilation phase, faster iteration                           |
| Node functions receive raw state                         | Nodes are full `Agent` instances                 | Inherit all Agent features: tools, traces, observers, guardrails |
| Complex interrupt/resume for human-in-the-loop           | Reuse existing `confirm_action` on `AgentConfig` | Zero new concepts for HITL                                       |

| Feature                     | Status    | Impact | Effort |
| --------------------------- | --------- | ------ | ------ |
| **AgentGraph + GraphState** | ✅ High   | High   | Large  |
| **Checkpointing**           | ✅ High   | High   | Medium |
| **Parallel Nodes**          | ✅ Medium | High   | Medium |
| **SupervisorAgent**         | ✅ Medium | High   | Medium |

---

## v0.18.x: Composability Layer 🟡

Focus: Give selectools a composable pipeline abstraction — the answer to LangChain's LCEL. Lets users chain agents, tools, and transforms with the `|` operator.

### Pipeline + @step Decorator

```python
from selectools import Pipeline, step

@step
def summarize(text: str) -> str:
    """Summarize input text."""
    return agent.run(f"Summarize: {text}").content

@step
def translate(text: str, lang: str = "es") -> str:
    """Translate text to target language."""
    return agent.run(f"Translate to {lang}: {text}").content

# Compose with | operator
pipeline = summarize | translate
result = pipeline.run("Long article text here...")

# Or build programmatically
pipeline = Pipeline(steps=[summarize, translate])
result = pipeline.run(input_data)
```

### Tool Composition

```python
from selectools import tool, compose

@tool()
def fetch_data(url: str) -> str: ...

@tool()
def parse_json(text: str) -> dict: ...

@tool()
def extract_field(data: dict, field: str) -> str: ...

# Chain tools into a single composite tool
fetch_and_extract = compose(fetch_data, parse_json, extract_field)
agent = Agent(tools=[fetch_and_extract], ...)
```

### Parallel & Branch Primitives

```python
from selectools import Pipeline, parallel, branch

# Fan-out to multiple steps, merge results
research = parallel(search_web, search_docs, search_db)

# Conditional branching
route = branch(
    lambda x: "technical" if "code" in x else "general",
    technical=code_review_pipeline,
    general=summarize_pipeline,
)

full_pipeline = research | route | final_review
```

### Key Differentiator

LangChain's LCEL is powerful but opaque — debugging `chain.invoke()` requires understanding Runnable internals. Selectools pipelines are plain Python: each `@step` is a function, `|` is sugar for sequential composition, and every step produces a trace entry.

| Feature | Status | Impact | Effort |
| --- | --- | --- | --- |
| **Pipeline + @step** | 🟡 High | High | Medium |
| **\| operator** | 🟡 High | High | Small |
| **Tool composition (@compose)** | 🟡 High | Medium | Small |
| **parallel() / branch()** | 🟡 Medium | Medium | Medium |
| **Type-safe step chaining** | 🟡 Medium | Medium | Medium |

---

## v0.19.0: Serve & Deploy 🟡

Focus: REST API deployment, agent templates, and developer experience — going from library to platform.

### Selectools Serve (REST API Deployment)

**New `src/selectools/serve/` package**:

```
src/selectools/serve/
    __init__.py          # Public exports: AgentRouter, AgentBlueprint, playground
    fastapi.py           # FastAPI router
    flask.py             # Flask blueprint
    playground.py        # Self-contained chat UI + server
    models.py            # Pydantic request/response models
```

#### FastAPI Router

```python
from selectools.serve import AgentRouter

router = AgentRouter(agent=my_agent, prefix="/agent")
# Creates:
#   POST /agent/invoke     — single prompt → AgentResult as JSON
#   POST /agent/batch      — multiple prompts → List[AgentResult]
#   POST /agent/stream     — single prompt → SSE stream
#   GET  /agent/schema     — OpenAPI schema for tools
#   GET  /agent/health     — health check

app = FastAPI()
app.include_router(router)
```

#### Flask Blueprint

```python
from selectools.serve import AgentBlueprint

blueprint = AgentBlueprint(agent=my_agent, prefix="/agent")
app = Flask(__name__)
app.register_blueprint(blueprint)
```

#### Playground

```python
from selectools.serve import playground

playground(agent=my_agent, port=8000)   # Chat UI at http://localhost:8000
```

Self-contained HTML page served by minimal HTTP server. Zero-dependency chat interface.

**Key differentiator vs LangServe**: Works with FastAPI AND Flask. Built-in playground with zero config.

### Templates & Configuration

**New `src/selectools/templates/` package**:

| Template                | Pre-configured with                                         |
| ----------------------- | ----------------------------------------------------------- |
| `customer_support.py`   | Support tools, system prompt, guardrails, topic restriction |
| `data_analyst.py`       | Code execution, data tools, CSV/JSON structured output      |
| `research_assistant.py` | Search tools, web tools, RAG pipeline                       |
| `code_reviewer.py`      | File tools, GitHub tools, structured output                 |
| `rag_chatbot.py`        | RAG pipeline, memory, knowledge base config                 |

**YAML Agent Configuration**:

```yaml
# agent.yaml
provider: openai
model: gpt-4o
tools:
  - selectools.toolbox.file_tools.read_file
  - selectools.toolbox.web_tools.http_get
  - ./my_custom_tool.py
system_prompt: "You are a helpful assistant..."
guardrails:
  input:
    - type: topic
      deny: [politics, religion]
```

```python
from selectools.templates import from_yaml
agent = from_yaml("agent.yaml")
```

### Observability (Trace Store + Export)

**New `src/selectools/observe/` package**:

```
src/selectools/observe/
    __init__.py          # Public exports
    trace_store.py       # TraceStore protocol + InMemory, SQLite, JSONL backends
    export.py            # Export formatters (HTML, CSV, Datadog, Langfuse, OTel)
```

```python
class TraceStore(Protocol):
    def save(self, trace: AgentTrace) -> str: ...
    def load(self, run_id: str) -> AgentTrace: ...
    def query(self, filters: TraceFilter) -> List[AgentTrace]: ...
    def list(self, limit: int, offset: int) -> List[AgentTraceSummary]: ...
```

Built-in: `InMemoryTraceStore`, `SQLiteTraceStore`, `JSONLTraceStore`.

Export formats: HTML (self-contained report), CSV, Datadog APM, Langfuse, OTel.

| Feature                  | Status    | Impact | Effort |
| ------------------------ | --------- | ------ | ------ |
| **FastAPI AgentRouter**  | 🟡 High   | High   | Medium |
| **Flask AgentBlueprint** | 🟡 Medium | Medium | Small  |
| **Playground**           | 🟡 Medium | High   | Medium |
| **Trace Store**          | 🟡 High   | High   | Medium |
| **HTML Trace Export**    | 🟡 Medium | High   | Medium |
| **Agent Templates**      | 🟡 Medium | Medium | Small  |
| **YAML Config**          | 🟡 Low    | Medium | Small  |

---

## v0.19.x: Enterprise Hardening 🟡

Focus: Production readiness signals that enterprise teams require before adopting a framework.

### Security Audit

- Dependency vulnerability scan (Snyk/Safety)
- Code audit for OWASP Top 10 in all provider/tool paths
- Bandit scan clean (currently uses `# nosec` annotations — verify each one)
- Document security model: what's trusted, what's validated, what's user-controlled

### Stability Markers

```python
# Module-level stability markers
__stability__ = "stable"      # agent, types, trace, observer
__stability__ = "beta"        # orchestration, evals, mcp
__stability__ = "alpha"       # serve (v0.19.0), pipeline (v0.18.x)
```

Documented in each module's docstring. Public API surface frozen for `stable` modules — breaking changes require a major version bump.

### Postgres Checkpoint Backend

```python
from selectools.orchestration import PostgresCheckpointStore

store = PostgresCheckpointStore(dsn="postgresql://user:pass@host/db")
result = graph.run("...", checkpoint_store=store)
```

Closes the gap vs LangGraph's `PostgresSaver`. Uses `asyncpg` (optional dependency).

### Additional Items

- Deprecation policy: 2-version warning before removal
- Compatibility matrix: Python 3.9-3.13, provider SDK versions
- SBOM generation for compliance teams

| Feature | Status | Impact | Effort |
| --- | --- | --- | --- |
| **Security audit** | 🟡 High | High | Medium |
| **Stability markers** | 🟡 High | Medium | Small |
| **Postgres checkpoint** | 🟡 High | High | Medium |
| **Deprecation policy** | 🟡 Medium | Medium | Small |
| **Compatibility matrix** | 🟡 Medium | Medium | Small |

---

## v0.20.0: Advanced Agent Patterns 🟡

Focus: Higher-level agent architectures built on the v0.18.0 orchestration primitives. Each pattern is a pre-built `AgentGraph` topology.

### PlanAndExecute Agent

```python
from selectools.patterns import PlanAndExecuteAgent

agent = PlanAndExecuteAgent(
    planner=planner_agent,
    executors={"research": researcher, "write": writer, "review": reviewer},
    provider=provider,
)
result = agent.run("Write a technical blog post about vector databases")
# Planner creates structured plan → executors handle each step → result aggregated
```

### ReflectiveAgent

```python
from selectools.patterns import ReflectiveAgent

agent = ReflectiveAgent(
    actor=writer_agent,
    critic=reviewer_agent,
    max_reflections=3,
)
result = agent.run("Draft a press release")
# Actor produces draft → Critic evaluates → Actor revises → repeat until satisfied
```

### Debate Pattern

```python
from selectools.patterns import DebateAgent

agent = DebateAgent(
    agents={"optimist": optimist_agent, "skeptic": skeptic_agent},
    judge=judge_agent,
    max_rounds=3,
)
result = agent.run("Should we adopt microservices?")
# Agents argue positions → Judge synthesizes final answer
```

### TeamLead Pattern

```python
from selectools.patterns import TeamLeadAgent

agent = TeamLeadAgent(
    lead=lead_agent,
    team={"analyst": analyst, "engineer": engineer, "writer": writer},
    delegation_strategy="dynamic",
)
result = agent.run("Investigate and fix the billing discrepancy")
# Lead delegates tasks, reviews work, coordinates handoffs
```

### Expanded Eval Suite (50+ evaluators)

- Semantic similarity (embedding-based)
- Multi-turn coherence (conversation-level)
- Custom rubric scoring
- Agent trajectory evaluation (did the agent follow the right path?)
- Tool efficiency (did the agent use the minimum tools needed?)

| Feature | Status | Impact | Effort |
| --- | --- | --- | --- |
| **PlanAndExecute** | 🟡 High | High | Medium |
| **ReflectiveAgent** | 🟡 High | High | Medium |
| **Debate** | 🟡 Medium | Medium | Medium |
| **TeamLead** | 🟡 Medium | Medium | Medium |
| **50+ evaluators** | 🟡 Medium | High | Large |

---

## v0.20.x: Connector Expansion + Performance 🟡

Focus: Close the integration gap with LangChain by adding high-demand document loaders, vector stores, and toolbox modules.

### Current Inventory

| Category            | Count    | Items                                    |
| ------------------- | -------- | ---------------------------------------- |
| Document Loaders    | 4        | text, file, directory, PDF               |
| Vector Stores       | 4        | Memory, SQLite, Chroma, Pinecone         |
| Embedding Providers | 4        | OpenAI, Anthropic/Voyage, Gemini, Cohere |
| Toolbox             | 24 tools | file, web, data, datetime, text          |
| Rerankers           | 2        | Cohere, Jina                             |

### New Document Loaders

Add to `src/selectools/rag/loaders.py` as new static methods on `DocumentLoader`. Refactor to `loaders/` subpackage with `__init__.py` re-exporting everything to support SaaS loaders as separate files.

| Loader                      | Method                                              | Dependencies                  | Complexity | Why it matters                                        |
| --------------------------- | --------------------------------------------------- | ----------------------------- | ---------- | ----------------------------------------------------- |
| **CSV**                     | `from_csv(path, content_columns, metadata_columns)` | stdlib `csv`                  | Small      | Most common structured data format                    |
| **JSON/JSONL**              | `from_json(path, text_field)` / `from_jsonl(...)`   | stdlib `json`                 | Small      | Standard for API responses, logs, datasets            |
| **HTML**                    | `from_html(path_or_content, extract_text=True)`     | `beautifulsoup4` (optional)   | Small      | Web scraping output, saved pages                      |
| **URL**                     | `from_url(url, timeout=30)`                         | `requests` + `beautifulsoup4` | Small      | Direct URL-to-document (2nd most requested after PDF) |
| **Markdown w/ Frontmatter** | `from_markdown(path)`                               | `pyyaml` (optional)           | Small      | Static sites, docs, wikis                             |
| **Google Drive**            | `from_google_drive(file_id, credentials)`           | `google-api-python-client`    | Medium     | Most-used enterprise doc platform                     |
| **Notion**                  | `from_notion(page_id, api_key)`                     | `requests` (existing)         | Medium     | 2nd most-requested SaaS loader                        |
| **GitHub**                  | `from_github(repo, path, branch, token)`            | `requests` (existing)         | Small      | Developer docs and code                               |
| **SQL Database**            | `from_sql(connection_string, query)`                | `sqlalchemy` (optional)       | Medium     | Enterprise data in databases                          |

### New Vector Stores

New files in `src/selectools/rag/stores/`. Each follows the same pattern as `chroma.py`: inherit `VectorStore`, implement `add_documents`, `search`, `delete`, `clear`, lazy-import the dependency. Register in `VectorStore.create()` factory.

| Store            | File          | Dependencies       | Complexity | Why it matters                                                            |
| ---------------- | ------------- | ------------------ | ---------- | ------------------------------------------------------------------------- |
| **FAISS**        | `faiss.py`    | `faiss-cpu`        | Medium     | De facto standard for local high-perf vector search (millions of vectors) |
| **Qdrant**       | `qdrant.py`   | `qdrant-client`    | Medium     | Fastest-growing vector DB, excellent filtering, cloud + self-hosted       |
| **pgvector**     | `pgvector.py` | `psycopg2-binary`  | Medium     | Use existing PostgreSQL — no new database needed                          |
| **Weaviate**     | `weaviate.py` | `weaviate-client`  | Medium     | Popular cloud vector DB with GraphQL API                                  |
| **Redis Vector** | `redis.py`    | `redis` (existing) | Medium     | Leverages existing Redis connection from `cache_redis.py`                 |

### New Toolbox Modules

New files in `src/selectools/toolbox/`. Follow `@tool` decorator pattern, register in `get_all_tools()` and `get_tools_by_category()`.

| Module                | Tools                                                           | Dependencies                   | Complexity   | Why it matters                        |
| --------------------- | --------------------------------------------------------------- | ------------------------------ | ------------ | ------------------------------------- |
| **`code_tools.py`**   | `execute_python`, `execute_shell`                               | stdlib `subprocess`            | Medium       | #1 most-used tool in agent frameworks |
| **`search_tools.py`** | `google_search`, `duckduckgo_search`                            | `duckduckgo_search` (optional) | Small-Medium | #2 most-used tool category            |
| **`github_tools.py`** | `create_issue`, `list_issues`, `create_pr`, `get_file_contents` | `requests` (existing)          | Medium       | Developer workflow automation         |
| **`db_tools.py`**     | `query_database`, `list_tables`, `describe_table`               | `sqlalchemy` (optional)        | Medium       | Enterprise data access                |

### Dependency Management

All new dependencies are optional and lazy-imported. Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
rag = [
    # existing deps ...
    "beautifulsoup4>=4.12.0",
    "faiss-cpu>=1.7.0",
    "qdrant-client>=1.7.0",
    "psycopg2-binary>=2.9.0",
    "weaviate-client>=4.0.0",
]
```

Individual stores/loaders remain installable a la carte: `pip install selectools faiss-cpu` works without the full `[rag]` group.

| Feature                    | Status    | Impact | Effort |
| -------------------------- | --------- | ------ | ------ |
| **CSV/JSON/JSONL Loaders** | 🟡 High   | High   | Small  |
| **HTML/URL Loaders**       | 🟡 High   | High   | Small  |
| **FAISS Vector Store**     | 🟡 High   | High   | Medium |
| **Qdrant Vector Store**    | 🟡 Medium | Medium | Medium |
| **pgvector Store**         | 🟡 Medium | High   | Medium |
| **Code Execution Tools**   | 🟡 High   | High   | Medium |
| **Search Tools**           | 🟡 High   | High   | Small  |
| **SaaS Loaders**           | 🟡 Medium | Medium | Medium |
| **GitHub/DB Toolbox**      | 🟡 Medium | Medium | Medium |

---

## v0.21.0: Polish & Community 🟡

Focus: Niche integrations, community sharing, and developer experience polish.

### Niche Document Loaders

| Loader                                      | Dependencies       |
| ------------------------------------------- | ------------------ |
| `from_slack(channel_id, token)`             | `requests`         |
| `from_confluence(page_id, base_url, token)` | `requests`         |
| `from_jira(project_key, token)`             | `requests`         |
| `from_discord(channel_id, token)`           | `requests`         |
| `from_email(imap_server, credentials)`      | stdlib `imaplib`   |
| `from_docx(path)`                           | `python-docx`      |
| `from_excel(path, sheet)`                   | `openpyxl`         |
| `from_xml(path, text_xpath)`                | stdlib `xml.etree` |

### Niche Vector Stores

| Store                   | Dependencies    |
| ----------------------- | --------------- |
| `MilvusVectorStore`     | `pymilvus`      |
| `OpenSearchVectorStore` | `opensearch-py` |
| `LanceVectorStore`      | `lancedb`       |

### Niche Toolbox Modules

| Module               | Tools                                                          |
| -------------------- | -------------------------------------------------------------- |
| `email_tools.py`     | `send_email`, `read_inbox`, `search_emails`                    |
| `calendar_tools.py`  | `create_event`, `list_events`, `find_free_slots`               |
| `browser_tools.py`   | `navigate`, `click`, `extract_text`, `screenshot` (Playwright) |
| `financial_tools.py` | `stock_price`, `exchange_rate`, `market_summary`               |

### Community Features

- **Tool Marketplace/Registry**: Publish and discover community `@tool` functions
- **Visual Agent Builder**: Web UI for designing agent configurations (generates YAML)
- **Enhanced HTML Trace Viewer**: Interactive timeline with filter, search, cost breakdown

| Feature                       | Status | Impact | Effort |
| ----------------------------- | ------ | ------ | ------ |
| **Niche Loaders (8)**         | 🟡 Low | Medium | Medium |
| **Niche Stores (3)**          | 🟡 Low | Low    | Medium |
| **Niche Toolbox (4 modules)** | 🟡 Low | Medium | Medium |
| **Tool Marketplace**          | 🟡 Low | High   | Large  |
| **Visual Agent Builder**      | 🟡 Low | Medium | Large  |

---

## Backlog (Unscheduled)

| Feature                  | Notes                               | Target   |
| ------------------------ | ----------------------------------- | -------- |
| Universal Vision Support | Unified vision API across providers | Deferred |
| Rate Limiting & Quotas   | Per-tool and per-user quotas        | v0.21.x  |
| CRM & Business Tools     | HubSpot, Salesforce integrations    | v0.21.x  |
