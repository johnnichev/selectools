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

## v0.16.0: Memory & Persistence

Focus: Durable conversation state, cross-session knowledge, and advanced memory strategies.

### Persistent Conversation Sessions

**Problem**: `ConversationMemory` is in-memory only. Process restarts lose all history. Chat applications need sessions that survive restarts.

**What it does**: `SessionStore` protocol with pluggable backends. Sessions auto-persist after each turn with TTL-based expiry.

**API**:

```python
from selectools.memory import JsonFileSessionStore

store = JsonFileSessionStore(directory="./sessions")
agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(session_store=store, session_id="user-123"),
)
result = agent.ask("What was my last question?")  # auto-persisted
```

**Scope**:

- `SessionStore` protocol: `save()`, `load()`, `list()`, `delete()`
- Three backends: JSON file, SQLite, Redis
- Auto-save after each turn
- TTL-based expiry
- Tool-pair-preserving trim on load

**Touches**: New `sessions.py`, `AgentConfig`, `agent/core.py`.

### Summarize-on-Trim

**Problem**: Old messages are silently dropped when history exceeds limits. Important early context is lost.

**What it does**: Before trimming, summarize the messages being removed and inject the summary as a system-level context message.

**API**:

```python
memory = ConversationMemory(
    max_messages=30,
    summarize_on_trim=True,
    summarize_provider=provider,
)
```

**Scope**:

- LLM-generated 2-3 sentence summary of trimmed messages
- Summary injected as system message at conversation start
- Configurable summary model (use a cheap model like Haiku)

**Touches**: `memory.py`, provider integration.

### Entity Memory

**Problem**: The agent can't track entities (people, orgs, projects) mentioned across turns. Each turn starts with no entity context.

**What it does**: Automatically extract named entities from conversation, maintain an entity registry, and inject relevant entity context into prompts.

**API**:

```python
from selectools.memory import EntityMemory

memory = EntityMemory(provider=provider)
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.ask("I'm working with Alice from Acme Corp on Project Alpha")
agent.ask("What project am I working on?")
# Agent knows: Alice (person, Acme Corp), Acme Corp (org), Project Alpha (project)
```

**Scope**:

- LLM-based entity extraction after each turn
- Entity types: person, organization, project, location, date, custom
- Entity registry: name → type, attributes, last mentioned
- System prompt injection of relevant entities
- Configurable: extraction model, max entities, relevance window

**Touches**: New `entity_memory.py`, `PromptBuilder` integration.

### Knowledge Graph Memory

**Problem**: Entity memory tracks individual entities but not relationships between them. "Alice manages Project Alpha" is lost.

**What it does**: Build a graph of (subject, relation, object) triples from conversations. Query the graph to inject relevant relationship context into prompts.

**API**:

```python
from selectools.memory import KnowledgeGraphMemory

memory = KnowledgeGraphMemory(provider=provider, storage="sqlite")
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.ask("Alice manages Project Alpha and reports to Bob")
# Graph: (Alice, manages, Project Alpha), (Alice, reports_to, Bob)

agent.ask("Who manages Project Alpha?")
# Relevant triples injected: (Alice, manages, Project Alpha)
```

**Scope**:

- LLM-based triple extraction
- Storage: in-memory dict (default), SQLite (persistent)
- Query: retrieve triples relevant to current query via keyword + embedding match
- System prompt injection of relevant triples
- Graph operations: add, query, merge, prune

**Touches**: New `knowledge_graph.py`, storage backend, `PromptBuilder`.

### Cross-Session Knowledge Memory

**Problem**: Even with persistent sessions, each session is isolated. There's no way for an agent to "remember" facts across conversations (e.g., user preferences, prior decisions).

**What it does**: A file-based or DB-backed knowledge memory with two layers: a daily log (append-only entries from the current day) and a long-term store (curated facts that persist indefinitely). A built-in `remember` tool lets the agent save facts explicitly. Relevant memories are auto-injected into the system prompt.

**API**:

```python
from selectools.memory import KnowledgeMemory

knowledge = KnowledgeMemory(
    directory="./workspace",
    recent_days=2,           # inject last 2 days into system prompt
    max_context_chars=5000,  # cap memory injection size
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(knowledge_memory=knowledge),
)
```

**Scope**:

- Daily log files (`memory/YYYY-MM-DD.md`) + persistent `MEMORY.md`
- Built-in `remember` tool: agent can save categorized facts
- System prompt auto-injection of recent and long-term memories
- Configurable retention and context window

**Touches**: New `knowledge.py` module, `PromptBuilder` integration, built-in tool in `toolbox/`.

| Feature                              | Priority  | Impact | Effort |
| ------------------------------------ | --------- | ------ | ------ |
| **Persistent Conversation Sessions** | ✅ Done   | High   | Medium |
| **Summarize-on-Trim**                | ✅ Done   | Medium | Small  |
| **Cross-Session Knowledge Memory**   | ✅ Done   | Medium | Medium |
| **Entity Memory**                    | ✅ Done   | High   | Medium |
| **Knowledge Graph Memory**           | ✅ Done   | High   | Large  |

---

## Implementation Order

```
v0.13.0  ✅ Structured Output + Safety Foundation (Complete)
         Tool-pair trimming → Structured output → Execution traces → Reasoning
         → Fallback providers → Batch → Tool policy → Human-in-the-loop

v0.14.0  ✅ AgentObserver Protocol + Production Hardening (Complete)
         AgentObserver (15 events) → LoggingObserver → OTel export
         → Model registry (145 models) → max_completion_tokens fix → 11 bug fixes

v0.14.1  ✅ Streaming & Provider Fixes (Complete)
         13 streaming bug fixes → 141 new tests → Unit tests for 6 untested modules

v0.15.0  ✅ Enterprise Reliability (Complete)
         Guardrails engine → Audit logging → Tool output screening → Coherence checking

v0.16.0  ✅ Memory & Persistence (Complete)
         Sessions → Summarize-on-trim → Knowledge memory → Entity memory → KG memory

v0.16.5  ✅ Design Patterns & Code Quality
         StepType/ModelType enums → Agent mixin split → Provider base class → Terminal actions
         → Async observers → Gemini 3.x thought signatures → Hooks deprecation → ADRs

v0.17.0  ✅ Eval Framework
         39 evaluators → A/B testing → snapshots → regression → HTML/JUnit → CI → templates

v0.17.1  🟡 MCP Client/Server
         MCPClient → mcp_tools() → MCPServer → MultiMCPClient → tool interop

v0.17.2  🟡 Multi-Agent Orchestration
         AgentGraph → GraphState → Checkpointing → Parallel nodes → SupervisorAgent

v0.18.0  🟡 Serve & Deploy
         FastAPI AgentRouter → Flask AgentBlueprint → Playground → Templates → YAML config

v0.19.0  🟡 Connector Expansion
         CSV/JSON/HTML/URL/Markdown loaders → FAISS/Qdrant/pgvector stores
         → Code/search tools → SaaS loaders → GitHub/DB toolbox

v0.20.0  🟡 Polish & Community
         HTML trace viewer → Niche integrations → Tool marketplace
```

---

## v0.16.5: Design Patterns & Code Quality ✅

Focus: Structural refactoring to prevent the class of bugs found in v0.16.0–v0.16.4 and prepare a clean foundation for v0.17.0 multi-agent orchestration.

### Highlights

- **Agent decomposed into 4 mixins** — `core.py` 3128→1448 lines (-54%)
- **StepType + ModelType** converted to `str, Enum` for type safety
- **Terminal actions** — `@tool(terminal=True)` and `stop_condition` callback
- **AsyncAgentObserver** — async lifecycle hooks with `blocking` flag
- **Gemini 3.x thought signatures** — `ToolCall.thought_signature` field
- **Hooks deprecated** — wrapped via `_HooksAdapter`, single observer pipeline
- **OpenAI/Ollama share base class** — Template Method pattern (-75% duplication)
- **163 new tests** (total: 1640), 53 architecture fitness tests
- **6 ADRs** in `docs/decisions/`

See [CHANGELOG](CHANGELOG.md) for full details.

---

## v0.16.6: Gemini thought_signature crash fix ✅

Fixed `UnicodeDecodeError` when Gemini 3.x returns non-UTF-8 binary `thought_signature`. Replaced UTF-8 encode/decode with base64 across all 5 affected locations. 2 new regression tests (total: 1642).

---

## v0.16.7: Cleanup ✅

Removed unused CLI module (`cli.py` + console script entry point). Completed README example table (28-38). Fixed stale counts across CONTRIBUTING.md, ARCHITECTURE.md, QUICKSTART.md, AGENT.md. Total: 1620 tests.

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

| Feature | Status | Impact | Effort |
|---|---|---|---|
| **EvalSuite + TestCase** | 🟡 High | High | Medium |
| **Built-in Evaluators (6)** | 🟡 High | High | Medium |
| **EvalReport** | 🟡 High | Medium | Small |
| **Dataset Loader (JSON/YAML)** | 🟡 Medium | Medium | Small |
| **Regression Detection** | 🟡 Medium | High | Medium |
| **HTML Report Export** | 🟡 Medium | Medium | Small |
| **JUnit XML for CI** | 🟡 Medium | Medium | Small |

---

## v0.17.2: Multi-Agent Orchestration

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
| **AgentGraph + GraphState** | 🟡 High   | High   | Large  |
| **Checkpointing**           | 🟡 High   | High   | Medium |
| **Parallel Nodes**          | 🟡 Medium | High   | Medium |
| **SupervisorAgent**         | 🟡 Medium | High   | Medium |

---

## v0.17.1: MCP Client/Server

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

| Feature | Status | Impact | Effort |
|---|---|---|---|
| **MCPClient** | 🟡 High | High | Medium |
| **mcp_tools() adapter** | 🟡 High | High | Small |
| **MCPServer** | 🟡 Medium | Medium | Medium |

---

## v0.18.0: Serve & Deploy

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

| Feature                     | Status    | Impact | Effort |
| --------------------------- | --------- | ------ | ------ |
| **FastAPI AgentRouter**     | 🟡 High   | High   | Medium |
| **Flask AgentBlueprint**    | 🟡 Medium | Medium | Small  |
| **Playground**              | 🟡 Medium | High   | Medium |
| **Trace Store**             | 🟡 High   | High   | Medium |
| **HTML Trace Export**       | 🟡 Medium | High   | Medium |
| **Agent Templates**         | 🟡 Medium | Medium | Small  |
| **YAML Config**             | 🟡 Low    | Medium | Small  |

---

## v0.19.0: Connector Expansion

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

## v0.20.0: Polish & Community

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

| Feature                    | Notes                                            |
| -------------------------- | ------------------------------------------------ |
| Tool Composition           | `@compose` decorator for chaining tools          |
| Universal Vision Support   | Unified vision API across providers              |
| AWS Bedrock Provider       | VPC-native model access (Claude, Llama, Mistral) |
| Rate Limiting & Quotas     | Per-tool and per-user quotas                     |
| Structured AgentConfig     | Group 41 fields into nested dataclasses (RetryConfig, CoherenceConfig, etc.) with backward compat |
| Enhanced Testing Framework | Snapshot testing, load tests                     |
| Documentation Generation   | Auto-generate docs from tool definitions         |
| Prompt Optimization        | Automatic prompt compression                     |
| CRM & Business Tools       | HubSpot, Salesforce integrations                 |

---

## Release History

### v0.16.1 - Consolidation & Hardening

- ✅ **6 bug fixes**: `arun()` tool_usage stats, `astream()` dead code after yield, `memory.from_dict()` boundary fix, `EntityMemory` thread safety, `KnowledgeMemory` thread safety, `SQLiteTripleStore` WAL mode
- ✅ **mypy**: Resolved all 5 type errors across `sessions.py` and `knowledge_graph.py` (0 errors now)
- ✅ **68 new tests** (total: 1487): Redis sessions, edge cases, memory boundary, memory integration, async memory, consolidation regression

### v0.16.0 - Memory & Persistence

- ✅ **Persistent Sessions**: `SessionStore` protocol with 3 backends (JSON file, SQLite, Redis), TTL expiry, auto-save/load via `AgentConfig`
- ✅ **Summarize-on-Trim**: LLM-generated summaries of trimmed messages, injected as system context; configurable provider/model
- ✅ **Entity Memory**: LLM-based entity extraction (person, org, project, location, date, custom), LRU-pruned registry, system prompt injection
- ✅ **Knowledge Graph Memory**: Triple extraction (subject, relation, object), in-memory and SQLite storage, keyword-based querying
- ✅ **Cross-Session Knowledge**: Daily log files + persistent `MEMORY.md`, auto-registered `remember` tool, system prompt injection
- ✅ **Memory Tools**: Built-in `remember` tool auto-registered when `knowledge_memory` is configured
- ✅ **4 new observer events**: `on_session_load`, `on_session_save`, `on_memory_summarize`, `on_entity_extraction` (total: 19)
- ✅ **5 new trace step types**: `session_load`, `session_save`, `memory_summarize`, `entity_extraction`, `kg_extraction`
- ✅ **182 new tests** (total: 1365)

### v0.15.0 - Enterprise Reliability

- ✅ **Guardrails Engine**: `GuardrailsPipeline` with 5 built-in guardrails (Topic, PII, Toxicity, Format, Length) and block/rewrite/warn actions
- ✅ **Audit Logging**: JSONL `AuditLogger` with 4 privacy levels, daily rotation, thread-safe writes
- ✅ **Tool Output Screening**: 15 prompt injection patterns, per-tool `screen_output=True` or global via config
- ✅ **Coherence Checking**: LLM-based intent verification before each tool execution
- ✅ **83 new tests** (total: 1183)

### v0.14.1 - Streaming & Provider Fixes

- ✅ **13 streaming bug fixes**: All providers' `stream()`/`astream()` now pass `tools` and yield `ToolCall` objects
- ✅ **Agent core fixes**: `_streaming_call`/`_astreaming_call` pass tools and don't stringify `ToolCall` objects
- ✅ **Ollama `_format_messages`**: Correct `TOOL` role mapping and `ASSISTANT` tool_calls inclusion
- ✅ **FallbackProvider `astream()`**: Error handling, failover, and circuit breaker support
- ✅ **141 new tests** (total: 1100): Regression tests, recording-provider tests, unit tests for 6 previously untested modules

### v0.14.0 - AgentObserver Protocol & Production Hardening

- ✅ **AgentObserver Protocol**: 15 lifecycle events with `run_id`/`call_id` correlation
- ✅ **LoggingObserver**: Structured JSON logs for ELK/Datadog
- ✅ **OTel Span Export**: `AgentTrace.to_otel_spans()` for OpenTelemetry
- ✅ **Model Registry Update**: 145 models with March 2026 pricing (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro)
- ✅ **OpenAI `max_completion_tokens`**: Auto-detection for GPT-5.x, GPT-4.1, o-series models
- ✅ **11 bug fixes**: Structured output parser bypass, policy bypass in parallel execution, memory trim observer gap, infinite recursion in batch+fallback, async policy timeout, None content handling, and more

### v0.13.0 - Structured Output, Observability & Safety

- ✅ **Structured Output Parsers**: Pydantic / JSON Schema `response_format` on `run()` / `arun()` / `ask()` with auto-retry
- ✅ **Execution Traces**: `result.trace` with `TraceStep` timeline (`llm_call`, `tool_selection`, `tool_execution`, `error`)
- ✅ **Reasoning Visibility**: `result.reasoning` and `result.reasoning_history` extracted from LLM responses
- ✅ **Provider Fallback Chain**: `FallbackProvider` with circuit breaker and `on_fallback` callback
- ✅ **Batch Processing**: `agent.batch()` / `agent.abatch()` with `max_concurrency` and per-request error isolation
- ✅ **Tool-Pair-Aware Trimming**: `ConversationMemory` preserves tool_use/tool_result pairs during sliding window trim
- ✅ **Tool Policy Engine**: `ToolPolicy` with glob-based allow/review/deny rules and argument-level conditions
- ✅ **Human-in-the-Loop Approval**: `confirm_action` callback for `review` tools with `approval_timeout`

### v0.12.x - Hybrid Search, Reranking, Advanced Chunking & Dynamic Tools

- ✅ **BM25**: Pure-Python Okapi BM25 keyword search; configurable k1/b; stop word removal; zero dependencies
- ✅ **HybridSearcher**: Vector + BM25 fusion via RRF or weighted linear combination
- ✅ **HybridSearchTool**: Agent-ready `@tool` with source attribution and score thresholds
- ✅ **FusionMethod**: `RRF` (rank-based) and `WEIGHTED` (normalised score) strategies
- ✅ **Reranker ABC**: Protocol for cross-encoder reranking with `rerank(query, results, top_k)`
- ✅ **CohereReranker**: Cohere Rerank API v2 (`rerank-v3.5` default)
- ✅ **JinaReranker**: Jina AI Rerank API (`jina-reranker-v2-base-multilingual` default)
- ✅ **HybridSearcher integration**: Optional `reranker=` param for post-fusion re-scoring
- ✅ **SemanticChunker**: Embedding-based topic-boundary splitting; cosine similarity threshold
- ✅ **ContextualChunker**: LLM-generated context prepended to each chunk (Anthropic-style contextual retrieval)
- ✅ **ToolLoader**: Discover `@tool` functions from modules, files, and directories; hot-reload support
- ✅ **Agent dynamic tools**: `add_tool`, `add_tools`, `remove_tool`, `replace_tool` with prompt rebuild

### v0.12.0 - Response Caching

- ✅ **InMemoryCache**: Thread-safe LRU + TTL cache with `OrderedDict`; zero dependencies
- ✅ **RedisCache**: Distributed TTL cache for multi-process deployments (optional `redis` dep)
- ✅ **CacheKeyBuilder**: Deterministic SHA-256 keys from (model, prompt, messages, tools, temperature)
- ✅ **Agent Integration**: `AgentConfig(cache=...)` checks cache before every provider call

### v0.11.0 - Streaming & Parallel Execution

- ✅ **E2E Streaming**: Native tool streaming via `Agent.astream` with `Union[str, ToolCall]` provider protocol
- ✅ **Parallel Tool Execution**: `asyncio.gather` for async, `ThreadPoolExecutor` for sync; enabled by default
- ✅ **Full Type Safety**: 0 mypy errors across 80+ source and test files

### v0.10.0 - Critical Architecture

- ✅ **Native Function Calling**: OpenAI, Anthropic, and Gemini native tool APIs
- ✅ **Context Propagation**: `contextvars.copy_context()` for async tool execution
- ✅ **Routing Mode**: `AgentConfig(routing_only=True)` for classification without execution

### v0.9.0 - Core Capabilities & Reliability

- ✅ **Custom System Prompt**: `AgentConfig(system_prompt=...)` for domain instructions
- ✅ **Structured AgentResult**: `run()` returns `AgentResult` with tool calls, args, and iterations
- ✅ **Reusable Agent Instances**: `Agent.reset()` clears history/memory for clean reuse

### v0.8.0 - Embeddings & RAG

- ✅ **Full RAG Stack**: VectorStore (Memory/SQLite/Chroma), Embeddings (OpenAI/Gemini), Document Loaders
- ✅ **RAG Tools**: `RAGTool` and `SemanticSearchTool` for knowledge base queries

### v0.6.0 - High-Impact Features

- ✅ **Observability Hooks**: `on_agent_start`, `on_tool_end` lifecycle events
- ✅ **Streaming Tools**: Generators yield results progressively

### v0.5.0 - Production Readiness

- ✅ **Cost Tracking**: Token counting and USD estimation
- ✅ **Better Errors**: PyTorch-style error messages with suggestions
