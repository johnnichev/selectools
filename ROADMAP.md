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

v0.18.0 ✅ Multi-Agent Orchestration + Composable Pipelines
AgentGraph → GraphState → Typed reducers → Resume-from-yield interrupts
→ Scatter fan-out → Checkpointing → SupervisorAgent → Graph visualization
→ Pipeline → @step → | operator → parallel() → branch()

v0.19.0 ✅ Serve, Deploy & Complete Composition
selectools serve CLI → Playground UI → YAML config → 5 agent templates
→ Structured AgentConfig → compose() → retry() / cache_step()
→ Type-safe step contracts → Streaming composition → pipeline.astream()
→ PostgresCheckpointStore → TraceStore (3 backends) → selectools doctor

v0.19.1 ✅ Advanced Agent Patterns
PlanAndExecute → ReflectiveAgent → Debate → TeamLead → 50+ evaluators

v0.19.2 ✅ Enterprise Hardening
Security audit → Stability markers (@stable/@beta/@deprecated) → Deprecation policy
→ Compatibility matrix → trace_to_html() waterfall viewer → SBOM
→ Property-based tests (Hypothesis) → Concurrency smoke suite → 5 production simulations
→ 3,344 tests, 76 examples

v0.19.3 ✅ Stability Markers Applied to All Public APIs
@stable on 60+ core symbols → @beta on 30+ orchestration/pipeline/patterns symbols
→ Full stability introspection via .__stability__ on every exported class and function

v0.20.0 ✅ Visual Agent Builder
Zero-install web UI → Drag-drop graph builder → YAML/Python export
→ Self-contained HTML (no React, no CDN) → One command: selectools serve --builder

v0.20.1 ✅ Builder Polish + Starlette + GitHub Pages
UI polish (20 features) → _static/ architecture split → Starlette ASGI app
→ Serverless mode (client-side AI/runs) → GitHub Pages deployment → Design system

v0.21.0 ✅ Connector Expansion + Multimodal + Observability
FAISS → Qdrant → pgvector vector stores
→ Azure OpenAI provider → Multimodal messages (images, audio)
→ CSV/JSON/HTML/URL document loaders
→ OTel observer → Langfuse observer
→ Code execution, web search, GitHub, DB toolbox tools

v1.0.0 🟡 Stable Release
API freeze → Stability markers on all modules → Deprecation policy
→ Security audit published → Compatibility matrix → 0.x→1.0 migration guide
→ PyPI classifier: Production/Stable

---

## v0.19.1: Advanced Agent Patterns ✅

Higher-level agent architectures built on the v0.18.0 orchestration primitives. Closes the "Advanced patterns" competitive gap. Each pattern is a standalone class — they wire up the AgentGraph topology for you.

### PlanAndExecute Agent

```python
from selectools.patterns import PlanAndExecuteAgent

agent = PlanAndExecuteAgent(
    planner=planner_agent,
    executors={"research": researcher, "write": writer, "review": reviewer},
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
    stop_condition="approved",
)
result = agent.run("Draft a press release")
# Actor produces draft → Critic evaluates → Actor revises → repeat until approved
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
    delegation_strategy="dynamic",  # or "sequential", "parallel"
)
result = agent.run("Investigate and fix the billing discrepancy")
# Lead delegates tasks, reviews work, coordinates handoffs
```

### Expanded Eval Suite (50 evaluators, up from 39)

11 new evaluators across two categories:

**New deterministic (+8):** `ReadabilityEvaluator`, `AgentTrajectoryEvaluator`, `ToolEfficiencyEvaluator`, `SemanticSimilarityEvaluator`, `MultiTurnCoherenceEvaluator`, `JsonSchemaEvaluator`, `KeywordDensityEvaluator`, `ForbiddenWordsEvaluator`

**New LLM-as-judge (+4):** `FactConsistencyEvaluator`, `CustomRubricEvaluator`, `AnswerAttributionEvaluator`, `StepReasoningEvaluator`


| Feature             | Status | Impact | Effort |
| ------------------- | ------ | ------ | ------ |
| **PlanAndExecute**  | ✅      | High   | Medium |
| **ReflectiveAgent** | ✅      | High   | Medium |
| **Debate**          | ✅      | Medium | Medium |
| **TeamLead**        | ✅      | Medium | Medium |
| **50 evaluators**   | ✅      | High   | Medium |


### Quality Infrastructure

- **Ralph loop** — autonomous hunt-and-fix convergence system (`scripts/ralph_bug_hunt.sh`, `/ralph-bug-hunt` skill)
- **Bandit in CI** — security scan job on every push
- **Property-based tests** — Hypothesis suite for structural invariants
- **Thread-safety smoke suite** — 10-thread × 20-op concurrency tests
- **Production simulations** — 16 integration tests covering memory pressure, provider failover, tool errors, concurrent load

---

## v0.19.2: Enterprise Hardening ✅

Focus: Production readiness and developer trust signals before the Visual Agent Builder in v0.20.0.

### Enterprise Hardening


| Feature                                                                 | Status | Impact | Effort |
| ----------------------------------------------------------------------- | ------ | ------ | ------ |
| **Security audit** (bandit + manual nosec review)                       | ✅      | High   | Medium |
| **Stability markers** (`@stable`, `@beta`, `@deprecated`)               | ✅      | Medium | Small  |
| **Deprecation policy** (2-version window, `docs/DEPRECATION_POLICY.md`) | ✅      | Medium | Small  |
| **Compatibility matrix** (Python × provider SDK × optional deps)        | ✅      | Medium | Small  |
| **SBOM** (`sbom.json` via CycloneDX, published in repo)                 | ✅      | Low    | Small  |
| **Enhanced trace viewer** (`trace_to_html()` waterfall HTML)            | ✅      | High   | Medium |


### Quality Infrastructure


| Feature                               | Status | Impact | Effort |
| ------------------------------------- | ------ | ------ | ------ |
| **Property-based tests** (Hypothesis) | ✅      | High   | Medium |
| **Thread-safety smoke suite**         | ✅      | High   | Medium |
| **Production simulations** (5 new)    | ✅      | High   | Medium |


---

## v0.19.3: Stability Markers Applied ✅

Focus: Apply `@stable` and `@beta` markers to every public symbol in the library, completing the stability annotation work started in v0.19.2.

### Stable APIs (60+ symbols)

Core types, providers, agent, memory, tools, evals, guardrails, sessions, knowledge, cache, cancellation, token estimation, analytics, audit — all marked `@stable`. Breaking changes to these require a major version bump.

### Beta APIs (30+ symbols)

Orchestration (`AgentGraph`, `SupervisorAgent`), pipelines (`Pipeline`, `@step`, `parallel`, `branch`), patterns (`PlanAndExecuteAgent`, `ReflectiveAgent`, `DebateAgent`, `TeamLeadAgent`), and composition (`compose`) — marked `@beta`. These may change in a minor release.

### Introspection

```python
from selectools import Agent, AgentGraph, PlanAndExecuteAgent
print(Agent.__stability__)              # "stable"
print(AgentGraph.__stability__)         # "beta"
print(PlanAndExecuteAgent.__stability__)  # "beta"
```

---

## v0.20.0: Visual Agent Builder ✅

The headline feature: a zero-install web UI for designing, testing, and exporting agent configurations. Served by `selectools serve --builder` — no separate app, no subscription, no desktop install required.

**Why a dedicated release:** LangGraph Studio is a paid desktop app. AutoGen Studio is a separate project. selectools ships a full visual builder in one command. This deserves its own announcement.

A web-based UI for designing, testing, and exporting agent configurations. Zero-install — served by `selectools serve --builder`.

```
┌─────────────────────────────────────────────────────┐
│  Visual Agent Builder                    [Export]   │
├─────────────┬───────────────────────────────────────┤
│             │                                       │
│  Components │    ┌──────────┐    ┌──────────┐       │
│  ─────────  │    │ Planner  │───▶│  Writer  │       │
│  ☐ Agent    │    └──────────┘    └────┬─────┘       │
│  ☐ Tool     │                         │             │
│  ☐ Router   │                    ┌────▼─────┐       │
│  ☐ Gate     │                    │ Reviewer │       │
│  ☐ Parallel │                    └──────────┘       │
│             │                                       │
├─────────────┼───────────────────────────────────────┤
│  Properties │    Model: gpt-4o  │ Tools: 3          │
│  ─────────  │    Strategy: plan │ Budget: $0.50     │
│  Name: ...  │                                       │
│  Model: ... │    [▶ Test Run]   [💾 Save YAML]      │
└─────────────┴───────────────────────────────────────┘
```

**Features:**

- Drag-and-drop graph builder for AgentGraph topologies
- Node palette: Agent, Tool, Router (conditional), Gate (HITL), Parallel group
- Visual edge wiring with routing condition editor
- Per-node configuration panel (model, tools, system prompt, budget)
- Live test: run the graph against real providers from the UI
- Export: generates `agent.yaml` or Python code
- Import: load existing YAML configs into the builder
- Served by selectools: `selectools serve --builder` (zero frontend deps)
- Built as self-contained HTML/JS (same pattern as playground.py)

**Technical approach:**

- Single HTML file with embedded JS (no React, no build step)
- Canvas-based graph rendering (or SVG with drag handlers)
- Backend: new `/builder` endpoint on AgentServer
  - `GET /builder` — serves the HTML
  - `POST /builder/validate` — validates graph structure
  - `POST /builder/export` — generates YAML or Python
  - `POST /builder/run` — executes the designed graph
- State stored in browser localStorage (no server state)

**Why this matters:**

- LangGraph has LangGraph Studio (paid, desktop app)
- CrewAI has no visual builder
- AutoGen has AutoGen Studio (separate app)
- selectools: zero-install, runs in browser, exports to YAML/Python


| Feature                                    | Status | Impact | Effort |
| ------------------------------------------ | ------ | ------ | ------ |
| **Graph canvas (drag-drop nodes + edges)** | ✅     | High   | Large  |
| **Node configuration panel**               | ✅     | High   | Medium |
| **YAML export/import**                     | ✅     | High   | Small  |
| **Python code export**                     | ✅     | Medium | Small  |
| **Live test execution**                    | ✅     | High   | Medium |
| **Self-contained HTML (no build step)**    | ✅     | High   | Medium |


---

## v0.20.1: Builder Polish + Starlette + GitHub Pages ✅

UI polish (20 features), `_static/` architecture split, Starlette ASGI app, serverless mode (client-side AI/runs), GitHub Pages deployment, design system.

- Visual builder live at: https://selectools.dev/builder/
- Examples gallery: https://selectools.dev/examples/
- 4,612 tests (95% coverage), 76 examples, 50 evaluators, 152 models

| Feature                                      | Status | Impact | Effort |
| -------------------------------------------- | ------ | ------ | ------ |
| **UI polish (20 features)**                  | ✅     | High   | Medium |
| **_static/ architecture split**              | ✅     | Medium | Small  |
| **Starlette ASGI app**                       | ✅     | High   | Medium |
| **Serverless mode (client-side AI/runs)**    | ✅     | High   | Medium |
| **GitHub Pages deployment**                  | ✅     | High   | Small  |
| **Design system**                            | ✅     | Medium | Small  |
| **Eval badges on builder nodes**             | ✅     | Medium | Small  |


---

## v0.21.0: Connector Expansion + Multimodal + Observability ✅

**Shipped:** FAISS + Qdrant + pgvector vector stores, CSV/JSON/HTML/URL document loaders, Azure OpenAI provider, OpenTelemetry + Langfuse observers, multimodal `ContentPart` + `image_message()` across OpenAI/Anthropic/Gemini/Ollama, new code/search/github/db toolbox modules (9 tools). 5215 tests (95% coverage), 88 examples, 5 LLM providers, 7 vector stores, 152 models.

Close integration gaps, add multimodal support (images/audio), and ship enterprise-grade observability (OTel + Langfuse). Full spec: `.private/07-v0.21.0-connector-expansion.md`

### Current Inventory

| Category            | Count    | Items                                    |
| ------------------- | -------- | ---------------------------------------- |
| Document Loaders    | 4        | text, file, directory, PDF               |
| Vector Stores       | 4        | Memory, SQLite, Chroma, Pinecone         |
| Embedding Providers | 4        | OpenAI, Anthropic/Voyage, Gemini, Cohere |
| LLM Providers       | 5        | OpenAI, Anthropic, Gemini, Ollama, Fallback |
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
| `**code_tools.py`     | `execute_python`, `execute_shell`                               | stdlib `subprocess`            | Medium       | #1 most-used tool in agent frameworks |
| `**search_tools.py`** | `google_search`, `duckduckgo_search`                            | `duckduckgo_search` (optional) | Small-Medium | #2 most-used tool category            |
| `**github_tools.py**` | `create_issue`, `list_issues`, `create_pr`, `get_file_contents` | `requests` (existing)          | Medium       | Developer workflow automation         |
| `**db_tools.py**`     | `query_database`, `list_tables`, `describe_table`               | `sqlalchemy` (optional)        | Medium       | Enterprise data access                |


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


| Feature                         | Status | Impact | Effort |
| ------------------------------- | ------ | ------ | ------ |
| **Multimodal messages**         | 🟡     | High   | Medium |
| **OTel observer**               | 🟡     | High   | Medium |
| **Azure OpenAI provider**       | 🟡     | High   | Small  |
| **Langfuse observer**           | 🟡     | High   | Small  |
| **FAISS Vector Store**          | 🟡     | High   | Small  |
| **Qdrant Vector Store**         | 🟡     | Medium | Small  |
| **pgvector Store**              | 🟡     | High   | Small  |
| **CSV/JSON/HTML/URL Loaders**   | 🟡     | High   | Small  |
| **Code Execution Tools**        | 🟡     | High   | Medium |
| **Web Search + GitHub Tools**   | 🟡     | High   | Small  |
| **Database Query Tools**        | 🟡     | Medium | Small  |


---

## Backlog (Unscheduled — Priority Ordered)

> **Research basis:** Competitive analysis of Agno (39k stars), PraisonAI (6.9k stars),
> and Superagent (6.5k stars) conducted 2026-04-10. Full findings in memory files.
>
> **Strategic thesis:** selectools wins on depth (50 evals, 7 vector stores, graph
> orchestration, pattern agents, 5,203 tests). Close the breadth gap cheaply, own the
> "production-ready" narrative, adopt the emerging A2A standard.

---

### P0 — Ship Next (High Impact, Low-Medium Effort)

#### Tool-Call Loop Detection
**Source:** PraisonAI's "doom loop detection"
**Gap:** selectools has graph-level loop/stall detection in AgentGraph, but no tool-call-level detection. An agent calling the same tool with the same args 20 times burns budget with no progress — `max_iterations` is too blunt.
**Spec:** Three parallel detectors running per tool execution:
- **Generic Repeat** — identical tool + identical args N times in a row
- **Poll No Progress** — tools matching polling patterns ("status", "check", "poll") returning unchanged results consecutively
- **Ping Pong** — alternating oscillation between two tools without advancement
Two-tier response: warn at `warn_threshold` (default 10) → block at `critical_threshold` (default 20).
```python
# Target API
from selectools.agent.loop_detection import LoopDetectionConfig
agent = Agent(tools, provider=provider, config=AgentConfig(
    loop_detection=LoopDetectionConfig(
        enabled=True,
        history_size=30,
        warn_threshold=10,
        critical_threshold=20,
        detectors={"generic_repeat": True, "poll_no_progress": True, "ping_pong": True}
    )
))
```
**Implementation:** New `agent/loop_detection.py`. Hook into `_process_response()` tool execution path. stdlib only (`hashlib`, `json`). Zero overhead when disabled.
**Effort:** Low (1-2 days). Pure Python, no deps.

#### Agentic Memory (Memory-as-Tool)
**Source:** Agno's `enable_agentic_memory=True`
**Gap:** selectools memory is always-on and passive — ConversationMemory stores everything, EntityMemory extracts automatically. The agent has no agency over what it remembers. For long-running agents, not every turn is worth persisting.
**Spec:** Two memory tools injected when `agentic_memory=True`:
- `remember(key, value, importance=0.8)` — agent explicitly stores a fact
- `recall(query, limit=5)` — agent explicitly retrieves relevant memories
Backed by existing `KnowledgeMemory` (already has importance scores, TTL, 4 backends).
```python
# Target API
agent = Agent(tools, provider=provider, config=AgentConfig(
    memory=MemoryConfig(agentic_memory=True, store=SQLiteKnowledgeStore("memory.db"))
))
# Agent now has remember() and recall() as tools alongside user tools
```
**Implementation:** New `agent/memory_tools.py` that wraps KnowledgeMemory as Tool objects. Inject into tool list during `_prepare_run()` when `agentic_memory=True`.
**Effort:** Low (1-2 days). Wraps existing KnowledgeMemory.

#### Agent-as-API (Production Serve)
**Source:** Agno's `AgentOS` — one line generates production FastAPI app
**Gap:** selectools serve/ has builder UI + playground, but no auto-generated production REST API. Users who want to deploy a selectools agent as an API must write their own FastAPI wrapper.
**Spec:** Auto-generate production endpoints from any Agent:
- `POST /v1/chat` — single-turn completion (JSON request/response)
- `POST /v1/chat/stream` — streaming completion (SSE)
- `POST /v1/sessions` — create session
- `GET /v1/sessions/{id}` — get session history
- `DELETE /v1/sessions/{id}` — delete session
- `GET /v1/health` — health check
Per-user isolation via `user_id` header. Optional API key auth.
```python
# Target API
from selectools.serve import AgentAPI
app = AgentAPI(agents=[my_agent, my_other_agent], auth_key="sk-...")
# Starlette ASGI app — run with: uvicorn app:app
```
Or via CLI: `selectools serve agent.yaml --api --port 8000`
**Implementation:** New `serve/api.py` building on existing Starlette infrastructure in `_starlette_app.py`. Standardized JSON schema for requests/responses. Session management via existing SessionStore backends.
**Effort:** Medium (3-5 days). Starlette already exists, plumbing is there.

---

### P1 — Ship Soon (High Impact, Medium Effort)

#### LiteLLM Provider Wrapper
**Source:** PraisonAI (24+ providers via litellm), Agno (40+ native providers)
**Gap:** selectools has 5 native providers (OpenAI, Anthropic, Gemini, Ollama, Azure OpenAI). Enterprise users need DeepSeek, Mistral, Groq, Together, Cohere, Fireworks, Bedrock, and more.
**Spec:** A `LiteLLMProvider` that delegates to the `litellm` library, instantly supporting 100+ models.
```python
# Target API
from selectools.providers.litellm_provider import LiteLLMProvider
provider = LiteLLMProvider(model="deepseek/deepseek-chat")
provider = LiteLLMProvider(model="groq/llama-3.1-70b")
provider = LiteLLMProvider(model="bedrock/anthropic.claude-3-sonnet")
agent = Agent(tools, provider=provider)
```
Must implement full Provider protocol: complete/acomplete/stream/astream, tool calling, structured output. Optional dep: `litellm>=1.0.0`.
**Implementation:** New `providers/litellm_provider.py`. Map selectools Message/ToolCall to litellm format. Register in `[providers]` extras group.
**Effort:** Medium (2-3 days). litellm handles the hard provider-specific work.
**Note:** Native providers remain for maximum control; LiteLLM is the "long tail" solution.

#### Cost-Optimized Model Router
**Source:** PraisonAI's "Model Router" / "RouterAgent"
**Gap:** selectools has FallbackProvider for reliability (try primary → secondary on failure) and pricing.py with cost data for 152 models, but no cost-optimized routing. Users manually pick models.
**Spec:** A `RouterProvider` that wraps multiple providers and routes based on task complexity + cost:
- Classify input complexity (simple factual → complex reasoning → code generation)
- Map to cheapest model capable of handling that complexity class
- Fall back to more expensive model if cheap model fails quality threshold
```python
# Target API
from selectools.providers import RouterProvider, OpenAIProvider, AnthropicProvider
router = RouterProvider(
    providers={
        "fast": OpenAIProvider(model="gpt-4o-mini"),  # $0.15/1M input
        "smart": AnthropicProvider(model="claude-sonnet-4-6"),  # $3/1M input
        "power": OpenAIProvider(model="gpt-5.4-pro"),  # $10/1M input
    },
    strategy="cost_optimized",  # or "quality_first", "balanced"
)
agent = Agent(tools, provider=router)
```
**Implementation:** New `providers/router.py`. Complexity classifier can be rule-based (tool count, input length, keyword detection) or LLM-based. Builds on FallbackProvider architecture.
**Effort:** Medium (3-5 days). Routing logic is the novel part.

#### A2A Protocol (Agent-to-Agent Communication)
**Source:** PraisonAI, Google-backed emerging standard
**Gap:** selectools has MCP for tool interop but no agent-to-agent communication protocol. Already in existing backlog for v0.22.0.
**Spec:** Two HTTP endpoints on existing Starlette serve infrastructure:
- `GET /.well-known/agent.json` — Agent Card (auto-generated from AgentConfig: name, description, capabilities, tools list)
- `POST /a2a` — JSON-RPC message handler (receive tasks, return results)
Task lifecycle: submitted → working → input-required → completed/failed/cancelled.
Message format: JSON-RPC with multimodal content parts (text, file, data).
Optional bearer token authentication on POST endpoint.
```python
# Target API — serving
from selectools.serve import A2AServer
server = A2AServer(agent=my_agent, auth_token="sk-...")
server.serve(port=8000)

# Target API — consuming
from selectools.a2a import A2AClient
client = A2AClient("https://other-agent.example.com")
card = await client.discover()  # reads /.well-known/agent.json
result = await client.send_task("Research quantum computing trends")
```
**Implementation:** New `a2a/` module with server.py + client.py. Server builds on serve/_starlette_app.py. Agent Card auto-generated from AgentConfig metadata.
**Effort:** Medium (3-5 days). Two routes + JSON-RPC message handler.

#### Expanded Toolbox (40 → 80+ tools)
**Source:** Agno has 131 built-in tools across 15 categories
**Gap:** selectools has 40+ tools across 10 categories. Missing enterprise-critical categories: communication (Slack, Discord, Email), project management (Notion, Linear, Jira), cloud (AWS S3, GCS), media (image generation).
**Priority additions (by user demand):**

| Category | Tools to add | Deps | Effort |
|---|---|---|---|
| **Slack** | send_message, read_channel, search | `slack-sdk` | Small |
| **Discord** | send_message, read_channel | `discord.py` | Small |
| **Email** | send_email, read_inbox | `smtplib`/`imaplib` (stdlib) | Small |
| **Notion** | create_page, search, update_page | `requests` | Small |
| **Linear** | create_issue, list_issues, update_issue | `requests` | Small |
| **AWS S3** | list_objects, get_object, put_object | `boto3` | Small |
| **Browser** | scrape_page, screenshot, click | `playwright` | Medium |
| **Image Gen** | generate_image (DALL-E) | `openai` (existing) | Small |
| **Calculator** | evaluate_expression, unit_convert | stdlib `ast` | Small |
| **PDF** | extract_text, extract_tables | `pdfplumber` | Small |

**Implementation:** New files in `src/selectools/toolbox/`. Follow existing @tool pattern. All deps optional with lazy imports. Register in `get_tools_by_category()`.
**Effort:** Medium total (1 day per category, parallelizable).

---

### P2 — Important but Not Urgent

#### Tool Result Compression
**Source:** Agno's `compress_tool_results=True`
**Gap:** selectools has CompressConfig for prompt compression but doesn't compress individual tool results. Verbose tool outputs (e.g., web scrape returning 10KB HTML) waste context.
**Spec:** When enabled, tool results exceeding a character threshold are summarized by a fast LLM before being added to the conversation.
```python
config = AgentConfig(tool=ToolConfig(compress_results=True, compress_threshold=2000))
```
**Implementation:** Add compression step in `_process_response()` after tool execution, before appending to messages. Use CompressConfig's existing compression logic.
**Effort:** Low (1 day).

#### Session History Search
**Source:** Agno's cross-session query capability
**Gap:** selectools session stores support save/load by session_id but can't search across sessions. An agent can't "remember what we discussed last Tuesday."
**Spec:** Add `search(query, user_id, limit)` method to SessionStore protocol. SQLiteSessionStore and RedisSessionStore implement full-text or embedding-based search.
```python
store = SQLiteSessionStore("sessions.db")
results = store.search("billing discrepancy", user_id="user-123", limit=5)
# Returns: list of (session_id, relevance_score, matched_messages)
```
**Implementation:** Add FTS5 index to SQLiteSessionStore. Add `SEARCH` command to RedisSessionStore. Protocol change requires @beta marker.
**Effort:** Medium (2-3 days).

#### Memory Tiering with Auto-Promotion
**Source:** PraisonAI's 4-tier memory with importance scoring
**Gap:** selectools has ConversationMemory, EntityMemory, KnowledgeMemory, KnowledgeGraphMemory as separate systems. They don't compose into a unified lifecycle with auto-promotion.
**Spec:** A `UnifiedMemory` that orchestrates all four:
- Short-term (ConversationMemory): rolling window, items auto-expire
- Long-term (KnowledgeMemory): items above `importance_threshold` auto-promoted from STM
- Entity (EntityMemory): structured entity tracking
- Episodic: date-based interaction history with configurable retention
Context compaction: auto-summarize when hitting 70% of token limit.
Importance scoring: LLM-based or rule-based (names=0.9, preferences=0.75, locations=0.6).
```python
config = AgentConfig(memory=MemoryConfig(
    unified=True,
    importance_threshold=0.7,
    short_term_limit=100,
    long_term_limit=1000,
    episodic_retention_days=30,
    auto_promote=True,
))
```
**Implementation:** New `unified_memory.py` orchestrating existing memory backends.
**Effort:** High (5-7 days). Requires importance scoring + lifecycle management.

#### Agent-Level Human-in-the-Loop
**Source:** Agno's approval workflows
**Gap:** selectools has InterruptRequest in graphs + ConfirmAction in ToolConfig. But a standalone Agent can't pause mid-execution for approval on arbitrary conditions (e.g., confidence below threshold, cost above limit).
**Spec:** Extend InterruptRequest to work outside of AgentGraph:
```python
config = AgentConfig(tool=ToolConfig(
    require_approval=["execute_shell", "send_email"],  # named tools
    approval_handler=my_callback,  # sync/async callable
))
```
**Implementation:** Lift InterruptRequest + checkpoint machinery from orchestration to agent level. Integrate with tool execution path.
**Effort:** Medium (3-4 days).

#### Planning-as-Config Flag
**Source:** PraisonAI's `planning=True`
**Gap:** selectools has PlanAndExecuteAgent as a separate pattern class. Users can't add planning to any existing agent with a config flag.
**Spec:** When `planning=True`, the agent auto-decomposes complex inputs before executing:
```python
config = AgentConfig(planning=PlanningConfig(
    enabled=True, llm="gpt-4o", auto_approve=True, reasoning=True
))
agent = Agent(tools, provider=provider, config=config)
# Agent internally: plan → approve → execute steps → synthesize
```
**Implementation:** Wrap existing PlanAndExecuteAgent logic into a mixin that activates via config. Reuses planner/executor infrastructure.
**Effort:** Low-Medium (2-3 days).

---

### P3 — Future / Watch

#### Shadow Git Checkpoints
**Source:** PraisonAI
File-level workspace snapshots via hidden git repo, independent of user's git history. Relevant if selectools moves toward coding agent use cases.
**Effort:** Medium.

#### Multi-Channel Bot Gateway
**Source:** PraisonAI (Telegram, Discord, Slack, WhatsApp routing)
Single routing layer for deploying agents to messaging platforms. Better as separate package.
**Effort:** High.

#### ML-Based Guard Models
**Source:** Superagent's open-weight 0.6B-4B parameter models
Prompt injection detection running locally, no API calls. Could wrap their models as a GuardrailProvider.
**Effort:** High (integration medium, but model hosting is the challenge).

#### Learning System
**Source:** Agno
Decision logging + preference tracking for continuous agent improvement over time.
**Effort:** High.

#### More Database Backends
**Source:** Agno (MongoDB, Firestore, DynamoDB, SurrealDB)
selectools has SQLite, PostgreSQL, Redis. NoSQL/cloud databases on demand.
**Effort:** Medium per backend.

#### Reasoning-as-Tool
**Source:** Agno's three reasoning modes
Reasoning step as composable tool (not just prompt strategy). Explicit min/max reasoning steps.
**Effort:** Medium.

#### Cron / Scheduled Agents
**Source:** PraisonAI
Background scheduling for periodic agent tasks (monitoring, reporting, cleanup).
**Effort:** Medium.

#### Episodic Memory
**Source:** PraisonAI
Date-based interaction history with configurable retention period and automatic cleanup.
**Effort:** Medium.

---

### Previously Planned (Retained)

| Feature                          | Notes                                            | Target  |
| -------------------------------- | ------------------------------------------------ | ------- |
| AWS Bedrock provider             | boto3 wrapper, enterprise gateway (or via LiteLLM) | v0.22.0 |
| Durable execution / webhooks     | Task queue, resume from checkpoint               | v0.22.0 |
| Code execution sandbox (Docker/E2B) | Sandboxed code execution for untrusted input   | v0.22.0 |
| Prompt registry / versioning     | Version, A/B test, rollback prompts              | v0.22.0 |
| Time-travel debugging / state replay | Rewind, edit, replay from any checkpoint      | v1.x    |
| Voice / real-time audio agents   | WebRTC, STT/TTS, sub-500ms latency              | v1.x    |
| Rate Limiting & Quotas           | Per-tool and per-user quotas                     | Future  |
| CRM & Business Tools             | HubSpot, Salesforce integrations                 | Future  |
| Niche Loaders                    | Slack, Confluence, Jira, Discord, Email, Docx    | Future  |
| Niche Vector Stores              | Weaviate, Redis Vector, Milvus, OpenSearch, Lance | Future  |
