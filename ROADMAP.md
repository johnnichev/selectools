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

v0.21.0 🟡 Connector Expansion + Multimodal + Observability
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

## v0.21.0: Connector Expansion + Multimodal + Observability 🟡

Close integration gaps, add multimodal support (images/audio), and ship enterprise-grade observability (OTel + Langfuse). Full spec: `.private/plans/07-v0.21.0-connector-expansion.md`

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

## Backlog (Unscheduled)

| Feature                                                                   | Notes                                           | Target  |
| ------------------------------------------------------------------------- | ----------------------------------------------- | ------- |
| AWS Bedrock provider                                                      | boto3 wrapper, enterprise gateway               | v0.22.0 |
| Google A2A protocol                                                       | Cross-framework agent interop (Linux Foundation) | v0.22.0 |
| Durable execution / webhooks                                              | Task queue, resume from checkpoint               | v0.22.0 |
| Code execution sandbox (Docker/E2B)                                       | Sandboxed code execution for untrusted input     | v0.22.0 |
| Prompt registry / versioning                                              | Version, A/B test, rollback prompts              | v0.22.0 |
| Time-travel debugging / state replay                                      | Rewind, edit, replay from any checkpoint         | v1.x    |
| Voice / real-time audio agents                                            | WebRTC, STT/TTS, sub-500ms latency              | v1.x    |
| Rate Limiting & Quotas                                                    | Per-tool and per-user quotas                     | Future  |
| CRM & Business Tools                                                     | HubSpot, Salesforce integrations                 | Future  |
| Niche Loaders (Slack, Confluence, Jira, Discord, Email, Docx, Excel, XML) | Community-driven                                 | Future  |
| Niche Vector Stores (Weaviate, Redis Vector, Milvus, OpenSearch, Lance)   | As demand dictates                               | Future  |
| Niche Toolbox (Email, Calendar, Browser, Financial)                       | As demand dictates                               | Future  |
