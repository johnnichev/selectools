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

v0.19.x 🟡 Enterprise Hardening + Advanced Patterns + Polish & Community
Security audit → Stability markers → Deprecation policy → Compatibility matrix
→ PlanAndExecute → ReflectiveAgent → Debate → TeamLead → 50+ evaluators
→ Published benchmarks → Enhanced trace viewer → Tool marketplace foundations

v0.20.0 🟡 Connector Expansion + Visual Agent Builder
AWS Bedrock → Azure OpenAI → FAISS → Qdrant
→ CSV/JSON/HTML/URL loaders → Visual Agent Builder (web UI)

v1.0.0 🟡 Stable Release
API freeze → Stability markers on all modules → Deprecation policy
→ Security audit published → Compatibility matrix → 0.x→1.0 migration guide
→ PyPI classifier: Production/Stable

---

## v0.19.x: Enterprise Hardening + Polish & Community + Advanced Agent Patterns 🟡

Focus: Production readiness, community growth, and developer trust signals. This is the "make it trustworthy" release series before adding new features in v0.20.

### Enterprise Hardening


| Feature                                                            | Status | Impact | Effort |
| ------------------------------------------------------------------ | ------ | ------ | ------ |
| **Security audit** (Snyk + bandit + OWASP review)                  | 🟡     | High   | Medium |
| **Stability markers** (`__stability__ = "stable"/"beta"/"alpha"`)  | 🟡     | Medium | Small  |
| **Deprecation policy** (2-version warning before removal)          | 🟡     | Medium | Small  |
| **Compatibility matrix** (Python 3.9-3.13 + provider SDK versions) | 🟡     | Medium | Small  |
| **SBOM generation** for compliance teams                           | 🟡     | Low    | Small  |
| **Enhanced trace viewer** (interactive HTML)                       | 🟡     | High   | Medium |


### Advanced Agent Patterns

Higher-level agent architectures built on the v0.18.0 orchestration primitives. Each pattern is a pre-built `AgentGraph` topology.

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


| Feature             | Status    | Impact | Effort |
| ------------------- | --------- | ------ | ------ |
| **PlanAndExecute**  | 🟡 High   | High   | Medium |
| **ReflectiveAgent** | 🟡 High   | High   | Medium |
| **Debate**          | 🟡 Medium | Medium | Medium |
| **TeamLead**        | 🟡 Medium | Medium | Medium |
| **50+ evaluators**  | 🟡 Medium | High   | Large  |


---

## v0.20.0: Connector Expansion + Visual Agent Builder 🟡

Close the integration gap with LangChain by adding high-demand providers, vector stores, document loaders, and a visual agent design tool.

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
| `**code_tools.py`**   | `execute_python`, `execute_shell`                               | stdlib `subprocess`            | Medium       | #1 most-used tool in agent frameworks |
| `**search_tools.py**` | `google_search`, `duckduckgo_search`                            | `duckduckgo_search` (optional) | Small-Medium | #2 most-used tool category            |
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
| **Visual Agent Builder**   | 🟡 High   | High   | Large  |

### Visual Agent Builder

A web-based UI for designing, testing, and exporting agent configurations. Zero-install — served by `selectools serve --builder`.

```
┌─────────────────────────────────────────────────────┐
│  Visual Agent Builder                    [Export]    │
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

| Feature | Status | Impact | Effort |
| --- | --- | --- | --- |
| **Graph canvas (drag-drop nodes + edges)** | 🟡 | High | Large |
| **Node configuration panel** | 🟡 | High | Medium |
| **YAML export/import** | 🟡 | High | Small |
| **Python code export** | 🟡 | Medium | Small |
| **Live test execution** | 🟡 | High | Medium |
| **Self-contained HTML (no build step)** | 🟡 | High | Medium |

---

## Backlog (Unscheduled)


| Feature                                                                   | Notes                               | Target |
| ------------------------------------------------------------------------- | ----------------------------------- | ------ |
| Universal Vision Support                                                  | Unified vision API across providers | Future |
| Rate Limiting & Quotas                                                    | Per-tool and per-user quotas        | Future |
| CRM & Business Tools                                                      | HubSpot, Salesforce integrations    | Future |
| Niche Loaders (Slack, Confluence, Jira, Discord, Email, Docx, Excel, XML) | Community-driven                    | Future |
| Niche Vector Stores (Milvus, OpenSearch, Lance)                           | As demand dictates                  | Future |
| Niche Toolbox (Email, Calendar, Browser, Financial)                       | As demand dictates                  | Future |
