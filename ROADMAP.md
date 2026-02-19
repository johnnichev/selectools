# Selectools Development Roadmap

> **Status Legend**
>
> - ✅ **Implemented** - Merged and available in latest release
> - 🔵 **In Progress** - Actively being worked on
> - 🟡 **Planned** - Scheduled for implementation
> - ⏸️ **Deferred** - Postponed to later release
> - ❌ **Cancelled** - No longer planned

---

## v0.9.0: Core Capabilities & Reliability (Available Now)

Recent major improvements focusing on agent control and reliability (Agent v0.9) and high-impact RAG features (Embeddings v0.9).

| Feature                        | Status    | Notes                                                                  |
| ------------------------------ | --------- | ---------------------------------------------------------------------- |
| **Custom System Prompt**       | ✅ v0.9.0 | Inject domain instructions via `AgentConfig(system_prompt=...)`        |
| **Structured Decision Result** | ✅ v0.9.0 | `run()` returns `AgentResult` with tool calls, args, and usage stats   |
| **Reusable Agent Instances**   | ✅ v0.9.0 | `Agent.reset()` clears history/memory for clean reuse between requests |
| **Embeddings & RAG**           | ✅ v0.8.0 | Vector stores, document loaders, semantic search, RAG tools            |

---

## v0.10.0: Critical Architecture (Complete)

Focus: Fixing architectural limitations and enabling production scaling.

| Feature                         | Status     | Impact | Description                                                              |
| ------------------------------- | ---------- | ------ | ------------------------------------------------------------------------ |
| **Native Function Calling**     | ✅ v0.10.0 | High   | Use OpenAI/Anthropic/Gemini native tool APIs instead of regex parsing    |
| **Context Propagation (Async)** | ✅ v0.10.0 | High   | `contextvars.copy_context()` for tracing/auth in async tools             |
| **Select-Only / Routing Mode**  | ✅ v0.10.0 | High   | Run agent for classification/routing without executing the selected tool |

---

## v0.11.0: Streaming & Performance (Complete)

Focus: E2E streaming, parallel execution, and type safety.

| Feature                     | Status     | Notes                                                                                                  |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| **E2E Streaming Response**  | ✅ v0.11.0 | Native tool streaming via `Agent.astream` with `Union[str, ToolCall]` provider protocol                |
| **Parallel Tool Execution** | ✅ v0.11.0 | `asyncio.gather` for async, `ThreadPoolExecutor` for sync; `AgentConfig(parallel_tool_execution=True)` |
| **Full Type Safety**        | ✅ v0.11.0 | 0 mypy errors across all source and test files; `disallow_untyped_defs` enforced                       |

---

## v0.12.0: Caching & Data (Complete)

Focus: Response caching and advanced RAG capabilities.

| Feature                  | Status     | Notes                                                                        |
| ------------------------ | ---------- | ---------------------------------------------------------------------------- |
| **Response Caching**     | ✅ v0.12.0 | `InMemoryCache` (LRU+TTL) and `RedisCache`; `AgentConfig(cache=...)`         |
| **Hybrid Search**        | ✅ v0.12.x | `BM25` + `HybridSearcher` with RRF/weighted fusion; `HybridSearchTool`       |
| **Reranking Models**     | ✅ v0.12.x | `CohereReranker` + `JinaReranker`; `HybridSearcher(reranker=...)`            |
| **Advanced Chunking**    | ✅ v0.12.x | `SemanticChunker` (embedding similarity) + `ContextualChunker` (LLM context) |
| **Dynamic Tool Loading** | ✅ v0.12.x | `ToolLoader` + `Agent.add_tool/remove_tool/replace_tool`; hot-reload         |

---

## v0.13.0: Structured Output & Safety (Next)

Focus: Typed classification results, tool execution safety, and production resilience.
This is the highest-impact release for routing/classification use cases (traffic cops, intent classifiers, request routers).

### Structured Output Parsers

**Problem**: `routing_only=True` returns `result.tool_name` (str) and `result.tool_args` (dict). Consumers must manually validate and cast these into domain types. There's no guarantee the LLM returned a valid classification.

**What it does**: Let callers pass a Pydantic model or JSON Schema as `response_format`. The agent enforces the schema using provider-native structured output (OpenAI JSON mode, Anthropic tool-use schemas) with automatic fallback to regex extraction + validation for providers that don't support it natively.

**API**:

```python
from pydantic import BaseModel
from typing import Literal

class Classification(BaseModel):
    intent: Literal["billing", "support", "sales", "cancel"]
    confidence: float
    priority: Literal["low", "medium", "high"]

result = agent.ask("I want to cancel my account", response_format=Classification)
# result.parsed -> Classification(intent="cancel", confidence=0.95, priority="high")
# result.content -> still available as raw string
```

**Scope**:

- New `response_format` param on `agent.ask()` / `agent.run()` / `agent.arun()`
- Pydantic v2 `BaseModel` support (automatic JSON Schema generation)
- Raw `dict` JSON Schema support for non-Pydantic users
- OpenAI `response_format={"type": "json_schema", ...}` native path
- Anthropic tool-use schema enforcement native path
- Fallback: extract JSON from response text, validate against schema, retry on failure
- `result.parsed` property returns the typed object; `result.content` stays as raw string
- Validation errors trigger a retry with the error message fed back to the LLM

**Touches**: `agent/core.py`, `types.py` (new `parsed` field on `AgentResult`), provider `complete()` methods, new `structured.py` module.

### Provider Fallback Chain

**Problem**: A routing agent is on the critical path — if the provider goes down, nothing gets classified. There's no automatic failover.

**What it does**: Wrap multiple providers in priority order. If the first fails (timeout, 5xx, rate limit), the next one is tried automatically. Circuit breaker logic prevents hammering a dead provider.

**API**:

```python
from selectools import FallbackProvider

provider = FallbackProvider([
    OpenAIProvider(default_model="gpt-4o-mini"),
    AnthropicProvider(default_model="claude-haiku"),
    LocalProvider(),  # last resort
])

agent = Agent(tools=[...], provider=provider, config=config)
# If OpenAI is down → tries Anthropic → tries Local
# result.provider_used tells you which one handled it
```

**Scope**:

- New `FallbackProvider` class implementing the `Provider` protocol
- Ordered list of providers, tried sequentially on failure
- Configurable failure conditions: timeout, HTTP 5xx, rate limit (429), connection error
- Built-in circuit breaker: after N consecutive failures, skip provider for M seconds
- `on_fallback` hook fires when a provider is skipped
- `result.provider_used` (str) on `AgentResult` for observability
- Thread-safe for concurrent requests

**Touches**: New `providers/fallback.py` module, `AgentResult` (new field), `AgentConfig` (optional).

### Batch Processing

**Problem**: Classifying one request at a time is wasteful for queue-based workloads (ticket triage, email classification, bulk routing). There's no batch API.

**What it does**: Process multiple requests concurrently with configurable parallelism, returning results in order.

**API**:

```python
tickets = [
    "I want to cancel my subscription",
    "How do I upgrade my plan?",
    "My payment failed",
    "Can I talk to someone in sales?",
]

results = await agent.abatch(tickets, max_concurrency=10)
# results[0].tool_name == "handle_cancellation"
# results[1].tool_name == "handle_upgrade"
# ...

# Sync variant
results = agent.batch(tickets, max_concurrency=5)
```

**Scope**:

- `agent.batch(prompts, max_concurrency=5)` — sync, uses ThreadPoolExecutor
- `agent.abatch(prompts, max_concurrency=10)` — async, uses asyncio.Semaphore + gather
- Returns `list[AgentResult]` in same order as input
- Per-request error isolation (one failure doesn't cancel the batch)
- Respects cache (cached results returned instantly, only uncached hit the provider)
- Progress callback: `on_batch_progress(completed, total)`
- Aggregated usage stats across the batch

**Touches**: `agent/core.py` (new methods), uses existing `run()` / `arun()` internally.

### Tool-Pair-Aware Trimming

**Problem**: `ConversationMemory._enforce_limits()` uses naive sliding-window trimming. If the cut happens between an assistant `tool_use` message and its `tool_result`, the LLM receives a malformed conversation that violates provider API contracts.

**What it does**: After trimming, scan forward to find the first safe boundary (a user text message that isn't a tool_result). Never orphan a tool_use without its result.

**Scope**:

- Fix `_enforce_limits()` in `memory.py`
- Ensure trimmed conversation always starts with a valid user text message
- Preserve at least one complete exchange
- Small, self-contained fix — no new modules

**Touches**: `memory.py` only.

### Tool Policy Engine

**Problem**: The agent trusts all LLM tool-call decisions unconditionally. In production, some tools (delete_record, send_email) should require approval or be blocked entirely.

**What it does**: Declarative allow/review/deny rules evaluated before every tool execution. Uses glob patterns for tool names and optional argument-level conditions.

**API**:

```python
from selectools import ToolPolicy

policy = ToolPolicy(
    allow=["search_*", "read_*", "get_*"],
    review=["send_*", "create_*", "update_*"],
    deny=["delete_*", "drop_*"],
)
agent = Agent(tools=[...], provider=provider, config=AgentConfig(tool_policy=policy))
```

**Evaluation order**: deny → review → allow → unknown defaults to review.

**Scope**:

- New `policy.py` module with `ToolPolicy` class
- Glob-based matching on tool names
- Argument-level conditions (`deny_when: tool=send_email, arg=to, pattern="*@external.com"`)
- YAML loading: `ToolPolicy.from_yaml("policies/default.yaml")`
- Integration point in agent loop (between tool selection and execution)

**Touches**: `agent/core.py`, new `policy.py`, `AgentConfig`.

### Human-in-the-Loop Approval

**Problem**: Even with a policy engine, there's no way to pause and ask "Are you sure?" before executing a flagged tool.

**What it does**: A confirmation callback the agent loop invokes for tools flagged as `review`. Supports both sync (CLI) and async (web API) patterns.

**API**:

```python
async def confirm(tool_name: str, tool_args: dict, reason: str) -> bool:
    # Show in UI, send Slack message, wait for response...
    return await get_user_approval(tool_name, tool_args)

config = AgentConfig(
    tool_policy=policy,
    confirm_action=confirm,
    approval_timeout=60,  # deny on timeout
)
```

**Agent loop behaviour**:

- `allow` → execute immediately
- `review` + `confirm_action` → call callback; execute if approved, deny if rejected
- `review` + no callback → deny with error message to LLM
- `deny` → return error to LLM, never execute

**Scope**:

- Sync and async callback support
- Configurable timeout with deny-on-timeout default
- Integrates with Tool Policy Engine (depends on it)

**Touches**: `agent/core.py`, `AgentConfig`, new `approvals.py`.

| Feature                        | Priority  | Impact | Effort |
| ------------------------------ | --------- | ------ | ------ |
| **Structured Output Parsers**  | 🟡 High   | High   | Medium |
| **Provider Fallback Chain**    | 🟡 High   | High   | Medium |
| **Batch Processing**           | 🟡 High   | High   | Small  |
| **Tool-Pair-Aware Trimming**   | 🟡 High   | High   | Small  |
| **Tool Policy Engine**         | 🟡 High   | High   | Medium |
| **Human-in-the-Loop Approval** | 🟡 Medium | High   | Medium |

---

## v0.14.0: Multi-Agent Orchestration

Focus: Composable agent graphs, delegation, and the classify-then-delegate pattern.
This is the natural evolution for traffic cop architectures — classifier selects a route, specialist agent handles it.

### Multi-Agent Graphs

**Problem**: Building pipelines where a classifier routes to specialist agents requires manual wiring. There's no built-in way to compose agents into workflows.

**What it does**: Define directed graphs where nodes are agents and edges are conditional transitions. Supports sequential chains, parallel fan-out, and conditional routing based on the previous agent's result.

**API**:

```python
from selectools import AgentGraph

graph = AgentGraph()

graph.add_node("classifier", classifier_agent)
graph.add_node("billing", billing_agent)
graph.add_node("support", support_agent)
graph.add_node("fallback", fallback_agent)

graph.add_edge("classifier", "billing", when=lambda r: r.tool_name == "route_billing")
graph.add_edge("classifier", "support", when=lambda r: r.tool_name == "route_support")
graph.add_edge("classifier", "fallback")  # default edge (no condition)

result = graph.run("I need to cancel my subscription", entry="classifier")
# result.path == ["classifier", "billing"]
# result.content == final response from billing_agent
```

**Scope**:

- `AgentGraph` builder with `add_node()` and `add_edge()`
- Conditional edges via `when=` lambda receiving `AgentResult`
- Default edge (fallback) when no condition matches
- Sequential execution: output of one node feeds as input to the next
- `result.path` tracks which nodes were visited
- Cycle detection at build time
- Max depth limit to prevent infinite loops

**Touches**: New `graph.py` module, new `GraphResult` type.

### Agent Handoffs

**Problem**: When one agent delegates to another, context (user intent, extracted entities, conversation history) is lost. The receiving agent starts cold.

**What it does**: First-class mechanism for agents to pass typed context to the next agent in a graph. The handoff payload is injected into the receiving agent's system prompt or initial messages.

**API**:

```python
from selectools import Handoff

graph.add_edge(
    "classifier", "billing",
    when=lambda r: r.tool_name == "route_billing",
    handoff=Handoff(
        pass_context=True,         # forward conversation history
        pass_tool_results=True,    # include tool call results
        inject_as="system",        # inject into system prompt vs user message
        summary=True,              # LLM-summarize context before passing (saves tokens)
    ),
)
```

**Scope**:

- `Handoff` config object controlling what context transfers between nodes
- Options: raw history, tool results only, LLM-generated summary, custom payload
- Inject mode: as system prompt extension or as first user message
- Token budget: truncate/summarize if handoff context exceeds a limit

**Touches**: `graph.py`, `Handoff` dataclass in `types.py`.

### Shared State & Blackboard

**Problem**: Agents in a graph can't share data except through handoffs. Sometimes you need a scratchpad — e.g., the classifier writes `customer_tier: "enterprise"` and the billing agent reads it.

**What it does**: A thread-safe key-value store accessible by all agents in a graph execution. Supports read/write scoping per node.

**API**:

```python
graph = AgentGraph(shared_state={"customer_id": None, "tier": None})

# Inside a tool:
@tool(description="Look up customer tier")
def lookup_tier(customer_id: str, _state: dict) -> str:
    tier = db.get_tier(customer_id)
    _state["tier"] = tier  # visible to downstream agents
    return tier

# _state is injected automatically when the tool is inside a graph
```

**Scope**:

- `dict`-based shared state initialized at graph creation
- Thread-safe via `threading.Lock` (sync) or `asyncio.Lock` (async)
- Optional per-node read/write permissions
- State accessible in tools via injected `_state` kwarg
- State snapshot included in `GraphResult`

**Touches**: `graph.py`, tool execution in `agent/core.py`.

### Supervisor Agent

**Problem**: For complex tasks, a single classifier isn't enough. You need a meta-agent that decomposes the task, delegates subtasks to specialists, and synthesizes the final answer.

**What it does**: A pre-built agent pattern that receives a task, breaks it into subtasks, dispatches each to the appropriate specialist, collects results, and produces a unified response.

**API**:

```python
from selectools import SupervisorAgent

supervisor = SupervisorAgent(
    specialists={
        "research": research_agent,
        "analysis": analysis_agent,
        "writing": writing_agent,
    },
    strategy="sequential",  # or "parallel", "adaptive"
    provider=provider,
)

result = supervisor.run("Write a market analysis report for Q1 2026")
# Supervisor decomposes → research gathers data → analysis processes → writing drafts
```

**Scope**:

- Built on top of `AgentGraph` (sugar, not a separate system)
- Three strategies: sequential (chain), parallel (fan-out + merge), adaptive (LLM decides)
- Automatic task decomposition prompt
- Result aggregation with configurable merge strategy
- Token/cost budget across all specialists

**Touches**: New `supervisor.py` module built on `graph.py`.

| Feature                       | Priority  | Impact | Effort |
| ----------------------------- | --------- | ------ | ------ |
| **Multi-Agent Graphs**        | 🟡 High   | High   | Large  |
| **Agent Handoffs**            | 🟡 High   | High   | Medium |
| **Shared State & Blackboard** | 🟡 Medium | Medium | Small  |
| **Supervisor Agent**          | 🟡 Medium | High   | Medium |

---

## v0.15.0: MCP & Ecosystem

Focus: Interoperability with the broader AI tooling ecosystem and web framework integration.

### MCP Client

**Problem**: Tools exist in external MCP-compliant servers (Cursor, Claude Desktop, custom servers) but there's no way for a selectools agent to discover and call them.

**What it does**: Connect to any MCP server, auto-discover its tools, and register them with the agent. Remote tools are called transparently — the agent doesn't know the difference between local and remote tools.

**API**:

```python
from selectools import MCPToolProvider

# Auto-discover tools from an MCP server
mcp_tools = MCPToolProvider.from_server("http://localhost:8080")
agent = Agent(tools=[local_tool, *mcp_tools], provider=provider)

# Or connect to stdio-based MCP servers
mcp_tools = MCPToolProvider.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem"])
```

**Scope**:

- MCP client implementing the Model Context Protocol spec
- HTTP/SSE and stdio transport support
- Auto-discovery: `tools/list` → register as selectools `Tool` objects
- Transparent execution: agent calls MCP tools like local tools
- Schema translation: MCP JSON Schema → selectools `ToolParameter`
- Error handling: MCP errors surfaced as tool execution errors

**Touches**: New `mcp/` subpackage (`client.py`, `transport.py`, `tools.py`).

### MCP Server

**Problem**: Other MCP clients (Cursor, Claude Desktop, custom apps) can't call selectools tools.

**What it does**: Expose any selectools `@tool` functions as an MCP-compliant server. Any MCP client can discover and call them.

**API**:

```python
from selectools.mcp import serve_tools

serve_tools(
    tools=[search_docs, create_ticket, query_database],
    transport="stdio",  # or "http"
    port=8080,
)
```

**Scope**:

- Expose `@tool` functions via MCP protocol
- Automatic schema generation from tool definitions
- HTTP/SSE and stdio transport
- Optional: expose an entire agent as a single MCP tool

**Touches**: New `mcp/server.py` module.

### Framework Integrations

**Problem**: Embedding a selectools agent into a FastAPI or Flask app requires boilerplate for request/response handling, streaming, and error mapping.

**What it does**: Thin adapters that wire a selectools agent into web frameworks with proper streaming, error handling, and OpenAPI schema generation.

**API**:

```python
from selectools.integrations.fastapi import AgentRouter

router = AgentRouter(agent=agent, prefix="/api/agent")
app.include_router(router)
# POST /api/agent/ask → { "content": "...", "tool_calls": [...] }
# POST /api/agent/stream → SSE stream
```

**Scope**:

- FastAPI adapter: `AgentRouter` with `/ask`, `/stream`, `/batch` endpoints
- Flask adapter: `AgentBlueprint` with equivalent routes
- Automatic OpenAPI schema from agent config and tool definitions
- SSE streaming for `astream()` responses
- Error mapping: selectools exceptions → HTTP status codes

**Touches**: New `integrations/` subpackage.

| Feature                    | Priority  | Impact | Effort |
| -------------------------- | --------- | ------ | ------ |
| **MCP Client**             | 🟡 High   | High   | Large  |
| **MCP Server**             | 🟡 Medium | Medium | Medium |
| **Framework Integrations** | 🟡 Medium | Medium | Medium |

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

| Feature                              | Priority  | Impact | Effort |
| ------------------------------------ | --------- | ------ | ------ |
| **Persistent Conversation Sessions** | 🟡 High   | High   | Medium |
| **Summarize-on-Trim**                | 🟡 Medium | Medium | Small  |
| **Entity Memory**                    | 🟡 Medium | High   | Medium |
| **Knowledge Graph Memory**           | 🟡 Low    | High   | Large  |

---

## v1.0.0: Enterprise Reliability

Focus: Guardrails, security hardening, observability, and audit compliance.

### Guardrails Engine

**Problem**: No systematic way to validate what goes into or comes out of the LLM. Users need content moderation, PII detection, topic restriction, and format enforcement.

**What it does**: A pluggable pipeline of validators that run before (input) and after (output) every LLM call. Each guardrail returns pass/fail/rewrite.

**API**:

```python
from selectools.guardrails import GuardrailsPipeline, TopicGuardrail, PIIGuardrail

guardrails = GuardrailsPipeline(
    input=[
        TopicGuardrail(deny=["politics", "religion"]),
        PIIGuardrail(action="redact"),  # redact SSNs, emails, etc.
    ],
    output=[
        FormatGuardrail(require_json=True),
        ToxicityGuardrail(threshold=0.8),
    ],
)

agent = Agent(tools=[...], provider=provider, config=AgentConfig(guardrails=guardrails))
```

**Scope**:

- `Guardrail` protocol: `check(content) -> GuardrailResult(passed, rewritten_content, reason)`
- `GuardrailsPipeline`: ordered list of input and output guardrails
- Built-in guardrails: `TopicGuardrail`, `PIIGuardrail`, `ToxicityGuardrail`, `FormatGuardrail`, `LengthGuardrail`
- Custom guardrails: subclass `Guardrail` and implement `check()`
- Actions on failure: block (raise), rewrite (modify content), warn (log + continue)
- Integration with agent loop: input guardrails before provider call, output guardrails after

**Touches**: New `guardrails/` subpackage, `agent/core.py`, `AgentConfig`.

### Audit Logging

**Problem**: No built-in audit trail. Compliance, debugging, and cost analysis require knowing what tools were called, when, and whether they succeeded.

**What it does**: JSONL append-only audit log with privacy controls and optional daily rotation.

**Scope**:

- `AuditLogger` class with configurable privacy (hash inputs, log arg keys only)
- JSONL format with timestamps, tool name, duration, success/failure
- Daily file rotation
- Queryable via existing `AgentAnalytics` or standalone

**Touches**: New `audit.py`, hooks into `agent/core.py`.

### Tool Output Screening

**Problem**: Tools returning untrusted content (web scraping, email, file parsing) can contain prompt injection payloads.

**What it does**: Optional screening of tool outputs before feeding them back to the LLM. Pattern-based, classifier-based, or LLM-based detection.

**Scope**:

- `@tool(screen_output=True)` flag
- Screening strategies: regex patterns, lightweight classifier, LLM-based
- Blocked outputs replaced with safe message
- Integrates with guardrails engine

**Touches**: `agent/core.py`, `Tool` class, new `security.py`.

### Coherence Checking

**Problem**: Prompt injection in tool outputs can cause the agent to call unrelated tools. User asks "summarize my emails" but injected content causes `send_email` instead.

**What it does**: Optional LLM-based check that verifies each tool call matches the user's original intent.

**Scope**:

- `AgentConfig(coherence_check=True)`
- Lightweight check using fast model (e.g., Haiku)
- Compare proposed tool call against original user intent
- Block incoherent calls with explanation to LLM

**Touches**: `agent/core.py`, `AgentConfig`.

| Feature                   | Priority  | Impact | Effort |
| ------------------------- | --------- | ------ | ------ |
| **Guardrails Engine**     | 🟡 High   | High   | Large  |
| **Audit Logging**         | 🟡 Medium | Medium | Small  |
| **Tool Output Screening** | 🟡 Medium | Medium | Medium |
| **Coherence Checking**    | 🟡 Medium | Medium | Medium |

---

## Implementation Order

```
v0.13.0  Structured Output + Safety Foundation
         Tool-pair trimming → Structured output → Fallback providers → Batch
         → Tool policy engine → Human-in-the-loop

v0.14.0  Multi-Agent Orchestration
         Agent graphs → Handoffs → Shared state → Supervisor

v0.15.0  MCP & Ecosystem
         MCP client → MCP server → FastAPI/Flask integrations

v0.16.0  Memory & Persistence
         Sessions → Summarize-on-trim → Entity memory → Knowledge graph

v1.0.0   Enterprise Reliability
         Guardrails → Audit logging → Output screening → Coherence checking
```

---

## Backlog (Unscheduled)

| Feature                    | Notes                                            |
| -------------------------- | ------------------------------------------------ |
| Tool Composition           | `@compose` decorator for chaining tools          |
| Tool Marketplace/Registry  | Community tool sharing                           |
| Universal Vision Support   | Unified vision API across providers              |
| AWS Bedrock Provider       | VPC-native model access (Claude, Llama, Mistral) |
| Observability & Debugging  | OpenTelemetry traces, execution replay           |
| Rate Limiting & Quotas     | Per-tool and per-user quotas                     |
| Interactive Debug Mode     | Step-through agent execution                     |
| Visual Agent Builder       | Web UI for agent design                          |
| Enhanced Testing Framework | Snapshot testing, load tests                     |
| Documentation Generation   | Auto-generate docs from tool definitions         |
| Prompt Optimization        | Automatic prompt compression                     |
| YAML Tool Configuration    | Declarative tool definitions without code        |
| CRM & Business Tools       | HubSpot, Salesforce integrations                 |
| Data Source Connectors     | SQL, vector DBs, cloud storage                   |

---

## Release History

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
