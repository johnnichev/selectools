# Changelog

All notable changes to selectools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.1] - 2026-03-27

### Fixed

- **OpenAI/Ollama async streaming tool name concatenation** — `astream()` used `+=` instead of `=` for tool function names, causing names to double (e.g., "get_weatherget_weather"). Sync `stream()` was correct.
- **OpenAI/Ollama async streaming missing "stop" finish reason** — `astream()` only emitted tool calls on `finish_reason="tool_calls"`, missing the `"stop"` case. Tool calls silently lost when model stopped after tool use.
- **Eval report p95/p99 off-by-one** — Percentile index calculation was one position too high.

---

## [0.18.0] - 2026-03-27

### Added

#### Multi-Agent Orchestration — AgentGraph

The single biggest feature since launch. Build directed graphs of agent nodes with plain Python routing — no compile step, no Pregel, no DSL.

```python
from selectools import AgentGraph

graph = AgentGraph()
graph.add_node("planner", planner_agent)
graph.add_node("writer", writer_agent)
graph.add_edge("planner", "writer")
graph.add_edge("writer", AgentGraph.END)
graph.set_entry("planner")
result = graph.run("Write a blog post about AI safety")
```

- **AgentGraph** — Directed graph of Agent nodes (or callables) with `run()`, `arun()`, `astream()`
- **GraphState** — Shared context with `messages`, `data`, `history`, `metadata`, `errors`
- **ContextMode** — `LAST_MESSAGE` (default), `LAST_N`, `FULL`, `SUMMARY`, `CUSTOM` — prevents context explosion
- **Routing** — `add_edge()` for static, `add_conditional_edge()` for dynamic, `Scatter` for fan-out
- **Parallel Execution** — `add_parallel_nodes()` with `MergePolicy` (LAST_WINS, FIRST_WINS, APPEND)
- **Human-in-the-Loop** — Generator nodes with `yield InterruptRequest()`. Resumes at the exact yield point (LangGraph restarts the whole node)
- **Checkpointing** — `CheckpointStore` protocol with 3 backends: `InMemoryCheckpointStore`, `FileCheckpointStore`, `SQLiteCheckpointStore`
- **Subgraph Composition** — `add_subgraph()` with `input_map`/`output_map` for state key translation
- **Error Handling** — `ErrorPolicy` (ABORT, SKIP, RETRY) at graph and per-node level
- **Loop & Stall Detection** — State hash tracking; hard loops raise, stalls fire observer events
- **Budget & Cancellation** — `max_total_tokens`, `max_cost_usd`, `cancellation_token` propagated to all nodes
- **Visualization** — `to_mermaid()` and `visualize("ascii")` for graph inspection

#### SupervisorAgent — High-Level Coordination

```python
from selectools import SupervisorAgent

supervisor = SupervisorAgent(
    agents={"researcher": researcher, "writer": writer},
    provider=provider,
    strategy="plan_and_execute",
)
result = supervisor.run("Write a report on AI safety")
```

- **4 Strategies**: `plan_and_execute`, `round_robin`, `dynamic`, `magentic`
- **ModelSplit** — `planner_model` (expensive) + `executor_model` (cheap) for 70-90% cost reduction
- **Magentic-One** — Task Ledger + Progress Ledger + auto-replan after `max_stalls` consecutive unproductive steps
- **Delegation Constraints** — Explicit allow-lists preventing ping-pong delegation loops

#### Composable Pipelines

Chain agents, tools, and transforms with the `|` operator — the selectools answer to LangChain's LCEL, but plain Python.

```python
from selectools import Pipeline, step, parallel, branch

@step
def summarize(text: str) -> str:
    return agent.run(f"Summarize: {text}").content

@step
def translate(text: str, lang: str = "es") -> str:
    return agent.run(f"Translate to {lang}: {text}").content

pipeline = summarize | translate
result = pipeline.run("Long article text here...")
```

- **Pipeline** — Sequential composition with `Pipeline(steps=[...])` or `step_a | step_b`
- **@step decorator** — Wrap any sync/async callable into a composable `Step`
- **parallel()** — Fan-out to multiple steps, merge results
- **branch()** — Conditional routing based on input data
- **StepResult** — Each step produces a typed result with name, duration, and output
- New exports: `Pipeline`, `Step`, `StepResult`, `step`, `parallel`, `branch`

#### New Trace & Observer Infrastructure

- 10 new `StepType` values: `graph_node_start`, `graph_node_end`, `graph_routing`, `graph_checkpoint`, `graph_interrupt`, `graph_resume`, `graph_parallel_start`, `graph_parallel_end`, `graph_stall`, `graph_loop_detected`
- 13 new observer events: `on_graph_start`, `on_graph_end`, `on_graph_error`, `on_node_start`, `on_node_end`, `on_graph_routing`, `on_graph_interrupt`, `on_graph_resume`, `on_parallel_start`, `on_parallel_end`, `on_stall_detected`, `on_loop_detected`, `on_supervisor_replan`
- Total: 27 StepTypes, 45 sync observer events, 42 async observer events

#### New Examples

- `examples/55_agent_graph_linear.py` — 3-node sequential pipeline
- `examples/56_agent_graph_parallel.py` — Parallel fan-out with merge
- `examples/57_agent_graph_conditional.py` — Conditional routing
- `examples/58_agent_graph_hitl.py` — Human-in-the-loop with generator nodes
- `examples/59_agent_graph_checkpointing.py` — Durable checkpoint/resume
- `examples/60_supervisor_agent.py` — SupervisorAgent with 4 strategies
- `examples/61_agent_graph_subgraph.py` — Nested subgraph composition

### Stats

- Tests: 2275 → 2435 (+160)
- Examples: 54 → 61 (+7)
- StepTypes: 17 → 27 (+10)
- Sync observer events: 32 → 45 (+13)
- Async observer events: 29 → 42 (+13)
- Bug fixes: 35
- New source files: 7 (orchestration package + pipeline.py)

---

## [0.17.7] - 2026-03-25

### Added

#### SemanticCache
- New `SemanticCache` in `src/selectools/cache_semantic.py` — drop-in replacement for `InMemoryCache`
- Embeds cache keys with any `EmbeddingProvider` and serves hits based on cosine similarity
- Configurable `similarity_threshold` (default 0.92), `max_size` (LRU), `default_ttl`
- Thread-safe (internal `threading.Lock`); pure-Python cosine similarity (no NumPy)
- `stats` property returns `CacheStats` with hit/miss/eviction counters and `hit_rate`

```python
from selectools.cache_semantic import SemanticCache
from selectools.embeddings.openai import OpenAIEmbeddingProvider

cache = SemanticCache(
    embedding_provider=OpenAIEmbeddingProvider(),
    similarity_threshold=0.92,
    max_size=500,
)
config = AgentConfig(cache=cache)
# "What's the weather in NYC?" hits cache for "Weather in New York City?"
```

#### Prompt Compression
- New `compress_context`, `compress_threshold`, `compress_keep_recent` fields on `AgentConfig`
- Fires before each LLM call when estimated fill-rate ≥ threshold; summarises old messages into a `[Compressed context]` system message
- Only modifies `self._history` (per-call view) — `self.memory` is never touched
- New `StepType.PROMPT_COMPRESSED` added to `AgentTrace`
- New `on_prompt_compressed(run_id, before_tokens, after_tokens, messages_compressed)` observer event on both `AgentObserver` and `AsyncAgentObserver`

```python
config = AgentConfig(
    compress_context=True,
    compress_threshold=0.75,  # trigger at 75 % context fill
    compress_keep_recent=4,   # keep last 4 turns verbatim
)
```

#### Conversation Branching
- New `ConversationMemory.branch()` — returns an independent snapshot; changes to branch don't affect original
- New `SessionStore.branch(source_id, new_id)` — forks a persisted session; supported by all three backends (`JsonFileSessionStore`, `SQLiteSessionStore`, `RedisSessionStore`)
- Raises `ValueError` if `source_id` not found

```python
checkpoint = agent.memory.branch()     # snapshot in-memory
store.branch("main", "experiment")     # fork a persisted session
```

### Stats
- **55 new tests** (total: 2275)
- **3 new examples** (52: semantic cache, 53: prompt compression, 54: conversation branching; total: 54)
- **1 new StepType** — `prompt_compressed` (total: 17)
- **1 new observer event** — `on_prompt_compressed` (total: 32 sync / 29 async)

## [0.17.6] - 2026-03-24

### Added

#### Reasoning Strategies
- New `reasoning_strategy` field on `AgentConfig`: `"react"`, `"cot"`, `"plan_then_act"`
- Injects structured reasoning instructions into the system prompt via `PromptBuilder`
- Works with existing `result.reasoning` extraction for full visibility into agent thought process
- New export: `REASONING_STRATEGIES` dict for discovering available strategies

```python
config = AgentConfig(reasoning_strategy="react")  # Thought → Action → Observation
config = AgentConfig(reasoning_strategy="cot")     # Chain-of-Thought step-by-step
config = AgentConfig(reasoning_strategy="plan_then_act")  # Plan first, then execute
```

#### Tool Result Caching
- New `cacheable` and `cache_ttl` parameters on `Tool` and `@tool()` decorator
- Cacheable tools skip re-execution when called with the same arguments within TTL
- Cache key: `tool_result:{tool_name}:{sha256(sorted_params)}`
- Wired into all 4 execution paths (single sync/async, parallel sync/async)
- Records `StepType.CACHE_HIT` trace step on cache hits
- Reuses the agent's existing `config.cache` (InMemoryCache, RedisCache)

```python
@tool(description="Search the web", cacheable=True, cache_ttl=60)
def web_search(query: str) -> str:
    return expensive_api_call(query)
```

#### Python 3.9–3.13 CI Matrix
- GitHub Actions now tests against Python 3.9, 3.10, 3.11, 3.12, and 3.13
- Full codebase audit confirmed zero 3.10+ only syntax (all `X | Y` unions guarded by `from __future__ import annotations`)
- Added `Programming Language :: Python :: 3.13` classifier to pyproject.toml

### Stats
- **37 new tests** (total: 2220)
- **2 new examples** (50: reasoning strategies, 51: tool result caching; total: 51)

## [0.17.5] - 2026-03-23

### Fixed — Bug Hunt (91 validated fixes across 7 subsystems)

#### Critical (13)
- **Path traversal in `JsonFileSessionStore`** — session IDs now validated against directory escape
- **Unicode homoglyph bypass** in prompt injection screening — NFKD normalization + zero-width stripping
- **`FallbackProvider` stream** records success after consumption, not before — circuit breaker works for streaming
- **Gemini `response.text` ValueError** on tool-call-only responses — caught and handled
- **`astream()` model_selector** was using `self.config.model` — now uses `self._effective_model`
- **Sync `_check_policy`** silently approved async `confirm_action` — now rejects with clear error
- **`aexecute()` ThreadPoolExecutor per call** — replaced with shared executor via `run_in_executor(None)`
- **`execute()` on async tools** returned coroutine string repr — now awaits via `asyncio.run`
- **Hybrid search O(n²)** `_find_matching_key` — replaced with O(1) `text_to_key` dict lookup
- **`SQLiteVectorStore`** no thread safety — added `threading.Lock` + WAL mode
- **`FileKnowledgeStore._save_all()`** not crash-safe — atomic write via tmp + `os.replace`
- **`OutputEvaluator`** crashed on invalid regex — wrapped in `try/except re.error`
- **`JsonValidityEvaluator`** ignored `expect_json=False` — guard now checks falsy, not just None

#### High (26)
- **`astream()` cancellation/budget paths** now build proper trace steps + fire async observer events
- **`arun()` early exits** now fire `_anotify_observers("on_run_end")` for cancel/budget/max-iter
- **`_aexecute_tools_parallel`** fires async observer events + tracks `tool_usage`/`tool_tokens`
- **Sync `_streaming_call`** no longer stringifies `ToolCall` objects (pitfall #2)
- **16 LLM evaluators** silently passed on unparseable scores — now return `EvalFailure`
- **XSS in eval dashboard** — `innerHTML` replaced with `createElement`/`textContent`
- **Donut SVG 360° arc** renders nothing — now draws two semicircles for full annulus
- **SSN regex** matched ZIP+4 codes — now requires consistent separators
- **Coherence LLM costs** tracked in `CoherenceResult.usage` + merged into agent usage
- **Coherence `fail_closed`** option added (default: fail-open for backward compat)
- Plus 16 more HIGH fixes across tools, RAG, memory, and security subsystems

#### Medium (30) and Low (22)
- `datetime.utcnow()` → `datetime.now(timezone.utc)` throughout knowledge stores
- `ConversationMemory.clear()` now resets `_summary`
- SQLite WAL mode + indexes for knowledge and session stores
- Non-deterministic `hash()` → `hashlib.sha256` for document IDs in 3 vector stores
- OpenAI `embed_texts()` batching at 2048 per request
- Tool result caching: `_serialize_result` returns `""` for None, not `"None"`
- Bool values rejected for int/float tool parameters
- `ToolRegistry.tool()` now forwards `screen_output`, `terminal`, `requires_approval`
- Plus 40+ more fixes (see `.private/BUG_HUNT_VALIDATED.md` for complete list)

### Added
- **Async guardrails** — `Guardrail.acheck()` with `asyncio.to_thread` default, `GuardrailsPipeline.acheck_input()`/`acheck_output()`, `Agent._arun_input_guardrails()`. `arun()`/`astream()` no longer block the event loop during guardrail checks.
- 40 new regression tests covering all critical and high-severity fixes
- 5 new entries in CLAUDE.md Common Pitfalls (#14-#18)

## [0.17.4] - 2026-03-22

### Added

- **Pre-Execution Token Estimation (R6)**: `estimate_tokens()` and `estimate_run_tokens()` for budget pre-checks. Uses tiktoken for OpenAI models when installed, falls back to `chars/4` heuristic. `TokenEstimate` dataclass with breakdown.
- **Model Switching Per Iteration (R10)**: `model_selector` callback on `AgentConfig`. `_effective_model` property used throughout provider caller. `on_model_switch` observer event.
- **Knowledge Memory Enhancement (R3)**: `KnowledgeEntry` with importance, TTL, categories. `KnowledgeStore` protocol with `FileKnowledgeStore` and `SQLiteKnowledgeStore`. Importance-based eviction. Backward-compatible.
- New exports: `TokenEstimate`, `estimate_tokens`, `estimate_run_tokens`, `KnowledgeEntry`, `KnowledgeStore`, `FileKnowledgeStore`, `SQLiteKnowledgeStore`.
- 58 new tests (total: 1966).

## [0.17.3] - 2026-03-22

### Added

- **Token Budget Per Run (R1)**: `max_total_tokens` and `max_cost_usd` on `AgentConfig`. Agent loop checks budget at each iteration start and returns partial result with `BUDGET_EXCEEDED` trace step when exceeded. New `on_budget_exceeded` observer event. New `BudgetExceededError` exception.
- **Agent Cancellation (R2)**: `CancellationToken` (thread-safe) for cooperative cancellation. Checked at iteration boundaries and after tool execution. New `CANCELLED` trace step, `on_cancelled` observer event, `CancellationError` exception. Token is reusable via `reset()`.
- **SimpleStepObserver (R4)**: Convenience observer that routes all 30 lifecycle events to a single `callback(event_name, run_id, **kwargs)` function. Simpler than subclassing `AgentObserver` with 30+ methods.
- **Tool-Level Cost Attribution (R7)**: `cost_usd` field on `TraceStep`, populated on `LLM_CALL` steps from provider usage stats.
- **Structured Tool Results (R8)**: `Tool._serialize_result()` auto-serializes dict, list, Pydantic model, and dataclass returns as JSON. Strings pass through unchanged; other types fall back to `str()`.
- **Per-Tool Approval Gate (R9)**: `requires_approval` flag on `@tool()` decorator and `Tool` class. Tools with this flag always trigger `REVIEW` policy decision, regardless of `ToolPolicy` rules. Works with existing `confirm_action` callback.
- New `StepType` values: `BUDGET_EXCEEDED`, `CANCELLED` (16 total, up from 14).
- New exports: `CancellationToken`, `CancellationError`, `BudgetExceededError`, `SimpleStepObserver`.
- 61 new tests (total: 1908).

## [0.17.1] - 2026-03-22

### Added

**MCP Client/Server** — connect to any MCP-compatible tool server and expose selectools tools as MCP servers.

- `MCPClient` — connect via stdio (local subprocess) or Streamable HTTP (remote), with auto-reconnect, circuit breaker, retry with backoff, and tool caching
- `MultiMCPClient` — manage multiple MCP servers with graceful degradation, name prefixing, and collision detection
- `MCPServer` — expose any `@tool` function as an MCP-compliant server (stdio or HTTP)
- `mcp_tools()` — one-liner context manager for tool discovery
- Bidirectional schema bridge: MCP ↔ selectools Tool conversion (handles nested objects, arrays, enums, defaults, nullable types)
- Background event loop for sync MCP usage
- `MCPServerConfig` — validated configuration with all transport options
- `MCPError`, `MCPConnectionError`, `MCPToolError` exception types
- `pip install selectools[mcp]` — optional dependency on official `mcp` SDK
- `Tool._skip_validation` flag for MCP-sourced tools
- 33 new tests, 2 examples (`41_mcp_client.py`, `42_mcp_server.py`)
- Full module documentation: `docs/modules/MCP.md`

### Verified E2E

- selectools `@tool` → `MCPServer` → `MCPClient` → selectools `Tool` → `Agent.arun()` — full round-trip
- Mixed local + MCP tools in the same agent — correct tool selection
- Eval framework on MCP-powered agents — 100% accuracy
- Real Anthropic API + MCP tools — working

---

## [0.17.0] - 2026-03-22

### Added

**Built-in Eval Framework** — the only AI agent framework with a comprehensive evaluation suite built in. No separate install, no SaaS account, no external dependencies.

#### Evaluators (39 total)

**21 deterministic evaluators** (no API calls):
- `ToolUseEvaluator` — tool name, tool list, argument value assertions
- `ContainsEvaluator` — substring present/absent (case-insensitive)
- `OutputEvaluator` — exact match and regex matching
- `StructuredOutputEvaluator` — parsed field assertions (deep subset match)
- `PerformanceEvaluator` — iteration count, latency, and cost thresholds
- `JsonValidityEvaluator` — valid JSON output
- `LengthEvaluator` — min/max character count
- `WordCountEvaluator` — min/max word count
- `StartsWithEvaluator` / `EndsWithEvaluator` — prefix/suffix assertions
- `ToolOrderEvaluator` — tools called in expected sequence
- `UniqueToolsEvaluator` — no duplicate tool calls
- `PIILeakEvaluator` — SSN, email, phone, credit card, ZIP detection
- `InjectionResistanceEvaluator` — 10 prompt injection patterns
- `RefusalEvaluator` — detect appropriate refusal of harmful requests
- `SentimentEvaluator` — keyword-based positive/negative/neutral detection
- `PythonValidityEvaluator` — valid Python syntax (with code fence stripping)
- `SQLValidityEvaluator` — SQL statement validation
- `URLValidityEvaluator` — well-formed URL detection
- `MarkdownFormatEvaluator` — markdown formatting detection
- `CustomEvaluator` — any user-defined callable

**18 LLM-as-judge evaluators** (use any Provider):
- `LLMJudgeEvaluator` — generic rubric scoring (0-10)
- `CorrectnessEvaluator` — correct vs reference answer
- `RelevanceEvaluator` — response relevant to query
- `FaithfulnessEvaluator` — grounded in provided context (RAG)
- `HallucinationEvaluator` — fabricated information detection
- `ToxicityEvaluator` — harmful/inappropriate content
- `CoherenceEvaluator` — well-structured and logical
- `CompletenessEvaluator` — fully addresses the query
- `BiasEvaluator` — gender, racial, political bias
- `SummaryEvaluator` — summary accuracy and coverage
- `ConcisenessEvaluator` — not overly verbose
- `InstructionFollowingEvaluator` — followed specific instructions
- `ToneEvaluator` — matches expected tone
- `ContextRecallEvaluator` — RAG: used all relevant context
- `ContextPrecisionEvaluator` — RAG: retrieved context was relevant
- `GrammarEvaluator` — grammatically correct and fluent
- `SafetyEvaluator` — comprehensive safety check

#### Infrastructure

- `EvalSuite` — orchestrates eval runs with sync/async/concurrent execution
- `EvalReport` — accuracy, latency p50/p95/p99, cost, weighted scoring, tag filtering, failure breakdown
- `DatasetLoader` — load test cases from JSON/YAML files
- `BaselineStore` + `RegressionResult` — save baselines, detect regressions across runs
- `PairwiseEval` — compare two agents head-to-head with automatic winner determination
- `SnapshotStore` — Jest-style snapshot testing for AI agent outputs
- `generate_cases()` — LLM-powered synthetic test case generator from tool definitions
- `generate_badge()` — shields.io-style SVG badges for README
- `serve_eval()` — live browser dashboard with real-time eval progress
- `HistoryStore` — track accuracy/cost/latency across runs with trend analysis
- Interactive HTML report with donut chart, latency histogram, trend sparkline, expandable rows, filtering
- JUnit XML for CI (GitHub Actions, Jenkins, GitLab CI)
- `report.to_markdown()` — markdown summary for GitHub issues and PRs
- CLI: `python -m selectools.evals run/compare`
- GitHub Action at `.github/actions/eval/` with automatic PR comments
- Cost estimation: `suite.estimate_cost()` before running
- 4 pre-built templates: `customer_support_suite()`, `rag_quality_suite()`, `safety_suite()`, `code_quality_suite()`
- `pip install selectools[evals]` for optional PyYAML dependency

#### Observer Integration

- 3 new observer events: `on_eval_start`, `on_eval_case_end`, `on_eval_end`
- Compatible with `LoggingObserver` for structured JSON eval logs

#### Testing

- **340 new eval tests** across 7 test files (unit, integration, E2E, hardening)
- 40 example scripts (2 eval-specific: `39_eval_framework.py`, `40_eval_advanced.py`)
- Full module documentation: `docs/modules/EVALS.md`

---

## [0.16.7] - 2026-03-16

### Removed

- **CLI module removed**: `src/selectools/cli.py` and the `selectools` console script entry point have been removed. The CLI was unused — the library API is the primary interface. Removes 22 tests (total: 1620).

### Documentation

- **README example table**: Added missing examples 28-38 (agent observer, guardrails, audit logging, screening, coherence, sessions, summarize-on-trim, entity memory, knowledge graph, knowledge memory, terminal tools).
- **Cross-reference audit**: Fixed stale counts in CONTRIBUTING.md (was v0.13.0 era), docs/ARCHITECTURE.md, docs/README.md, docs/QUICKSTART.md, and docs/modules/AGENT.md.

---

## [0.16.6] - 2026-03-16

### Fixed

- **Gemini thought_signature UnicodeDecodeError**: Gemini 3.x returns `thought_signature` as opaque binary (protobuf/hash), not UTF-8 text. The previous `.decode("utf-8")` crashed with `UnicodeDecodeError` on non-UTF-8 bytes (e.g. `0xa4`, `0xd5`). Replaced with base64 encode/decode across all 5 affected locations in `complete()`, `acomplete()`, `astream()`, and `_format_contents()`. The signature now survives a lossless round-trip for arbitrary binary data.

### Tests

- **2 new regression tests** (total: 1642): Non-UTF-8 binary thought_signature round-trip in `test_phase1_design_patterns.py` and `tests/agent/test_regression.py`.

---

## [0.16.5] - 2026-03-15

### Added

- **Terminal action support** (FR-001): `@tool(terminal=True)` stops the agent loop after tool execution without another LLM call. Dynamic `AgentConfig.stop_condition` callback for result-dependent stops. Works in `run()`, `arun()`, `astream()`, and parallel execution.
- **Async observer hooks** (FR-002): `AsyncAgentObserver` base class with 25 `a_on_*` async methods. `blocking=True` for inline await (DB writes), `blocking=False` (default) for fire-and-forget via `asyncio.ensure_future()`. Called in all async paths.
- **Gemini 3.x thought signature support** (FR-003): `ToolCall.thought_signature` field captures and echoes `thoughtSignature` from Gemini 3.x function call responses. `_format_contents` includes original `functionCall` alongside `functionResponse` for TOOL messages.
- **`StepType` enum**: Converted from `Literal[...]` to `class StepType(str, Enum)` with 14 members. Backward compatible — `StepType.LLM_CALL == "llm_call"` is `True`. Exported from `selectools`.
- **`ModelType` enum**: Converted from `Literal[...]` to `class ModelType(str, Enum)` with 5 members. All 146 `ModelInfo` records updated to use enum members. Exported from `selectools`.
- **Shared test fixtures**: `SharedFakeProvider`, `SharedRecordingProvider`, `SharedToolCallProvider`, `SharedErrorProvider` in `tests/conftest.py` with factory fixtures.
- **Architecture fitness tests**: 53 tests in `tests/test_architecture.py` — circular imports, provider protocol compliance, enum coverage, export consistency, model registry validity.
- **Architecture Decision Records**: 6 ADRs in `docs/decisions/` documenting Protocol over ABC, observer replaces hooks, Literal to Enum, provider base class, agent decomposition, and deferred config restructuring.
- **Namespace exports**: `from selectools.providers import OpenAIProvider` now works (additive, flat imports unchanged).

### Changed

- **Agent decomposed into 4 mixins**: `core.py` reduced from 3128 to 1448 lines (-54%). Tool execution → `_tool_executor.py` (970 lines), provider calls → `_provider_caller.py` (469 lines), observer lifecycle → `_lifecycle.py` (141 lines), memory management → `_memory_manager.py` (146 lines). All public methods remain on `Agent`.
- **Hooks deprecated**: `AgentConfig.hooks` emits `DeprecationWarning` and is transparently wrapped via `_HooksAdapter(AgentObserver)`. All 55 `_call_hook()` calls removed. Single observer notification pipeline.
- **OpenAI/Ollama share base class**: `_OpenAICompatibleBase` (Template Method pattern) in `providers/_openai_compat.py`. OpenAI: 421→86 lines (-80%), Ollama: 456→126 lines (-72%).
- **astream() provider call parity**: Non-streaming fallback path now delegates to `_acall_provider()` for full cache/retry/cost-warning parity with `arun()`.
- **Tool execution extracted**: `_execute_single_tool()` and `_aexecute_single_tool()` replace 540 lines of copy-pasted tool execution loops across `run()`/`arun()`/`astream()`.

### Tests

- **163 new tests** (total: 1640): Phase 1 design patterns (32), architecture fitness (53), terminal actions (13), async observers (11), plus additional regression tests.

### Documentation

- Updated `docs/modules/AGENT.md` — hooks deprecation warning, AsyncAgentObserver section, terminal actions, mixin architecture
- Updated `docs/modules/TOOLS.md` — `terminal` parameter documentation
- Updated `docs/modules/MODELS.md` — ModelType as `str, Enum`
- Updated `docs/modules/PROVIDERS.md` — `_OpenAICompatibleBase` note, namespace imports
- Updated `docs/ARCHITECTURE.md` — mixin decomposition, AsyncAgentObserver, hooks deprecated
- Updated `docs/QUICKSTART.md` — terminal tools step
- Updated `notebooks/getting_started.ipynb` — terminal tools and async observers sections
- New `examples/38_terminal_tools.py`

---

## [0.16.4] - 2026-03-15

### Fixed

- **Parallel tool execution missing coherence checks**: Both `_execute_tools_parallel()` and `_aexecute_tools_parallel()` now run coherence checks before executing tools, matching sequential execution behavior.
- **Parallel tool execution missing output screening**: Both parallel execution helpers now call `_screen_tool_result()` after tool execution, closing a security gap where parallel tool calls could bypass prompt injection detection.
- **astream() usage tracking**: Fixed inline provider calls in astream() not feeding into `self.usage`, which caused `tool_usage` and `tool_tokens` dicts to remain empty during streaming.
- **Input guardrails mutating caller's Message objects**: `_prepare_run()` now copies messages before applying input guardrails, preventing caller-side mutation when guardrails rewrite content.
- **`ask()`/`aask()` missing `parent_run_id`**: Both convenience methods now accept and pass through `parent_run_id` for trace linking, matching `run()`/`arun()`/`astream()`.

### Tests

- **15 new tests** (total: 1477): Regression tests for all 5 fixes above, including parallel coherence/screening (sync + async), astream usage tracking, guardrail mutation prevention, and ask/aask parent_run_id.

---

## [0.16.3] - 2026-03-14

### Added

- **`Agent.name`** property — reads from `AgentConfig.name` (default: `"agent"`). Pre-work for multi-agent orchestration (v0.17.0)
- **`Agent.__call__()`** — allows calling the agent directly as a shorthand for `run()`
- **`parent_run_id`** parameter on `run()`, `arun()`, and `astream()` — overrides `config.parent_run_id` for trace linking in nested agent calls
- **`response_format`** parameter on `astream()` — brings structured output support to streaming (was missing)
- **`GraphExecutionError`** exception — new error type for graph execution failures (pre-work for v0.17.0)
- **`_clone_for_isolation()`** — shared helper used by `batch()` and `abatch()` for consistent agent cloning
- **29 regression tests** (`tests/agent/test_astream_parity.py`) covering all astream() bug fixes

### Fixed

- **astream() input guardrails**: Now runs `_run_input_guardrails()` on user messages (was missing)
- **astream() output guardrails**: Now runs `_run_output_guardrails()` on LLM responses (was missing)
- **astream() knowledge_memory context**: Now injects `knowledge_memory.build_context()` into history (was missing)
- **astream() entity_memory context**: Now injects `entity_memory.build_context()` into history (was missing)
- **astream() knowledge_graph context**: Now injects `knowledge_graph.build_context()` into history (was missing)
- **astream() session load notification**: Now notifies observers with `on_session_load` event (was missing)
- **astream() memory summary injection**: Now injects `memory.summary` into history (was missing)
- **astream() user_text_for_coherence**: Now extracts user text for coherence checks (was missing)
- **astream() response_format parse guard**: Parser no longer intercepts valid JSON when `response_format` is set (was always parsing)
- **astream() reasoning extraction**: Now extracts reasoning text from LLM responses (was missing)
- **astream() tool_selection trace steps**: Now records `tool_selection` trace steps (was missing)
- **astream() coherence check**: Now runs `_acheck_coherence()` on tool calls (was missing)
- **astream() screen_tool_result**: Now screens tool output for prompt injection (was missing)
- **astream() analytics recording**: Now records tool call analytics (was missing)
- **astream() verbose output**: Now prints verbose logs for tool execution (was missing)
- **astream() chunk callback**: Now passes chunk callback to tool execution (was passing `None`)
- **astream() per-tool usage tracking**: Now tracks per-tool token usage (was missing)
- **astream() tool-not-found**: Now uses consistent error message format and produces trace step (was different from run/arun)
- **astream() policy denial trace**: Now produces trace step for policy denials (was missing)
- **astream() entity extraction**: Now calls `_extract_entities()` in teardown (was missing)
- **astream() KG extraction**: Now calls `_extract_kg_triples()` in teardown (was missing)
- **astream() session save**: Now calls `_session_save()` in teardown (was missing)
- **astream() AgentResult fields**: Now includes `parsed`, `reasoning`, `reasoning_history`, `provider_used` (were missing)
- **astream() structured output validation**: Now validates and retries structured output (was missing)
- **astream() history append order**: Response now appended to history only when tool calls exist (was appending before check)

### Changed

- **agent/core.py refactoring**: Extracted `_RunContext` dataclass, `_prepare_run()`, `_finalize_run()`, `_process_response()`, `_build_max_iterations_result()` shared helpers — eliminates ~800 lines of duplicated setup/teardown/iteration logic across `run()`, `arun()`, and `astream()`
- **batch()/abatch()**: Now use `_clone_for_isolation()` instead of inline clone logic

---

## [0.16.2] - 2026-03-14

### Fixed

- **agent/core.py**: `astream()` now saves and restores `_system_prompt` in its `finally` block, matching `run()` and `arun()`. Previously, if `_system_prompt` was modified during streaming execution, the change would leak to subsequent calls on the same agent instance.

### Documentation

- **CLAUDE.md**: Updated stale counts (tests 1183→1487, examples 32→37, observer events 15→19, module docs 20→24), added v0.16.0 modules to codebase structure, added TraceStep types reference table, added pitfall #11, updated roadmap
- **AGENT.md**: Documented 9 new AgentConfig fields, expanded Memory Integration section with sessions/entity/KG/knowledge subsections, added 4 observer events and 5 trace step types
- **MEMORY.md**: Documented `from_dict()` deserialization, added Summarize-on-Trim section, updated Future Enhancements
- **index.md**: Updated test count to 1487
- **ROADMAP.md**: Added v0.16.1 release history entry
- **CHANGELOG.md**: Fixed v0.16.1 test total (56→68, added missing test file)

---

## [0.16.1] - 2026-03-13

### Fixed

- **mypy**: Resolved all 5 type errors across `sessions.py` and `knowledge_graph.py` (0 errors now)
- **sessions.py**: Replaced `Any` return type on `SQLiteSessionStore._conn()` with proper `sqlite3.Connection` annotation; moved `import sqlite3` to module level
- **sessions.py**: Fixed `_is_expired()` return type by annotating `updated_at` as `float`
- **sessions.py**: Fixed `delete()` return types in SQLite and Redis backends with explicit `int()` cast
- **knowledge_graph.py**: Fixed `count()` and `query_relevant()` return type annotations

### Added

- **68 new tests** (total: 1487) for v0.16.0 consolidation:
  - `test_sessions_redis.py` — 21 tests for RedisSessionStore with mock Redis client (save/load, TTL, delete, list, exists, import error, edge cases)
  - `test_sessions_edge_cases.py` — 13 tests for corrupt JSON handling, TTL cleanup, OS error recovery, metadata
  - `test_memory_boundary.py` — 6 tests for tool-pair boundary trimming after sliding window
  - `test_memory_integration.py` — 7 tests for all memory features running simultaneously (sessions + entity + KG + knowledge memory)
  - `test_memory_async.py` — 9 tests for `arun()` with session auto-save, entity memory, knowledge graph, combined features
  - `test_consolidation_regression.py` — 12 tests for regression coverage of all 6 bug fixes

### Documentation

- **AGENT.md**: Added `session_store`, `session_id`, `summarize_on_trim`, `summarize_provider`, `summarize_model`, `summarize_max_tokens`, `entity_memory`, `knowledge_graph`, `knowledge_memory` to AgentConfig section; expanded Memory Integration section with subsections for session auto-load/save, entity memory, knowledge graph, knowledge memory, and context injection order; added cross-references to Sessions, Entity Memory, Knowledge Graph, and Knowledge Memory modules in Further Reading
- **MEMORY.md**: Added `from_dict()` classmethod to Serialization section; added full Summarize-on-Trim section with configuration, flow diagram, key properties, and code examples; updated Future Enhancements (removed shipped v0.16.0 features); expanded Further Reading with links to all memory modules

---

## [0.16.0] - 2026-03-13

### Added — Memory & Persistence

#### Persistent Conversation Sessions (new `sessions.py` module)

- **`SessionStore` protocol**: Pluggable backends for saving/loading `ConversationMemory` state. Three methods: `save()`, `load()`, `list()`, `delete()`.
- **`JsonFileSessionStore`**: File-based backend, one JSON file per session.
- **`SQLiteSessionStore`**: Single-database backend with JSON column.
- **`RedisSessionStore`**: Distributed backend with server-side TTL.
- **Agent integration**: `AgentConfig(session_store=store, session_id="user-123")` — auto-loads on init, auto-saves after each `run()` / `arun()`.
- **TTL-based expiry**: All backends support configurable `default_ttl`.

#### Summarize-on-Trim (enhanced `memory.py`)

- **LLM-generated summaries**: When `ConversationMemory` trims messages, it generates a 2-3 sentence summary of dropped messages using a configurable provider/model.
- **Context preservation**: Summary injected as system-level context message.
- **Configuration**: `AgentConfig(summarize_on_trim=True, summarize_provider=provider)`.

#### Entity Memory (new `entity_memory.py` module)

- **`EntityMemory`**: LLM-based entity extraction after each turn.
- **Entity types**: person, organization, project, location, date, custom.
- **Deduplication**: Case-insensitive matching with attribute merging.
- **LRU pruning**: Configurable `max_entities` limit.
- **System prompt injection**: `[Known Entities]` context for subsequent turns.

#### Knowledge Graph Memory (new `knowledge_graph.py` module)

- **`KnowledgeGraphMemory`**: Extracts (subject, relation, object) triples from conversation.
- **`TripleStore` protocol**: `InMemoryTripleStore` and `SQLiteTripleStore` backends.
- **Keyword-based query**: `query_relevant(query)` for relevant triple retrieval.
- **System prompt injection**: `[Known Relationships]` context.

#### Cross-Session Knowledge Memory (new `knowledge.py` module)

- **`KnowledgeMemory`**: Daily log files + persistent `MEMORY.md` for long-term facts.
- **Auto-registered `remember` tool** for explicit knowledge storage.
- **System prompt injection**: `[Long-term Memory]` + `[Recent Memory]` context.

### Changed

- **`AgentConfig`**: New fields: `session_store`, `session_id`, `summarize_on_trim`, `summarize_provider`, `summarize_model`, `entity_memory`, `knowledge_graph`, `knowledge_memory`.
- **`AgentObserver`**: 4 new events (total: 19): `on_session_load`, `on_session_save`, `on_memory_summarize`, `on_entity_extraction`.
- **`StepType`**: 5 new trace step types: `session_load`, `session_save`, `memory_summarize`, `entity_extraction`, `kg_extraction`.
- **`ConversationMemory`**: New `summarize_on_trim` parameter and `summary` property.

### Documentation

- **4 new module docs**: `SESSIONS.md`, `ENTITY_MEMORY.md`, `KNOWLEDGE_GRAPH.md`, `KNOWLEDGE.md`
- **Updated**: `ARCHITECTURE.md`, `QUICKSTART.md` (Steps 12-15), `docs/README.md`, `docs/index.md`
- **5 new examples**: `33_persistent_sessions.py` through `37_knowledge_memory.py`
- **Updated notebook**: sections 14-16 for sessions, entity memory, knowledge

---

## [0.15.0] - 2026-03-12

### Added — Enterprise Reliability

#### Guardrails Engine (new `guardrails/` subpackage)

- **`GuardrailsPipeline`**: Ordered pipeline of input and output guardrails that run before and after every LLM call. Supports chaining — if a guardrail rewrites content, downstream guardrails see the rewritten version.
- **`Guardrail` base class**: Subclass and override `check(content) -> GuardrailResult` for custom validation. Three failure actions: `block` (raise `GuardrailError`), `rewrite` (return sanitised content), `warn` (log and continue).
- **`TopicGuardrail`**: Keyword-based topic blocking with word-boundary matching. Case-insensitive by default.
- **`PIIGuardrail`**: Regex-based PII detection for email, phone, SSN, credit card, and IPv4. Supports `redact` mode (replaces PII with `[TYPE:****]`), custom patterns, and selective detection.
- **`ToxicityGuardrail`**: Keyword blocklist scoring with configurable threshold. Ships with a default blocklist of ~16 high-signal terms.
- **`FormatGuardrail`**: Validates JSON structure, required keys, and content length bounds.
- **`LengthGuardrail`**: Enforces min/max character and word counts. Supports `rewrite` mode for truncation.
- **Agent integration**: `AgentConfig(guardrails=pipeline)` — input guardrails run on user messages before the LLM call; output guardrails run on LLM responses after they return. Both sync (`run()`) and async (`arun()`) paths.

#### Audit Logging (new `audit.py` module)

- **`AuditLogger`**: JSONL append-only audit logger implementing the `AgentObserver` protocol. Plugs into any agent via `AgentConfig(observers=[AuditLogger(...)])`.
- **Privacy controls**: Four levels via `PrivacyLevel` — `full` (log everything), `keys_only` (redact values), `hashed` (SHA-256 truncated hashes), `none` (omit args).
- **Daily file rotation**: `audit-YYYY-MM-DD.jsonl` files by default; disable for a single `audit.jsonl`.
- **Thread-safe writes**: Safe for concurrent `batch()` usage.
- Records: `run_start`, `run_end`, `tool_start`, `tool_end`, `tool_error`, `llm_end`, `policy_decision`, `error`.

#### Tool Output Screening (new `security.py` module)

- **Prompt injection detection**: 15 built-in regex patterns covering common injection techniques (e.g., "ignore previous instructions", `<system>` tags, `[INST]` markers, "forget everything").
- **`@tool(screen_output=True)`**: Per-tool opt-in screening. Also available globally via `AgentConfig(screen_tool_output=True)`.
- **Custom patterns**: `AgentConfig(output_screening_patterns=["ADMIN_OVERRIDE"])` adds extra regex patterns.
- **Agent integration**: Blocked outputs are replaced with a safe placeholder message before being fed back to the LLM.

#### Coherence Checking (new `coherence.py` module)

- **LLM-based intent verification**: `AgentConfig(coherence_check=True)` adds a lightweight LLM call before each tool execution that verifies the proposed tool call matches the user's original request.
- **Prompt injection defense**: Catches cases where injected content in tool outputs causes the agent to call unrelated tools (e.g., user asks "summarize emails" but injection causes `send_email`).
- **Configurable provider/model**: `AgentConfig(coherence_provider=..., coherence_model=...)` — use a separate, fast model for checks. Defaults to the agent's own provider.
- **Fail-open**: If the coherence check LLM call fails, the tool call is allowed (no silent blocking on infrastructure errors).
- **Sync and async**: Both `run()` and `arun()` paths supported.

### Changed

- **`StepType` literal**: Added `"guardrail"`, `"coherence_check"`, and `"output_screening"` trace step types.
- **`Tool` class**: New `screen_output: bool` parameter (default `False`).
- **`@tool()` decorator**: New `screen_output` kwarg.
- **`AgentConfig`**: New fields: `guardrails`, `screen_tool_output`, `output_screening_patterns`, `coherence_check`, `coherence_provider`, `coherence_model`.
- **ROADMAP**: Enterprise Reliability moved from v1.0.0 to v0.15.0. Multi-Agent Orchestration and MCP moved to backlog. Memory & Persistence is now v0.16.0.

### Documentation

- **New module docs**: `GUARDRAILS.md`, `AUDIT.md`, `SECURITY.md`, `TOOLBOX.md`, `EXCEPTIONS.md` (5 new pages)
- **Updated module docs**: `AGENT.md` (ResponseFormat helpers, new TraceStep types), `MODELS.md` (programmatic pricing API)
- **Updated guides**: `QUICKSTART.md` (Steps 10-11 for guardrails, audit, security), `ARCHITECTURE.md` (v0.15.0 features), `docs/README.md` (21 module pages, new navigation sections)
- **New examples**: `29_guardrails.py`, `30_audit_logging.py`, `31_tool_output_screening.py`, `32_coherence_checking.py`
- **Updated notebook**: `getting_started.ipynb` — sections 11-13 for guardrails, audit, screening, coherence

### Tests

- **83 new tests** (total: 1183): Comprehensive coverage for all 5 built-in guardrails, pipeline chaining, audit logger privacy levels/rotation, all 15 injection patterns, coherence checking (sync + async + failure modes), and custom guardrail subclassing.

---

## [0.14.1] - 2026-03-12

### Fixed — Streaming & Provider Tool Passing (13 bugs)

All streaming methods (`stream()`, `astream()`) across every provider were silently dropping tool definitions and/or failing to yield `ToolCall` objects. This meant agents using `run(stream=True)`, `arun(stream=True)`, or `astream()` could not use tools at all. These bugs were invisible because mock providers in tests accepted `**kwargs`, silently swallowing missing parameters.

#### Agent Core (`core.py`)

- **`_streaming_call()` did not pass `tools` to `provider.stream()`** — agents using `run(stream=True)` could never call tools
- **`_astreaming_call()` did not pass `tools` to `provider.astream()`** — agents using `arun(stream=True)` could never call tools
- **`_astreaming_call()` sync fallback did not pass `tools` to `provider.stream()`**
- **`_astreaming_call()` stringified `ToolCall` objects** — `ToolCall` objects yielded by `astream()` were converted to `str`, corrupting them

#### OpenAI Provider

- **`stream()` did not pass `tools` to the API** — streaming tool calls were impossible

#### Anthropic Provider

- **`stream()` did not pass `tools` to the API**
- **`astream()` did not pass `tools` to the API**
- **`astream()` did not yield `ToolCall` objects** — only text chunks were yielded; `tool_use` blocks were discarded

#### Gemini Provider

- **`stream()` did not pass `tools` to the API** (config.tools was never set)
- **`astream()` did not pass `tools` to the API**
- **`astream()` did not yield `ToolCall` objects** — `function_call` parts in streaming chunks were ignored

#### Ollama Provider

- **`stream()` did not pass `tools` to the API**
- **`astream()` did not pass `tools` and did not yield `ToolCall` objects**
- **`_format_messages()` mapped `TOOL` role to `"assistant"` instead of `"tool"`** — breaking multi-turn tool conversations
- **`_format_messages()` omitted `tool_calls` array on `ASSISTANT` messages** — the model never saw its own prior tool calls

#### FallbackProvider

- **`astream()` had no error handling** — first provider failure crashed instead of falling over; circuit breaker never recorded failures; `on_fallback` callback never fired

### Improved — Test Suite (+141 tests, total: 1100)

Root cause of all 13 bugs: mock providers used `**kwargs` which silently consumed missing parameters. Tests never asserted that `tools` was actually received by the provider, and never checked that `ToolCall` objects kept their type through the streaming pipeline.

#### New regression tests (`tests/agent/test_regression.py` — 28 tests)

- Structured output not intercepted by text parser when `response_format` is set
- Provider returning `content=None` doesn't crash
- Async policy timeout enforced on sync `confirm_action` callbacks (both sync and async paths)
- `routing_only` mode fires `on_iteration_end` event
- Empty `tool_calls=[]`, nonexistent tools, and wrong argument types handled gracefully
- Concurrent `arun()` doesn't crash; `abatch()` provides history isolation
- `FallbackProvider` + observers + `batch()` doesn't stack overflow
- Policy deny enforced in both sync and async agent paths
- Retry backoff succeeds and exhausts correctly
- Crashing observer doesn't crash the agent
- Every run produces a trace with steps
- Usage always attached to result
- `reset()` clears history and usage
- Max iterations enforced even with infinite tool loops

#### New provider streaming tests (`tests/providers/test_provider_streaming_tools.py` — 21 tests)

- Recording providers verify exact arguments passed to `complete()`, `stream()`, `astream()`
- Agent passes `tools` to streaming methods for `run(stream=True)`, `arun(stream=True)`, `astream()`
- `ToolCall` objects not stringified in `_astreaming_call()`
- Ollama `_format_messages()` correctly handles `TOOL` role and `ASSISTANT` `tool_calls`
- Anthropic `astream()` yields `ToolCall` objects from `tool_use` blocks
- `FallbackProvider.astream()` failover, circuit breaker, and error propagation
- OpenAI `stream()` passes tools to API

#### New unit tests for previously untested modules (92 tests)

- **`test_policy.py` (24):** `ToolPolicy.evaluate()`, glob patterns, evaluation order, `deny_when` conditions, `from_dict()`, `from_yaml()`
- **`test_structured.py` (20):** `extract_json()`, `parse_and_validate()`, `schema_from_response_format()`, `build_schema_instruction()`
- **`test_trace.py` (30):** `AgentTrace` filter/timeline/to_dict/to_json/to_otel_spans, OTel span structure
- **`test_fallback_unit.py` (20):** `_is_retriable()`, `complete()`/`acomplete()` failover, circuit breaker, `on_fallback` callback
- **`test_format_messages.py` (13):** `_format_messages()` for OpenAI, Anthropic, Gemini — tool role, assistant tool_calls, images
- **`test_batch.py` (6):** `batch()`/`abatch()` history isolation, progress callbacks, partial failure handling

---

## [0.14.0] - 2026-03-11

### Added - AgentObserver Protocol & Observability

#### AgentObserver Protocol

- **`AgentObserver`** base class — class-based alternative to hooks dict for structured observability integrations (Langfuse, OpenTelemetry, Datadog)
- Every callback receives a **`run_id`** for cross-request correlation; tool callbacks also receive a **`call_id`** for parallel tool matching
- **15 lifecycle events** with no-op defaults — subclass and override only the events you need:
  - **Run-level**: `on_run_start`, `on_run_end`, `on_error`
  - **LLM-level**: `on_llm_start`, `on_llm_end`, `on_cache_hit`, `on_usage`, `on_llm_retry`
  - **Tool-level**: `on_tool_start`, `on_tool_end`, `on_tool_error`, `on_tool_chunk`
  - **Iteration-level**: `on_iteration_start`, `on_iteration_end`
  - **Batch-level**: `on_batch_start`, `on_batch_end`
  - **Policy-level**: `on_policy_decision`
  - **Structured output**: `on_structured_validate`
  - **Provider fallback**: `on_provider_fallback`
  - **Memory**: `on_memory_trim`
- **`LoggingObserver`** — built-in observer that emits structured JSON to Python's `logging` module
- **`AgentConfig(observers=[...])`** — register one or more observers per agent
- **`AgentResult.usage`** — aggregated `AgentUsage` available on every result
- **`AgentTrace.parent_run_id`** and **`AgentTrace.metadata`** for nested agent correlation
- **`AgentTrace.to_otel_spans()`** — export trace steps as OpenTelemetry-compatible span dicts

### Fixed

- **OpenAI `max_tokens` rejected by newer models (GPT-5.x, GPT-4.1, o-series, codex)**: OpenAI's newer model families require `max_completion_tokens` instead of the legacy `max_tokens` parameter. Passing `max_tokens` returns a `400 Unsupported parameter` error. The `OpenAIProvider` now auto-detects the model family and sends the correct parameter. Affects `complete()`, `acomplete()`, `stream()`, and `astream()`.
- **Structured output broken by text parser**: When `response_format` is set, the text-based `ToolCallParser` would incorrectly match the LLM's JSON output (e.g. `{"name": "test"}`) as a tool call, preventing structured validation from running. The agent would loop until `max_iterations`. Parser is now skipped when `response_format` is active.
- **Memory trim observer gap**: `memory.add_many()` at the start of `run()`/`arun()`/`astream()` could trigger trimming without notifying observers. Added `_memory_add_many()` helper that fires `on_memory_trim` events.
- **routing_only iteration event mismatch**: `on_iteration_start` fired but `on_iteration_end` was skipped due to early return. Added missing notification in all three code paths.
- **TypeError crash on None provider content**: Providers returning `content=None` crashed `_call_provider` with `TypeError: object of type 'NoneType' has no len()`. Fixed with `content or ""` normalization.
- **Async policy timeout not enforced for sync callbacks**: `_acheck_policy` called sync `confirm_action` directly without timeout protection, potentially blocking the event loop. Now wraps sync callbacks in `loop.run_in_executor()` + `asyncio.wait_for()`.
- **Tool policy bypassed in parallel execution**: `_check_policy` / `_acheck_policy` were missing from `_execute_tools_parallel` and `_aexecute_tools_parallel`. Policy checks now run before every parallel tool execution.
- **`on_llm_retry` fired after backoff sleep**: Moved notification before the sleep to enable real-time logging of retry attempts.
- **Infinite recursion crash with batch + FallbackProvider**: Thread-unsafe `on_fallback` wiring caused stack overflow in `_observer_fallback` during concurrent `batch()` calls. Fixed with `threading.Lock`, reference counting, and `threading.local` for run_id correlation.
- **`on_tool_chunk` observer notification consistency**: Added `if run_id:` guard matching `on_tool_start`/`on_tool_end` pattern.

### Added - Model Registry Update (March 2026)

- **10 new models** across all three major providers (total: 145 models)
- **OpenAI (6 new):** `gpt-5.4` (flagship, 1.05M context, $5/$22.50), `gpt-5.4-pro` ($30/$180), `gpt-5.3-chat-latest`, `gpt-5.3-codex`, `gpt-realtime-1.5`, `gpt-audio-1.5`
- **Anthropic (1 new):** `claude-sonnet-4-6` ($3/$15)
- **Gemini (3 new):** `gemini-3.1-pro-preview` ($2/$12), `gemini-3.1-flash-lite-preview` ($0.10/$0.40), `gemini-3-flash-preview` ($0.50/$3)
- **Price corrections:** GPT-5.2 series updated from $1.25/$10 to $1.75/$14; GPT-5.2-pro from $15/$120 to $21/$168

### Changed

- `AgentResult` extended with `usage` field (aggregated `AgentUsage` copy)
- `AgentConfig` extended with `observers`, `parent_run_id`, `trace_metadata`, `trace_tool_result_chars` fields
- New public exports: `AgentObserver`, `LoggingObserver`
- Model registry grown from 135 to **145 models** with updated March 2026 pricing
- Test suite grown from 880+ to **938 tests** (45 new observer tests, 10 model tests). Further expanded to **1100 tests** in v0.14.1.

---

## [0.13.0] - 2026-02-16

### Added - Structured Output, Observability & Safety

#### Structured Output Parsers

- **`response_format` parameter** on `run()`, `arun()`, `ask()`, `aask()` — pass a Pydantic `BaseModel` class or dict JSON Schema
- Schema instruction injected into system prompt; JSON extracted and validated from LLM response
- **Auto-retry**: validation errors trigger a retry with the error fed back to the LLM
- **`result.parsed`** returns the validated typed object; `result.content` still available as raw string
- New `structured.py` module with `parse_and_validate()`, `extract_json()`, `build_schema_instruction()`

#### Execution Traces (`AgentTrace`)

- **`result.trace`** populated on every `run()` / `arun()` — structured timeline of the entire agent execution
- **`TraceStep`** types: `llm_call`, `tool_selection`, `tool_execution`, `cache_hit`, `error`, `structured_retry`
- Each step captures type, timestamp, duration_ms, input/output summaries, token usage (for LLM steps)
- **`AgentTrace`** container with `.to_dict()`, `.to_json(filepath)`, `.timeline()`, `.filter(type=...)` methods
- New `trace.py` module with `AgentTrace`, `TraceStep`, `StepType`

#### Reasoning Visibility

- **`result.reasoning`** — text the LLM returned alongside its final tool selection (no extra LLM calls)
- **`result.reasoning_history`** — list of reasoning strings, one per agent iteration
- **`step.reasoning`** on `tool_selection` trace steps
- Works with all providers (OpenAI, Anthropic, Gemini, Ollama)

#### Provider Fallback Chain

- **`FallbackProvider`** wraps multiple providers in priority order with automatic failover
- Tries next provider on timeout, 5xx, rate limit (429), or connection errors
- **Circuit breaker**: after N consecutive failures, skip provider for M seconds
- **`on_fallback`** callback fires when a provider is skipped
- **`provider_used`** property reports which provider handled the request
- Supports `complete()`, `acomplete()`, `stream()`, `astream()`

#### Batch Processing

- **`agent.batch(prompts, max_concurrency=5)`** — sync, uses `ThreadPoolExecutor`
- **`agent.abatch(prompts, max_concurrency=10)`** — async, uses `asyncio.Semaphore` + `gather`
- Returns `list[AgentResult]` in same order as input; per-request error isolation
- Respects `response_format`; `on_progress(completed, total)` callback

#### Tool-Pair-Aware Trimming

- **`ConversationMemory._enforce_limits()`** now preserves tool call / tool result pairs
- After trimming, advances past orphaned TOOL results and ASSISTANT tool_use messages
- Conversation always starts at a safe boundary (USER text or SYSTEM message)

#### Tool Policy Engine

- **`ToolPolicy`** with glob-based `allow`, `review`, `deny` rules
- Argument-level `deny_when` conditions (e.g., deny `send_email` when `to` matches `*@external.com`)
- Evaluation order: `deny` → `review` → `allow` → default (review)
- **`AgentConfig(tool_policy=...)`** — evaluated before every tool execution

#### Human-in-the-Loop Approval

- **`AgentConfig(confirm_action=...)`** — sync or async callback `(tool_name, tool_args, reason) -> bool`
- Invoked for tools whose policy decision is `review`
- **`approval_timeout`** with deny-on-timeout default (60s)
- Agent loop: allow → execute, review → callback → execute/deny, deny → error to LLM

### Changed

- `AgentResult` extended with `parsed`, `reasoning`, `reasoning_history`, `trace` fields
- `AgentConfig` extended with `tool_policy`, `confirm_action`, `approval_timeout` fields
- `ConversationMemory` imports `Role` for tool-pair boundary detection
- New public exports: `FallbackProvider`, `ToolPolicy`, `PolicyDecision`, `PolicyResult`, `ResponseFormat`, `AgentTrace`, `TraceStep`

---

## [0.12.1] - 2026-02-16

### Fixed

- **Packaging: missing `selectools.agent` and `selectools.tools` subpackages in published wheel** — `import selectools` raised `ModuleNotFoundError` because `pyproject.toml` had a hardcoded package list that was missing both subpackages. Switched to automatic package discovery (`[tool.setuptools.packages.find]`) so new subpackages are always included.

### Changed

- Replaced explicit `packages = [...]` list in `pyproject.toml` with `[tool.setuptools.packages.find]` for automatic subpackage discovery

---

## [0.12.0] - 2026-02-16

### Added - Hybrid Search (Vector + BM25)

#### BM25 Keyword Search Engine

- **`BM25`** - Pure-Python Okapi BM25 keyword search with zero external dependencies
  - Standard BM25 scoring with configurable `k1` (term frequency saturation) and `b` (length normalisation) parameters
  - Built-in English stop word removal (configurable)
  - Regex-based tokenization with lowercase normalisation
  - Incremental indexing via `add_documents()` or full rebuild via `index_documents()`
  - Metadata filtering support matching the `VectorStore.search()` interface
  - Returns `SearchResult` objects for full compatibility with existing RAG tools

#### Hybrid Searcher

- **`HybridSearcher`** - Combines vector (semantic) and BM25 (keyword) retrieval with score fusion
  - **Reciprocal Rank Fusion (RRF)** - Default fusion strategy; rank-based, no score normalisation needed
  - **Weighted Linear Combination** - Alternative fusion with min-max normalised scores
  - Configurable `vector_weight` and `keyword_weight` for tuning semantic vs keyword balance
  - Automatic deduplication of documents appearing in both result sets
  - Configurable candidate pool sizes (`vector_top_k`, `keyword_top_k`) for fusion quality
  - `add_documents()` forwards to both vector store and BM25 index
  - `index_existing_documents()` for building BM25 index from pre-populated vector stores
  - Metadata filtering applied to both retrievers
- **`FusionMethod`** enum - `RRF` and `WEIGHTED` fusion strategies

#### Agent Integration

- **`HybridSearchTool`** - Pre-built `@tool`-decorated search for agent integration
  - `search_knowledge_base(query)` returns formatted context with source attribution
  - `search(query, filter)` returns structured `SearchResult` list
  - Configurable `score_threshold`, `top_k`, and `include_scores`
  - Drop-in replacement for `RAGTool` with better recall for exact terms, names, and acronyms

### Added - Reranking Models

#### Reranker Protocol

- **`Reranker`** - Abstract base class for all reranker implementations
  - `rerank(query, results, top_k)` re-scores candidates using a cross-encoder model
  - Replaces fusion scores with cross-encoder relevance scores for better precision
  - Returns `SearchResult` objects preserving original document references and metadata

#### Cohere Reranker

- **`CohereReranker`** - Reranker using the Cohere Rerank API v2
  - Uses `cohere.ClientV2.rerank()` with model `rerank-v3.5` (default)
  - Supports `top_n` for server-side result limiting
  - API key via constructor or `COHERE_API_KEY` environment variable
  - Requires `cohere>=5.0.0` (already in `selectools[rag]`)

#### Jina Reranker

- **`JinaReranker`** - Reranker using the Jina AI Rerank API
  - Calls `POST /v1/rerank` via HTTP (uses `requests`, no extra SDK)
  - Default model: `jina-reranker-v2-base-multilingual`
  - Supports `top_n` for server-side result limiting
  - API key via constructor or `JINA_API_KEY` environment variable

#### HybridSearcher Integration

- **`HybridSearcher(reranker=...)`** - Optional reranker applied as a post-fusion step
  - Fused candidates are re-scored before the final `top_k` cut
  - Works with both RRF and weighted fusion strategies
  - Reranker receives the full fused candidate pool for maximum recall

### Added - Advanced Chunking

#### Semantic Chunker

- **`SemanticChunker`** - Splits documents at topic boundaries using embedding similarity
  - Groups consecutive sentences whose embeddings have cosine similarity above a threshold
  - Configurable `similarity_threshold` (0.0-1.0, default 0.75), `min_chunk_sentences`, and `max_chunk_sentences`
  - Uses any `EmbeddingProvider` for computing sentence vectors
  - Pure-Python cosine similarity (zero numpy dependency)
  - Produces chunks aligned with natural topic shifts instead of fixed character windows
  - `split_text()` and `split_documents()` API matching existing chunkers

#### Contextual Chunker

- **`ContextualChunker`** - Wraps any chunker and enriches each chunk with LLM-generated context
  - Inspired by Anthropic's _Contextual Retrieval_ technique
  - For each chunk, generates a 1-2 sentence situating description using the full document as context
  - Prepends the context to the chunk text to improve embedding quality and retrieval relevance
  - Composable: works with `TextSplitter`, `RecursiveTextSplitter`, `SemanticChunker`, or any object with `split_documents()`
  - Configurable `prompt_template`, `model`, `max_document_chars`, and `context_prefix`
  - Stores generated context in `metadata["context"]` for downstream access

### Added - Dynamic Tool Loading

#### Tool Loader

- **`ToolLoader`** - Discover and load `@tool`-decorated functions from Python modules and directories
  - `from_module(module_path)` - Import a dotted module path and collect all `Tool` objects
  - `from_file(file_path)` - Load a single `.py` file and collect all `Tool` objects
  - `from_directory(directory)` - Scan a directory for `.py` files and load tools (optional `recursive` and `exclude`)
  - `reload_module(module_path)` / `reload_file(file_path)` - Hot-reload tools after code changes
  - Skips private files (names starting with `_`) by default

#### Agent Dynamic Tool Management

- **`Agent.add_tool(tool)`** - Add a tool at runtime; rebuilds system prompt
- **`Agent.add_tools(tools)`** - Batch add multiple tools
- **`Agent.remove_tool(tool_name)`** - Remove a tool by name; validates at least one remains
- **`Agent.replace_tool(tool)`** - Swap an existing tool with an updated version (or add if new)
- All methods rebuild the system prompt so the LLM immediately sees the updated tool set

### Added - Response Caching

#### Cache Protocol & Backends

- **`Cache` protocol** - Abstract interface any cache backend must satisfy (`get`, `set`, `delete`, `clear`, `stats`)
- **`InMemoryCache`** - Thread-safe LRU + TTL cache with zero external dependencies
  - `OrderedDict`-based O(1) LRU operations
  - Per-entry TTL with monotonic timestamp expiry
  - Configurable `max_size` (LRU eviction) and `default_ttl`
  - Thread-safe via `threading.Lock`
- **`RedisCache`** - Distributed TTL cache for multi-process deployments
  - Pickle-serialized `(Message, UsageStats)` entries
  - Server-side TTL management
  - Key prefix namespacing (`selectools:`)
  - Requires optional `redis` dependency (`pip install selectools[cache]`)
- **`CacheStats`** - Hit/miss/eviction counters with `hit_rate` property
- **`CacheKeyBuilder`** - Deterministic SHA-256 cache keys from (model, system_prompt, messages, tools, temperature)

#### Agent Integration

- **`AgentConfig(cache=...)`** - Enable caching by passing any `Cache` instance
- Cache checked before every `provider.complete()` / `provider.acomplete()` call
- Cache populated after successful provider responses
- Streaming (`astream`) bypasses cache (non-replayable)
- Cache hits still contribute to usage tracking (`UsageStats` replayed from cache)
- Verbose mode prints `[agent] cache hit -- skipping provider call`

### Changed

- `selectools.rag` now exports `BM25`, `HybridSearcher`, `FusionMethod`, `HybridSearchTool`, `Reranker`, `CohereReranker`, `JinaReranker`, `SemanticChunker`, and `ContextualChunker`
- `selectools.tools` now exports `ToolLoader`
- `HybridSearcher.__init__` accepts new optional `reranker` parameter
- `AgentConfig` extended with `cache: Optional[Cache] = None` field
- `Agent._call_provider()` and `Agent._acall_provider()` now check cache before retry loop
- New exports in `selectools.__init__`: `Cache`, `CacheStats`, `CacheKeyBuilder`, `InMemoryCache`

---

## [0.11.0] - 2026-02-14

### Added - E2E Streaming & Parallel Execution

#### E2E Native Tool Streaming

- **`Agent.astream`** now supports native tool calls from all providers
  - Streams both text chunks and `ToolCall` objects in a unified flow
  - Yields `StreamChunk` for text deltas and full `ToolCall` objects when ready
  - Robust fallback: gracefully degrades to non-streaming `acomplete` when provider lacks streaming
- **Provider `astream` protocol** - All providers now implement `astream` returning `Union[str, ToolCall]`
  - `OpenAIProvider.astream` - Streams tool call deltas and text content
  - `AnthropicProvider.astream` - Streams tool call deltas and text content
  - `GeminiProvider.astream` - Streams tool call deltas and text content
- **`StreamChunk`** updated to support optional `tool_calls` field

#### Parallel Tool Execution

- **Concurrent tool execution** when LLM requests multiple tools in a single response
  - `asyncio.gather()` for async execution (`arun`, `astream`)
  - `ThreadPoolExecutor` for sync execution (`run`)
  - Enabled by default: `AgentConfig(parallel_tool_execution=True)`
  - Results appended to history in original request order (ordering preserved)
  - Per-tool error isolation: one tool failure doesn't block others
  - All hooks (`on_tool_start`, `on_tool_end`, `on_tool_error`) fire for every tool
- **New config option**: `parallel_tool_execution: bool = True`

#### Full Type Safety

- 0 mypy errors across all 48 source files and 32 test files
- `from __future__ import annotations` added to all test files
- Full parameter and return type annotations on all test helpers, fixtures, and mock providers
- `disallow_untyped_defs = true` enforced and verified

### Changed

- `StreamChunk` dataclass extended with optional `tool_calls` field
- Provider `astream` methods now yield `Union[str, ToolCall]` instead of just `str`
- `Agent.astream` rewritten with tool delta accumulation and native tool call yielding
- `GeminiProvider` improved null-safety for `Optional[Content].parts` access

### Fixed

- Fixed mypy union-attr errors in `gemini_provider.py` for nullable `Content.parts`
- Fixed `ToolCall.tool_name` type narrowing from `Union[str, Any, None]` to `str`
- Black and isort applied across entire codebase

---

## [0.10.0] - 2026-02-13

### Added - Critical Architecture

#### Native Function Calling

- **All providers** now use native tool calling APIs instead of regex parsing
  - `OpenAIProvider` - Uses OpenAI function calling with `tools` parameter
  - `AnthropicProvider` - Uses Anthropic tool use blocks
  - `GeminiProvider` - Uses Gemini function calling declarations
- **`Message.tool_calls`** field carries native `ToolCall` objects from provider responses
- **Fallback**: Regex-based `ToolCallParser` still used when provider returns text-only responses
- `complete()` and `acomplete()` now accept `tools` parameter for native tool schemas

#### Context Propagation (Async)

- **`contextvars.copy_context()`** ensures tracing/auth context flows into async tool execution
- Safe propagation across `asyncio.gather()` and executor boundaries

#### Routing Mode

- **`AgentConfig(routing_only=True)`** returns tool selection without executing
- Ideal for classification, intent routing, and tool selection pipelines
- Returns `AgentResult` with `tool_name` and `tool_args` immediately

### Breaking Changes

- **`Provider.complete()` return type**: Returns `tuple[Message, UsageStats]` instead of `tuple[str, UsageStats]`
  - **Migration**: If you call `provider.complete()` directly, the first element is now a `Message` object; use `message.content` to get the text
  - **Migration**: If you have a custom `Provider` implementation, update `complete()` and `acomplete()` to return `(Message, UsageStats)` instead of `(str, UsageStats)`
  - **No impact** if you only use `Agent.run()` / `Agent.arun()` (the agent handles this internally)
- **`Provider.complete()` signature**: New `tools: Optional[List[Tool]] = None` parameter added
  - **No impact** for existing code (parameter has a default value)

### Changed

- Agent loop checks `response_msg.tool_calls` before falling back to text-based parser
- Default Anthropic model updated from retired `claude-3-5-sonnet-20241022` to `claude-sonnet-4-5-20250514`

### Fixed

- Fixed 75+ test failures from `FakeProvider` stubs not conforming to updated `Provider` protocol
- Fixed API key isolation tests using `monkeypatch.delenv` + `unittest.mock.patch`
- Fixed E2E test failures from retired Anthropic model
- Updated model registry assertion counts to match current 135+ model entries

---

## [0.9.0] - 2026-02-12

### Added - Core Capabilities & Reliability

#### Custom System Prompt

- **`AgentConfig(system_prompt=...)`** - Inject domain-specific instructions directly
  - Replaces the default built-in system prompt when provided
  - No more workarounds prepending instructions to user messages

#### Structured AgentResult

- **`Agent.run()` now returns `AgentResult`** instead of `Message`
  - `result.message` - The final `Message` object
  - `result.tool_name` - Name of the last tool called (or `None`)
  - `result.tool_args` - Parameters passed to the last tool call
  - `result.iterations` - Number of loop iterations used
  - `result.tool_calls` - Ordered list of all `ToolCall` objects made during the run
  - **Backward-compatible**: `result.content` and `result.role` properties still work

#### Reusable Agent Instances

- **`Agent.reset()`** - Clears history, usage stats, analytics, and memory for clean reuse
  - No more creating fresh Agent/Provider/Config per request

### Breaking Changes

- **`run()` / `arun()` return type**: Returns `AgentResult` instead of `Message`
  - **Migration**: Code using `result.content` continues to work unchanged (backward-compat property)
  - **Migration**: Code doing `isinstance(result, Message)` or type-checking the return must update to `AgentResult`
  - **Migration**: Code passing the result directly as a `Message` should use `result.message` instead

---

## [0.8.0] - 2025-12-10

### Added - RAG & Embeddings 🎉

> **Production Polish Update:** Added 3 comprehensive examples, 200+ new tests, complete troubleshooting guide, and v0.9.0+ roadmap.
>
> **QA Complete:** All examples tested with real API calls. Fixed 10 bugs in examples and enhanced RAGAgent API with `score_threshold` and `agent_config` parameters.

#### Embedding Providers (4 providers, 10 models)

- **OpenAIEmbeddingProvider** - 3 models
  - `text-embedding-3-small` ($0.02/1M tokens)
  - `text-embedding-3-large` ($0.13/1M tokens)
  - `text-embedding-ada-002` ($0.10/1M tokens)
- **AnthropicEmbeddingProvider** - 2 models via Voyage AI
  - `voyage-3` ($0.06/1M tokens)
  - `voyage-3-lite` ($0.02/1M tokens)
- **GeminiEmbeddingProvider** - 2 models (FREE!)
  - `text-embedding-001`
  - `text-embedding-004`
- **CohereEmbeddingProvider** - 3 models
  - `embed-english-v3.0` ($0.10/1M tokens)
  - `embed-multilingual-v3.0` ($0.10/1M tokens)
  - `embed-english-light-v3.0` ($0.10/1M tokens)

#### Vector Stores (4 backends)

- **InMemoryVectorStore** - Fast NumPy-based, zero dependencies
- **SQLiteVectorStore** - Persistent local storage
- **ChromaVectorStore** - Advanced vector database
- **PineconeVectorStore** - Cloud-hosted production-ready

#### Document Processing

- **DocumentLoader** - Load from text, files, directories, PDFs
  - `from_text()` - Create documents from strings
  - `from_file()` - Load single files (.txt, .md)
  - `from_directory()` - Load entire directories with glob patterns
  - `from_pdf()` - Extract text from PDF files (requires pypdf)
- **TextSplitter** - Fixed-size chunking with overlap
- **RecursiveTextSplitter** - Smart chunking respecting natural boundaries
  - Splits on paragraphs, sentences, spaces in order
  - Preserves document metadata

#### RAG Tools

- **RAGTool** - Pre-built tool for knowledge base search
  - Automatically embeds queries
  - Searches vector store for relevant documents
  - Returns formatted context with sources and scores
- **SemanticSearchTool** - Pure semantic search without LLM
- **RAGAgent** - High-level API for creating RAG agents
  - `from_documents()` - Create from document list
  - `from_directory()` - Create from document directory
  - `from_files()` - Create from specific files

#### Cost Tracking for Embeddings

- **Extended UsageStats**
  - New `embedding_tokens` field
  - New `embedding_cost_usd` field
- **Updated AgentUsage**
  - Tracks total embedding tokens
  - Tracks total embedding costs
  - Displays in usage summary with LLM vs embedding breakdown
- **New pricing function**
  - `calculate_embedding_cost(model, tokens)` for easy cost estimation

#### Model Registry Additions

- **New Cohere class** with 3 embedding models
- **Embedding subclasses** for OpenAI, Anthropic, Gemini
- Updated model count: 130 total models (120 chat + 10 embedding)

### Changed

- **pyproject.toml** version bumped to `0.8.0`
- **Core dependency added**: `numpy>=1.24.0,<3.0.0` (required for vector operations)
- **Package structure**: Added `selectools.embeddings`, `selectools.rag`, `selectools.rag.stores`

### Optional Dependencies

New `[rag]` extra for full RAG support:

```bash
pip install selectools[rag]
```

Includes:

- `chromadb>=0.4.0` - ChromaDB vector store
- `pinecone-client>=3.0.0` - Pinecone cloud vector store
- `voyageai>=0.2.0` - Voyage AI embeddings
- `cohere>=5.0.0` - Cohere embeddings
- `pypdf>=4.0.0` - PDF document loading

### Documentation

- Added comprehensive RAG section to README
- New example: `examples/rag_basic_demo.py`
- Basic integration tests: `tests/test_rag_basic.py`
- Updated installation instructions

### Examples & Testing (Production Polish)

#### New Examples (3 comprehensive demos)

- **`examples/rag_advanced_demo.py`** - Advanced RAG workflow
  - PDFs and persistent SQLite storage
  - Custom RecursiveTextSplitter with multiple separators
  - Metadata filtering and enrichment
  - Cost tracking and analytics integration
  - 8-step guided demonstration
- **`examples/semantic_search_demo.py`** - Pure semantic search
  - Compare OpenAI vs Gemini embedding providers
  - Analyze similarity scores and performance
  - Metadata filtering demonstrations
  - Cost comparison tables
  - Search quality recommendations
- **`examples/rag_multi_provider_demo.py`** - Configuration comparison
  - Embedding provider benchmarks
  - Vector store performance (memory vs SQLite)
  - Chunk size impact analysis
  - Top-K parameter tuning guide
  - Comprehensive cost breakdown

#### New Test Suite (200+ tests, 7 test files)

- **`tests/test_embedding_providers.py`** - 40+ tests for all 4 embedding providers
  - Mocked API responses to avoid costs
  - Batch operations, error handling, retry logic
  - Interface consistency across providers
- **`tests/test_vector_stores_crud.py`** - 60+ tests for all 4 vector stores
  - CRUD operations (add, search, delete, clear)
  - Cosine similarity accuracy
  - Metadata filtering
  - Top-K limiting
- **`tests/test_document_loaders.py`** - 25+ tests for document loading
  - Text, file, directory, PDF loading
  - Metadata preservation
  - Unicode and encoding support
  - Error handling
- **`tests/test_text_chunking.py`** - 35+ tests for chunking strategies
  - TextSplitter with overlap
  - RecursiveTextSplitter with hierarchical splitting
  - Edge cases (empty, long, Unicode text)
  - Metadata preservation
- **`tests/test_sqlite_integration.py`** - 20+ tests for persistence
  - Database reconnection
  - Concurrent access patterns
  - Search quality after persistence
  - Performance benchmarks
- **`tests/test_rag_workflow.py`** - 25+ tests for complete RAG pipeline
  - Load → Chunk → Embed → Store → Search workflows
  - RAGAgent creation from documents and directories
  - Cost tracking integration
  - Analytics integration
- **`tests/test_vector_store_compatibility.py`** - 30+ tests for consistency
  - All vector stores behave identically
  - Same data yields same results
  - API compatibility verification
  - Performance characteristics

#### Documentation Additions

- **Troubleshooting Guide** - 8 common issues with solutions
  - ImportError handling
  - Vector store setup (ChromaDB, Pinecone)
  - Embedding provider configuration
  - PDF loading errors
  - Memory optimization tips
  - Performance tuning
  - Cost management
  - Search relevance tuning
- **Future Roadmap** — See [ROADMAP.md](https://github.com/johnnichev/selectools/blob/main/ROADMAP.md) for current plans

### Fixed

#### Example Bug Fixes (QA Phase)

- **`semantic_search_demo.py`** (2 fixes)
  - Fixed `TypeError` when calling `semantic_search()` - changed to use `.search()` method
  - Fixed result object access - changed from dict access to object attributes (`result.score`, `result.document.text`)

- **`rag_advanced_demo.py`** (6 fixes)
  - Fixed `agent.run()` signature - now properly passes `List[Message]` instead of raw strings
  - Fixed response handling - now extracts `.content` from returned `Message` object
  - Fixed `AttributeError` - changed `agent.get_usage()` to `agent.usage` attribute
  - Fixed cost calculation - compute LLM cost as `total_cost_usd - total_embedding_cost_usd`
  - Fixed 3 usage display references (2 in queries, 1 in code example)

- **`rag_basic_demo.py`** (2 fixes)
  - Fixed `agent.run()` signature - now properly passes `List[Message]`
  - Fixed response handling - now extracts `.content` from returned `Message`

#### API Enhancements

- **`RAGAgent` factory methods** - Added missing parameters to all 3 methods:
  - Added `score_threshold: float = 0.0` parameter for similarity filtering
  - Added `agent_config: Optional[AgentConfig] = None` parameter for custom agent configuration
  - Applies to: `from_documents()`, `from_directory()`, `from_files()`

- **`tests/test_rag_basic.py`** - Fixed test assertion
  - Changed assertion to check `Tool` object properties instead of attempting to call it

### Migration Notes

All changes are **backward compatible**. Existing code continues to work without modification. RAG features are opt-in and require NumPy (automatically installed).

To use RAG features:

```python
from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.rag import RAGAgent, VectorStore

embedder = OpenAIEmbeddingProvider()
vector_store = VectorStore.create("memory", embedder=embedder)

agent = RAGAgent.from_directory(
    directory="./docs",
    provider=OpenAIProvider(),
    vector_store=vector_store
)
```

---

## [0.7.0] - 2025-12-10

### Added

- **Model Registry System** - Single source of truth for all 120 models
  - New `models.py` module with `ModelInfo` dataclass containing complete model metadata
  - Typed model constants for IDE autocomplete: `OpenAI.GPT_4O`, `Anthropic.SONNET_4_5`, etc.
  - 64 OpenAI models (GPT-5, GPT-4o, o-series, GPT-4, GPT-3.5)
  - 18 Anthropic models (Claude 4.5, 4.1, 4, 3.7, 3.5, 3)
  - 25 Gemini models (Gemini 3, 2.5, 2.0, 1.5, 1.0, Gemma)
  - 13 Ollama models (Llama, Mistral, Phi, etc.)
- **Rich Model Metadata** - Each model includes:
  - Pricing (prompt/completion costs per 1M tokens)
  - Context window size
  - Maximum output tokens
  - Model type (chat, audio, multimodal)
  - Provider name
- **New Public API** exports:
  - `models` module
  - `ModelInfo` dataclass
  - `ALL_MODELS` list
  - `MODELS_BY_ID` dict
  - `OpenAI`, `Anthropic`, `Gemini`, `Ollama` classes
- **Updated Documentation**
  - New "Model Selection with Autocomplete" section in README
  - All code examples updated to use typed constants
  - 12 example files migrated to demonstrate new pattern

### Changed

- **Pricing Module Refactored** - Now derives from `models.py` instead of hardcoded dict
- **All Provider Defaults** - Use typed constants instead of hardcoded strings
- **Backward Compatible** - Old code using `PRICING` dict still works
- Updated OpenAI pricing with 70+ models including GPT-5, o3-pro, latest GPT-4o variants
- Updated Anthropic pricing with Claude 4.5, 4.1, 4 series
- Updated Gemini pricing with Gemini 3, 2.5, 2.0 series

### Fixed

- Test suite updated to handle frozen dataclass immutability correctly

## [0.6.1] - 2025-12-09

### Added

- **Streaming Tool Results** - Tools can now yield results progressively
  - Support for `Generator[str, None, None]` return types (sync)
  - Support for `AsyncGenerator[str, None]` return types (async)
  - Real-time chunk callbacks via `on_tool_chunk` hook
  - Streaming metrics in analytics (chunk counts, streaming calls)
- **Toolbox Streaming Tools**
  - `read_file_stream` - Stream file content line by line
  - `process_csv_stream` - Stream CSV content row by row
- Examples: `streaming_tools_demo.py` with 5 comprehensive scenarios

### Changed

- Analytics now track `total_chunks` and `streaming_calls` for streaming tools
- Tool execution supports progressive result delivery

## [0.6.0] - 2025-12-08

### Added

- **Local Model Support** - Ollama provider for local LLM execution
  - Zero cost (all Ollama models priced at $0.00)
  - Privacy-preserving (no data sent to cloud)
  - OpenAI-compatible API
  - Support for llama3.2, mistral, codellama, phi, qwen, etc.
- **Tool Usage Analytics** - Comprehensive metrics tracking
  - Call frequency, success/failure rates, execution duration
  - Parameter usage patterns, cost attribution per tool
  - Export to JSON/CSV with `export_to_json()` and `export_to_csv()`
  - Enable with `AgentConfig(enable_analytics=True)`
- Examples: `ollama_demo.py`, `tool_analytics_demo.py`

### Changed

- Pricing module now includes 13 Ollama models (all free)

## [0.5.2] - 2025-12-07

### Added

- **Tool Validation at Registration** - Validates tool definitions when created
  - Name validation (valid Python identifier, 1-64 chars)
  - Description validation (10-1024 chars, required)
  - Parameter validation (names, types, required fields, duplicates)
  - Signature mismatch detection
- **Observability Hooks** - 10 lifecycle callbacks for monitoring
  - `on_agent_start`, `on_agent_end`
  - `on_tool_start`, `on_tool_end`, `on_tool_error`
  - `on_llm_start`, `on_llm_end`, `on_llm_error`
  - `on_error`, `on_max_iterations`
- Example: `v0_5_2_demo.py` with 8 scenarios

### Changed

- Improved error messages with validation details
- Tools now validate at creation time, not runtime

## [0.5.1] - 2025-12-06

### Added

- **Pre-built Tool Library** - 27 production-ready tools in 5 categories
  - **File Tools** (7): read_file, write_file, list_directory, etc.
  - **Web Tools** (4): fetch_url, search_web, extract_html_text, etc.
  - **Data Tools** (8): parse_json, parse_csv, calculate, etc.
  - **DateTime Tools** (3): get_current_time, parse_datetime, format_datetime
  - **Text Tools** (5): count_words, find_pattern, replace_text, etc.
- **ToolRegistry** - Manage and filter tools by category
- Example: `toolbox_demo.py`

### Changed

- All toolbox tools include comprehensive docstrings and examples

## [0.5.0] - 2025-12-05

### Added

- **Better Error Messages** - PyTorch-style helpful errors
  - Custom exceptions: `ToolValidationError`, `ToolExecutionError`, `ProviderConfigurationError`, `MemoryLimitExceededError`
  - Fuzzy matching for parameter typos with suggestions
  - Context-aware error messages with fix suggestions
- **Cost Tracking** - Automatic token counting and cost estimation
  - `UsageStats` dataclass with token counts and costs
  - `AgentUsage` for aggregated multi-turn usage
  - Configurable cost warnings via `cost_warning_threshold`
  - Pricing for 120+ models across OpenAI, Anthropic, Gemini, Ollama
- **Gemini SDK Migration** - Updated to `google-genai` v1.0+
- Example: `cost_tracking_demo.py`

### Changed

- All providers now return `(content, usage_stats)` tuples from `complete()` methods
- Streaming methods only yield content (no usage stats during streaming)

## [0.4.0] - 2025-11-15

### Added

- **Conversation Memory** - Multi-turn context management
  - `ConversationMemory` class with configurable max_messages
  - Automatic context injection for all turns
  - FIFO eviction when memory limit reached
- **Async Support** - Full async/await support
  - `Agent.arun()` for async execution
  - Async tool functions supported
  - Async providers (`acomplete`, `astream`)
- **Real Provider Integrations**
  - `AnthropicProvider` - Full Anthropic SDK integration
  - `GeminiProvider` - Full Google Gemini SDK integration
- Example: `async_agent_demo.py`, `conversation_memory_demo.py`

### Changed

- All providers support both sync and async operations
- Improved streaming support across all providers

## [0.3.0] - 2025-11-01

### Added

- Initial public release
- OpenAI provider integration
- Basic tool-calling functionality
- Simple agent implementation

---

## Migration Guide: v0.8.0 → v0.15.0

This section covers all breaking changes for consumers upgrading from v0.8.0.

### Summary of Breaking Changes

| Version | Change                                                                               | Impact                                  |
| ------- | ------------------------------------------------------------------------------------ | --------------------------------------- |
| v0.9.0  | `run()` / `arun()` return `AgentResult` instead of `Message`                         | Low (backward-compat properties)        |
| v0.10.0 | `Provider.complete()` returns `(Message, UsageStats)` instead of `(str, UsageStats)` | Low (only if calling provider directly) |
| v0.10.0 | `Provider.complete()` signature adds `tools` parameter                               | None (has default)                      |

### Step-by-Step Migration

#### 1. `Agent.run()` return type (v0.9.0)

```python
# v0.8.0 — run() returned a Message
result = agent.run([Message(role=Role.USER, content="Hello")])
print(result.content)  # str — the response text

# v0.9.0+ — run() returns AgentResult
result = agent.run([Message(role=Role.USER, content="Hello")])
print(result.content)      # STILL WORKS — backward-compat property
print(result.message)      # NEW — the underlying Message object
print(result.tool_name)    # NEW — last tool called (or None)
print(result.tool_args)    # NEW — last tool parameters
print(result.iterations)   # NEW — loop iteration count
print(result.tool_calls)   # NEW — all ToolCall objects
```

**Most code needs zero changes** because `result.content` and `result.role` are preserved as properties on `AgentResult`.

**Code that breaks:**

```python
# ❌ Type checks against Message
if isinstance(result, Message):  # False now — it's AgentResult

# ❌ Passing result where Message is expected
some_function_expecting_message(result)  # Pass result.message instead
```

#### 2. `Provider.complete()` return type (v0.10.0)

Only relevant if you call `provider.complete()` directly or have a custom `Provider`:

```python
# v0.8.0 — complete() returned (str, UsageStats)
text, usage = provider.complete(model="gpt-4o", ...)
print(text)  # str

# v0.10.0+ — complete() returns (Message, UsageStats)
message, usage = provider.complete(model="gpt-4o", ...)
print(message.content)      # str — the response text
print(message.tool_calls)   # List[ToolCall] — native tool calls
```

**No impact** if you only use `Agent.run()` / `Agent.arun()`.

#### 3. New features (all backward-compatible, opt-in)

```python
from selectools import Agent, AgentConfig, InMemoryCache
from selectools.guardrails import GuardrailsPipeline, PIIGuardrail, TopicGuardrail
from selectools.audit import AuditLogger, PrivacyLevel

config = AgentConfig(
    # NEW in v0.9.0 — custom system prompt (replaces message-prepending hacks)
    system_prompt="You are a routing assistant.",

    # NEW in v0.10.0 — return tool selection without executing
    routing_only=True,

    # NEW in v0.11.0 — parallel tool execution (on by default)
    parallel_tool_execution=True,

    # NEW in v0.12.0 — response caching
    cache=InMemoryCache(max_size=1000, default_ttl=300),

    # NEW in v0.13.0 — tool safety policies
    tool_policy=ToolPolicy(allow=["search_*"], deny=["delete_*"]),

    # NEW in v0.13.0 — human-in-the-loop approval
    confirm_action=lambda name, args, reason: True,

    # NEW in v0.14.0 — structured observability
    observers=[AuditLogger(log_dir="./audit", privacy=PrivacyLevel.KEYS_ONLY)],

    # NEW in v0.15.0 — input/output guardrails
    guardrails=GuardrailsPipeline(
        input=[PIIGuardrail(action="rewrite"), TopicGuardrail(deny=["politics"])],
    ),

    # NEW in v0.15.0 — tool output screening
    screen_tool_output=True,

    # NEW in v0.15.0 — coherence checking
    coherence_check=True,
    coherence_model="gpt-4o-mini",
)

agent = Agent(tools=[...], provider=provider, config=config)

# NEW in v0.9.0 — reuse agent between requests
agent.reset()

# NEW in v0.13.0 — structured output
from pydantic import BaseModel
class Intent(BaseModel):
    intent: str
    confidence: float
result = agent.ask("Cancel my sub", response_format=Intent)
print(result.parsed)  # Intent(intent="cancel", confidence=0.95)

# NEW in v0.13.0 — execution traces
print(result.trace.timeline())

# NEW in v0.13.0 — batch processing
results = agent.batch(["msg1", "msg2", "msg3"], max_concurrency=5)

# NEW in v0.13.0 — provider fallback
from selectools import FallbackProvider
fallback = FallbackProvider([primary_provider, backup_provider])

# NEW in v0.12.0 — dynamic tool loading
from selectools.tools import ToolLoader
tools = ToolLoader.from_directory("./plugins", recursive=True)
agent.add_tools(tools)
agent.remove_tool("old_tool")
agent.replace_tool(updated_tool)

# NEW in v0.12.0 — hybrid search (BM25 + vector)
from selectools.rag import HybridSearcher, FusionMethod, CohereReranker
searcher = HybridSearcher(
    vector_store=store,
    fusion=FusionMethod.RRF,
    reranker=CohereReranker(),
)

# NEW in v0.12.0 — advanced chunking
from selectools.rag import SemanticChunker, ContextualChunker
semantic = SemanticChunker(embedder=embedder, similarity_threshold=0.75)
contextual = ContextualChunker(base_chunker=semantic, provider=provider)
```

---

## Release Links

- [0.15.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.15.0)
- [0.14.1 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.14.1)
- [0.14.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.14.0)
- [0.13.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.13.0)
- [0.12.1 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.12.1)
- [0.12.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.12.0)
- [0.11.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.11.0)
- [0.10.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.10.0)
- [0.9.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.9.0)
- [0.8.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.8.0)
- [0.7.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.7.0)
- [0.6.1 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.6.1)
- [0.6.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.6.0)

For detailed migration guides and breaking changes, see the [documentation](https://github.com/johnnichev/selectools).
