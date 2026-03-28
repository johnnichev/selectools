# Selectools — Agent Instructions

This file provides context for AI coding agents (Claude, Cursor, Copilot, etc.) working on the selectools codebase.

## Project Overview

**Selectools** is a production-ready Python library for building AI agents with tool calling, RAG, and hybrid search. It supports OpenAI, Anthropic, Gemini, and Ollama providers.

- **Version**: Check `src/selectools/__init__.py` for current version
- **Python**: 3.9+ (CI runs 3.9, 3.10, 3.11, 3.12, 3.13)
- **Package manager**: pip + setuptools
- **Source layout**: `src/selectools/` (src-layout)
- **Tests**: `tests/` (pytest)
- **Docs**: `docs/` (MkDocs Material, deployed to GitHub Pages)
- **PyPI**: https://pypi.org/project/selectools/
- **Docs site**: https://johnnichev.github.io/selectools

## Codebase Structure

```
src/selectools/
├── __init__.py              # Public API exports + __version__
├── agent/
│   ├── core.py              # Main agent loop (run, arun, astream, batch)
│   ├── config.py            # AgentConfig dataclass
│   ├── _tool_executor.py    # Tool execution pipeline (policy, coherence, timeouts)
│   ├── _provider_caller.py  # LLM provider calls (cache, retry, streaming)
│   ├── _lifecycle.py        # Observer notification, fallback wiring
│   └── _memory_manager.py   # Memory operations, session save, entity/KG extraction
├── providers/
│   ├── base.py              # Provider protocol
│   ├── openai_provider.py   # OpenAI adapter (max_completion_tokens handling)
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   ├── ollama_provider.py
│   ├── fallback.py          # FallbackProvider with circuit breaker
│   ├── _openai_compat.py    # Shared OpenAI/Ollama base (Template Method)
│   └── stubs.py             # LocalProvider for testing without API keys
├── tools/
│   ├── base.py              # Tool class
│   ├── decorators.py        # @tool decorator
│   ├── loader.py            # ToolLoader for dynamic loading
│   └── registry.py          # Tool registry
├── toolbox/                 # 24 pre-built tools (file, web, data, datetime, text)
├── guardrails/              # Input/output validation pipeline
│   ├── base.py              # Guardrail protocol, GuardrailAction, GuardrailResult
│   ├── pipeline.py          # GuardrailsPipeline
│   ├── pii.py, topic.py, toxicity.py, format.py, length.py
├── rag/                     # RAG pipeline (loaders, chunking, hybrid search)
│   ├── stores/              # Vector stores (memory, sqlite, chroma, pinecone)
│   ├── hybrid.py, bm25.py, reranker.py, chunking.py
├── embeddings/              # Embedding providers (openai, anthropic, gemini, cohere)
├── memory.py                # ConversationMemory with sliding window + tool-pair trimming + summarize-on-trim
├── sessions.py              # SessionStore protocol + JSON/SQLite/Redis backends
├── entity_memory.py         # EntityMemory (LLM-based entity extraction)
├── knowledge_graph.py       # KnowledgeGraphMemory (triple extraction + storage)
├── knowledge.py             # KnowledgeMemory + KnowledgeEntry + KnowledgeStore + File/SQLite backends
├── knowledge_store_redis.py # RedisKnowledgeStore (optional dep: redis)
├── knowledge_store_supabase.py # SupabaseKnowledgeStore (optional dep: supabase)
├── cancellation.py          # CancellationToken (thread-safe cooperative cancellation)
├── token_estimation.py      # estimate_tokens(), estimate_run_tokens(), TokenEstimate
├── models.py                # 152 model registry with pricing (single source of truth)
├── pricing.py               # Derives pricing from models.py
├── usage.py                 # Token + cost tracking
├── trace.py                 # AgentTrace, TraceStep (27 step types — see list below)
├── observer.py              # AgentObserver (45 sync events) + AsyncAgentObserver (42 async events) + LoggingObserver + SimpleStepObserver
├── policy.py                # ToolPolicy (allow/review/deny rules)
├── parser.py                # ToolCallParser (JSON extraction from LLM responses)
├── prompt.py                # PromptBuilder (system prompt generation)
├── structured.py            # Structured output (Pydantic/JSON Schema validation)
├── audit.py                 # AuditLogger (JSONL with privacy levels)
├── security.py              # Tool output screening (prompt injection detection)
├── coherence.py             # Coherence checking (LLM-based intent verification)
├── cache.py                 # InMemoryCache (LRU + TTL)
├── cache_redis.py           # RedisCache
├── exceptions.py            # SelectoolsError hierarchy
├── analytics.py             # AgentAnalytics
├── types.py                 # Core types (Message, Role, ToolCall, AgentResult)
├── env.py                   # Environment variable helpers
└── evals/                   # Built-in eval framework (39 evaluators)
    ├── types.py             # TestCase, CaseResult, CaseVerdict, EvalFailure
    ├── evaluators.py        # 22 deterministic evaluators
    ├── llm_evaluators.py    # 17 LLM-as-judge evaluators
    ├── suite.py             # EvalSuite orchestration
    ├── report.py            # EvalReport with stats and export
    ├── dataset.py           # DatasetLoader (JSON/YAML)
    ├── regression.py        # BaselineStore, RegressionResult
    ├── pairwise.py          # PairwiseEval A/B comparison
    ├── generator.py         # Synthetic test case generator
    ├── snapshot.py          # SnapshotStore (Jest-style)
    ├── badge.py             # SVG badge generator
    ├── serve.py             # Live eval dashboard
    ├── html.py              # Interactive HTML report
    ├── junit.py             # JUnit XML for CI
    └── __main__.py          # CLI: python -m selectools.evals
├── orchestration/           # Multi-agent orchestration (v0.18.0)
│   ├── __init__.py          # Public exports (30+ symbols)
│   ├── state.py             # GraphState, MergePolicy, ContextMode, InterruptRequest, Scatter
│   ├── node.py              # GraphNode, ParallelGroupNode, SubgraphNode
│   ├── graph.py             # AgentGraph engine (~700 lines)
│   ├── checkpoint.py        # CheckpointStore protocol + 3 backends
│   └── supervisor.py        # SupervisorAgent with 4 strategies
├── pipeline.py              # Pipeline, Step, StepResult, @step, parallel(), branch() — composable pipelines (v0.18.0)

tests/                       # 2529 tests (unit, integration, regression, E2E)
├── agent/                   # Agent core tests
├── providers/               # Provider-specific tests
├── rag/                     # RAG pipeline tests
├── tools/                   # Tool system tests
├── integration/             # Cross-module integration tests
├── core/                    # Framework-level tests
└── test_*.py                # Module-level unit tests

examples/                    # 61 numbered example scripts (01-61)
notebooks/getting_started.ipynb  # Interactive getting-started guide

docs/                        # MkDocs Material documentation
├── index.md                 # Landing page
├── QUICKSTART.md            # 5-minute quickstart
├── ARCHITECTURE.md          # System architecture
├── modules/                 # 24 module-specific docs
├── decisions/               # Architecture Decision Records (ADRs)
└── stylesheets/extra.css    # Custom theme CSS
```

## Development Commands

```bash
# Tests
pytest tests/ -x -q                    # All tests
pytest tests/ -k "not e2e" -x -q       # Skip E2E (no API keys needed)
pytest tests/providers/test_models.py   # Specific test file

# Formatting & linting
black src/ tests/ --line-length=100
isort src/ tests/ --profile=black --line-length=100
flake8 src/
mypy src/

# Docs
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs serve   # Local preview
mkdocs build                                          # Build static site

# Build & publish
python3 -m build
python3 -m twine upload dist/*
```

## Key Conventions

### Code Style
- Line length: 100 characters
- Formatter: Black + isort
- Type hints: Required on all public APIs (mypy enforced)
- No `any` types — always explicit types
- No comments explaining what code does — only non-obvious intent

### File Naming
- Source modules: `snake_case.py`
- Test files: `test_<module>.py`
- Example files: `NN_descriptive_name.py` (zero-padded number)
- Doc files: `UPPER_CASE.md` for modules, `lower_case.md` for guides

### Provider Pattern
Every provider implements the `Provider` protocol from `providers/base.py`:
- `complete()` / `acomplete()` — synchronous/async completion
- `stream()` / `astream()` — synchronous/async streaming (yields `str | ToolCall`)
- `_format_messages()` — converts internal `Message` objects to provider-specific format
- Must pass `tools` parameter to all methods (streaming included)
- Must handle `ToolCall` objects properly (not stringify them)

### Agent Core Pattern
- `run()` / `arun()` — main sync/async execution
- `astream()` — async streaming with tool calls
- `batch()` / `abatch()` — concurrent multi-prompt execution
- All methods produce `AgentResult` with `.content`, `.trace`, `.reasoning`, `.usage`

### Testing Pattern
- Unit tests mock providers — never call real APIs
- Use `RecordingProvider` to verify exact arguments passed to provider methods
- E2E tests use `@pytest.mark.e2e` and are skipped in CI
- Regression tests go in `tests/agent/test_regression.py`
- Every bug fix gets a dedicated regression test
- Test model counts when adding/removing models

## Feature Development Checklist

When implementing a new feature, ALWAYS complete ALL of these steps:

### 1. Cross-Feature Impact Analysis
- [ ] How does this feature interact with existing features?
- [ ] Does it need integration in `agent/core.py` (the main loop)?
- [ ] Does `AgentConfig` need new fields?
- [ ] Does `__init__.py` need new exports?
- [ ] Does `AgentTrace` need new `StepType` values?
- [ ] Does the `AgentObserver` protocol need new events?
- [ ] Does `AgentResult` need new fields?

### 2. Implementation
- [ ] Create new module(s) in `src/selectools/`
- [ ] Integrate with `agent/core.py` if it affects the agent loop
- [ ] Update `agent/config.py` with new config options
- [ ] Update `src/selectools/__init__.py` with new public exports
- [ ] Run `black` and `isort` on all new/modified files
- [ ] Run `mypy src/` to check types
- [ ] Run `flake8 src/` to check linting

### 3. Testing (CRITICAL — bugs found in production are unacceptable)
- [ ] Unit tests for the new module (`tests/test_<module>.py`)
- [ ] Integration tests if it touches the agent loop
- [ ] Regression tests for any edge cases discovered
- [ ] Update model count tests if models changed
- [ ] Run full suite: `pytest tests/ -x -q` — ALL must pass
- [ ] Verify no tests were broken by the change

### 4. Documentation Updates (ALL of these, every time)
- [ ] **Module doc**: Create or update `docs/modules/<MODULE>.md`
- [ ] **Architecture doc**: Update `docs/ARCHITECTURE.md` if it adds a new component
- [ ] **Quickstart**: Update `docs/QUICKSTART.md` if it's user-facing
- [ ] **docs/README.md**: Update the documentation index
- [ ] **docs/index.md**: Update the landing page feature table and model counts
- [ ] **Notebook**: Add section to `notebooks/getting_started.ipynb`
- [ ] **Example script**: Add `examples/NN_<feature>.py`
- [ ] Verify all internal links work: `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build`

### 5. Release Artifacts (for each release)
- [ ] **Version bump**: `src/selectools/__init__.py` + `pyproject.toml`
- [ ] **CHANGELOG.md**: Add detailed entry with features, fixes, and migration guide
- [ ] **README.md**: Update "What's New" section, feature table, stats (model count, test count, example count)
- [ ] **ROADMAP.md**: Mark features as completed, update future versions
- [ ] **Git**: Create branch, commit, push, create PR, merge to main
- [ ] **Tag**: `git tag -a vX.Y.Z -m "..."`
- [ ] **PyPI**: `python3 -m build && python3 -m twine upload dist/*`
- [ ] **Verify**: GitHub Pages docs auto-deploy after merge to main

### 6. Cross-Reference Audit
- [ ] Search for hardcoded counts (model count, test count, example count) across all docs
- [ ] Verify pricing references match `models.py`
- [ ] Ensure no broken links (relative paths, anchor references)
- [ ] Verify the `mkdocs.yml` nav includes any new pages

## Release History Pattern

Each release follows this branch + PR workflow:

```
git checkout -b feat/<feature-name>
# ... implement, test, document ...
git add -A && git commit -m "feat: ..."
git push -u origin HEAD
gh pr create --title "..." --body "..."
gh pr merge <number> --merge --delete-branch
```

For releases with version bumps:
```
git checkout -b release/vX.Y.Z
# bump version, update changelog, etc.
git tag -a vX.Y.Z -m "..."
git push origin main --tags
python3 -m build && python3 -m twine upload dist/*
```

## TraceStep Types

Every `AgentTrace` contains `TraceStep` entries with one of these types:

| StepType | Added | Description |
|---|---|---|
| `llm_call` | v0.13.0 | Provider API call (model, tokens, duration) |
| `tool_selection` | v0.13.0 | LLM chose a tool (name, args, reasoning) |
| `tool_execution` | v0.13.0 | Tool executed (name, result, duration) |
| `cache_hit` | v0.13.0 | Response served from cache |
| `error` | v0.13.0 | Error during execution |
| `structured_retry` | v0.13.0 | Structured output validation failed, retrying |
| `guardrail` | v0.15.0 | Input/output guardrail triggered |
| `coherence_check` | v0.15.0 | Coherence check blocked a tool call |
| `output_screening` | v0.15.0 | Tool output screening detected injection |
| `session_load` | v0.16.0 | Session loaded from store |
| `session_save` | v0.16.0 | Session saved to store |
| `memory_summarize` | v0.16.0 | Trimmed messages summarized |
| `entity_extraction` | v0.16.0 | Entities extracted from conversation |
| `kg_extraction` | v0.16.0 | Knowledge graph triples extracted |
| `budget_exceeded` | v0.17.3 | Agent stopped due to token/cost budget limit |
| `cancelled` | v0.17.3 | Agent run cancelled via CancellationToken |
| `prompt_compressed` | v0.17.7 | Older history summarised to free context window |
| `graph_node_start` | v0.18.0 | Graph node execution began |
| `graph_node_end` | v0.18.0 | Graph node execution completed |
| `graph_routing` | v0.18.0 | Graph router resolved next node |
| `graph_checkpoint` | v0.18.0 | Graph state checkpointed |
| `graph_interrupt` | v0.18.0 | Graph paused for human input |
| `graph_resume` | v0.18.0 | Graph execution resumed from checkpoint |
| `graph_parallel_start` | v0.18.0 | Parallel group execution began |
| `graph_parallel_end` | v0.18.0 | Parallel group execution completed |
| `graph_stall` | v0.18.0 | Graph state unchanged for N steps |
| `graph_loop_detected` | v0.18.0 | Hard loop detected (same state hash) |

## Common Pitfalls (from past bugs)

1. **Provider streaming must pass `tools`**: All `stream()` and `astream()` methods MUST forward the `tools` parameter to the underlying API. This was a bug across ALL providers.

2. **`ToolCall` objects must not be stringified**: In `_astreaming_call()`, check `isinstance(chunk, str)` before appending to text buffer. `ToolCall` objects are yielded alongside text chunks.

3. **OpenAI `max_tokens` vs `max_completion_tokens`**: Newer models (GPT-5.x, o-series, GPT-4.1) require `max_completion_tokens`. The `_uses_max_completion_tokens()` helper in `openai_provider.py` handles this.

4. **FallbackProvider.astream() needs error handling**: Must include try/except with `_is_retriable`, `_record_failure`, `on_fallback` callback, and `_record_success` — matching the pattern in `complete()` and `acomplete()`.

5. **Thread safety for FallbackProvider + observers**: The `_wire_fallback_observer` uses a `threading.Lock` and reference count to prevent stack overflow during batch processing.

6. **Structured output vs text parser**: When `response_format` is set, the `ToolCallParser` must NOT intercept valid JSON responses. Guard with `elif response_format is None:`.

7. **`None` content from providers**: Always use `response_msg.content or ""` to prevent `TypeError` when providers return `None` content.

8. **Memory `on_memory_trim` notification**: Use `_memory_add_many()` helper instead of direct `self.memory.add_many()` to ensure observer notifications fire.

9. **Pre-commit YAML check**: mkdocs.yml uses Python tags for emoji extensions. The `check-yaml` hook needs `args: ["--unsafe"]`.

10. **MkDocs links**: Files outside `docs/` (CHANGELOG.md, ROADMAP.md, examples/) must use absolute GitHub URLs, not relative paths.

11. **`astream()` must restore `_system_prompt` in finally**: All three execution methods (`run`, `arun`, `astream`) save `original_system_prompt` before the try block and restore it in `finally`. This prevents modified prompts (e.g. from `response_format`) from leaking to future calls. Was missing from `astream()` until v0.16.1.

12. **`astream()` must have full feature parity with `run()`/`arun()`**: As of v0.16.3, all three methods share `_prepare_run()`, `_finalize_run()`, `_process_response()`, and `_build_max_iterations_result()` helpers. When adding new features to the agent loop, add them to these shared helpers rather than to individual methods. The `_RunContext` dataclass carries all per-run state.

13. **Hooks are deprecated — use observers**: `AgentConfig.hooks` (a plain dict of callbacks) is deprecated. Passing `hooks` emits a `DeprecationWarning` and internally wraps the dict via `_HooksAdapter(AgentObserver)`. New code should always use `AgentObserver` or `AsyncAgentObserver` instead.

14. **FallbackProvider `stream()` / `astream()` must record success AFTER consumption**: The generator must be fully consumed before calling `_record_success()`. Recording before consumption means the circuit breaker never trips on streaming errors. Fixed in v0.17.5.

15. **`astream()` direct provider calls must use `self._effective_model`**: Unlike `run()`/`arun()` which go through `_call_provider`/`_acall_provider`, `astream()` calls providers directly. All model references in `astream()` must use `self._effective_model`, not `self.config.model`.

16. **Async observer events must fire in all exit paths**: The shared `_build_cancelled_result`, `_build_budget_exceeded_result`, and `_build_max_iterations_result` only fire sync observers. In `arun()`/`astream()`, always add `await self._anotify_observers(...)` after calling these helpers.

17. **`datetime.utcnow()` is deprecated — use `datetime.now(timezone.utc)`**: All datetime defaults in dataclasses must use `field(default_factory=lambda: datetime.now(timezone.utc))`, not `default_factory=datetime.utcnow`. The `is_expired` property and pruning code must also use aware datetimes.

18. **Guardrails have async support**: `Guardrail.acheck()` runs sync `check()` via `asyncio.to_thread` by default. `GuardrailsPipeline` has `acheck_input()`/`acheck_output()`. `arun()`/`astream()` use `_arun_input_guardrails()` with `skip_guardrails=True` in `_prepare_run()` to avoid blocking the event loop.

## Current Roadmap

- **v0.15.0** ✅ Enterprise Reliability (guardrails, audit, screening, coherence)
- **v0.16.0** ✅ Memory & Persistence (sessions, summarize-on-trim, entity memory, knowledge graph)
- **v0.16.1** ✅ Consolidation (6 bug fixes, thread safety, 68 new tests, mypy 0 errors)
- **v0.16.2** ✅ astream() prompt leak fix + documentation updates
- **v0.16.3** ✅ Agent refactoring + astream() full parity (14+ bug fixes, 29 new tests, ~800 lines dedup)
- **v0.16.4** ✅ Parallel execution safety + 5 bug fixes
- **v0.16.5** ✅ Design Patterns & Code Quality (agent decomposition, provider Template Method, async observers, terminal actions, hooks deprecation, ADRs) — see `docs/decisions/`
- **v0.16.6** ✅ Gemini thought_signature crash fix (base64 round-trip for non-UTF-8 binary signatures)
- **v0.16.7** ✅ Cleanup (CLI removal, README example table, doc count audit)
- **v0.17.0** ✅ Eval Framework (39 evaluators, A/B testing, regression detection, HTML reports, JUnit XML, snapshot testing, live dashboard, badges, CLI, templates, history, observer events)
- **v0.17.1** ✅ MCP Client/Server — MCPClient, mcp_tools(), MCPServer, MultiMCPClient, circuit breaker
- **v0.17.3** ✅ Agent Runtime Controls — token budget, cancellation, cost attribution, structured results, approval gate, SimpleStepObserver
- **v0.17.4** ✅ Agent Intelligence — token estimation, model switching, knowledge memory enhancement (4 store backends)
- **v0.17.5** ✅ Bug Hunt & Async Guardrails — 91 validated fixes, async guardrails, 40 regression tests
- **v0.17.6** ✅ Quick Wins — ReAct/CoT reasoning strategies, tool result caching, Python 3.9–3.13 CI matrix
- **v0.17.7** ✅ Caching & Context — semantic caching, prompt compression, conversation branching
- **v0.18.0** ✅ Multi-Agent Orchestration + Composable Pipelines — AgentGraph, SupervisorAgent, HITL, checkpointing, parallel execution; Pipeline + `@step` + `|` operator + `parallel()` + `branch()`
- **v0.18.x** 🟡 Advanced Composition — type-safe step contracts, streaming composition, tool composition (`@compose`)
- **v0.19.0** 🟡 Serve & Deploy — Structured AgentConfig, `selectools serve`, FastAPI/Flask, YAML config, templates, playground
- **v0.19.x** 🟡 Enterprise Hardening + Polish & Community — security audit, stability markers, trace viewer, benchmarks, launch, community growth
- **v0.20.0** 🟡 Advanced Agent Patterns + Connectors — PlanAndExecute, ReflectiveAgent, Debate, TeamLead, 50+ evaluators, Bedrock, Azure, FAISS, Qdrant, loaders
