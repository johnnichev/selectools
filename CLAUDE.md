# Selectools — Agent Instructions

This file provides context for AI coding agents (Claude, Cursor, Copilot, etc.) working on the selectools codebase.

## Project Overview

**Selectools** is a production-ready Python library for building AI agents with tool calling, RAG, and hybrid search. It supports OpenAI, Anthropic, Gemini, and Ollama providers.

- **Version**: Check `src/selectools/__init__.py` for current version
- **Python**: 3.9+
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
├── knowledge.py             # KnowledgeMemory (cross-session durable memory)
├── models.py                # 146 model registry with pricing (single source of truth)
├── pricing.py               # Derives pricing from models.py
├── usage.py                 # Token + cost tracking
├── trace.py                 # AgentTrace, TraceStep (14 step types — see list below)
├── observer.py              # AgentObserver (25 sync events) + AsyncAgentObserver (25 async events) + LoggingObserver
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
└── env.py                   # Environment variable helpers

tests/                       # 1640 tests (unit, integration, regression, E2E)
├── agent/                   # Agent core tests
├── providers/               # Provider-specific tests
├── rag/                     # RAG pipeline tests
├── tools/                   # Tool system tests
├── integration/             # Cross-module integration tests
├── core/                    # Framework-level tests
└── test_*.py                # Module-level unit tests

examples/                    # 38 numbered example scripts (01-38)
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

## Current Roadmap

- **v0.15.0** ✅ Enterprise Reliability (guardrails, audit, screening, coherence)
- **v0.16.0** ✅ Memory & Persistence (sessions, summarize-on-trim, entity memory, knowledge graph)
- **v0.16.1** ✅ Consolidation (6 bug fixes, thread safety, 68 new tests, mypy 0 errors)
- **v0.16.2** ✅ astream() prompt leak fix + documentation updates
- **v0.16.3** ✅ Agent refactoring + astream() full parity (14+ bug fixes, 29 new tests, ~800 lines dedup)
- **v0.16.4** ✅ Parallel execution safety + 5 bug fixes
- **v0.16.5** ✅ Design Patterns & Code Quality (agent decomposition, provider Template Method, async observers, terminal actions, hooks deprecation, ADRs) — see `docs/decisions/`
- **v0.17.0** 🔵 Multi-Agent Orchestration — see `MULTI_AGENT_PLAN.md`
- **Backlog**: Connector Expansion, Ecosystem Parity, Structured AgentConfig, Polish & Community
