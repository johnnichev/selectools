# Session Handoff

## What I Was Doing (2026-06-10)

Prepared and shipped the **v0.24.0 — Production Interop** release. The
2026-06-10 mega-session merge queue is EMPTY: all 14 PRs (#67-#80)
merged to main, including the review-fix passes (per-request agent
isolation in serve + a2a, whole-message confirm anchoring, FTS5
capability probe, bounded bm25, multimodal-aware routing, reserved
litellm kwargs). #73 was superseded by #80 (deferred confirmation).

## Current State

- **Version:** `0.24.0` in both `src/selectools/__init__.py` and
  `pyproject.toml`.
- **CHANGELOG:** `## [0.24.0] - 2026-06-10 — Production Interop`
  finalized and synced to `docs/CHANGELOG.md`.
- **Suite:** 5,968 tests collected (5,721 passed / 22 skipped / 231
  e2e-deselected locally), 106 examples, 152 models, 48 toolbox tools.
- **Quality gate:** ruff format + check clean, bandit clean, mkdocs
  build clean. mypy: 3 accepted pre-existing errors in `agent/core.py`
  plus 1 pre-existing in `serve/api.py` (`_InMemorySessionStore` lacks
  the `search()` added to the `SessionStore` protocol by #79 — typing
  only, never called at runtime; needs a tiny follow-up PR).
- **Docs swept:** README (What's New v0.24, feature table, stats),
  ROADMAP (v0.24.0 ✅ block, backlog items marked shipped), docs/index,
  CONTRIBUTING (both copies, re-synced to Ruff wording), this file.

## What Shipped in v0.24.0 (14 PRs, #67-#80)

Wave 1 (issues): #67 Anthropic prompt caching, #68 Agent-as-API (P0),
#69 KnowledgeBackend (Supabase/Redis), #70 Gemini schema sanitization
+ flash-lite compat, #72 ToolResult/Artifact, #80 deferred
confirmation (supersedes #73).

Wave 2 (roadmap): #74 LiteLLMProvider, #75 RouterProvider, #76 A2A
protocol, #77 toolbox expansion (33 → 48), #78 UnifiedMemory,
#79 cross-session search. Plus #71 integration sweep.

## Next Arcs

- **Sheriff adoption** (after v0.24.0 hits PyPI): enable Anthropic
  prompt caching; replace its SupabaseSessionStore copy, knowledge.py
  shim, and pending_actions.py with the upstream versions. Each is a
  small PR in ~/projects/sheriff.
- **core.py-gated P2 trio** (needs John's AgentConfig/core.py
  sign-off): tool result compression, agent-level HITL,
  planning-as-config — plus UnifiedMemory config wiring
  (`MemoryConfig(unified=True)`).
- **v1.0.0 prep**: API freeze, 0.x→1.0 migration guide, stability
  promotion of the new @beta surface after it bakes a release or two.
- **Remaining backlog:** prompt registry/versioning, durable execution,
  code sandbox, Bedrock-native provider (LiteLLM covers it meanwhile),
  P3 items.

## Watch Out For

- **Venv quirks**: `.venv` has NO ruff (use system) and NO mkdocs (use
  ~/Library/Python/3.9/bin/mkdocs). Stale 0.8.0 editable .pth: always
  test with `PYTHONPATH=$PWD/src`.
- **`serve/api.py` mypy nit**: add `search()` to `_InMemorySessionStore`
  (or type `_store` as a union) in a follow-up PR to get back to the
  3 known `agent/core.py` errors.
- **`Tool._serialize_result` re-injects `kind`** for ToolResult
  subclasses (#72) — ClassVars never survive `asdict()`.
- **Thread-pool tool execution copies contextvars per submission**
  (#72); `Context.run` raises on concurrent re-entry of one Context.
- **`RedisPendingStore` needs Redis >= 6.2** (GETDEL claim); closures
  never serialized — cross-process confirms need
  `register_executor_factory`.
- **RouterProvider** classifier is deterministic by design; breaker
  state is per-escalation-chain; auth errors propagate (only retriable
  errors escalate).
- **A2A** follows the JSON-RPC v0.2.x wire shape, not the newest gRPC
  revision (documented deviation); uses `canceled` spec spelling.
- **LiteLLMProvider**: set `AgentConfig(model="provider/model")` to
  match the provider's model — agent passes config.model per call (same
  caveat as Ollama).
- **SQLite session search**: FTS5 index is additive; old DBs backfill
  lazily on first search; bm25 is near-zero on tiny corpora so score =
  hit count + bm25 tiebreak.
- **`AgentAPI` trust model**: `user_id` is self-asserted (SERVE.md).
- **Gemini schemas sanitized before send** (#70); flash-lite + tools is
  unreliable upstream (docs/COMPATIBILITY.md) — don't re-litigate.
- **examples numbering**: 97 agent-as-api, 98 knowledge backend, 99
  tool results, 100 deferred confirmation, 101 litellm, 102 router,
  103 a2a, 104 toolbox, 105 session search, 106 unified memory.
