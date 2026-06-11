# Session Handoff

## What I Was Doing (2026-06-10)

Single-evening parallel-subagent mega-session: cleared all 5 open issues,
shipped all 4 P1 roadmap features, 2 P2 features, and the Agent-as-API
P0 item. 13 PRs total (#67-#79): 3 merged, 10 awaiting John's clicks.

## Current State

- **Branch:** `chore/v0.24-integration` (PR #71 — consolidated CHANGELOG
  for ALL of #67-#79, example renumber, AgentAPI trust-model docs, this
  handoff)
- **Main:** `0520b20` — #67, #68, #69 merged; external #65 also landed
- **Integration rehearsal PASSED:** all 10 open branches merged locally
  in queue order with ZERO conflicts; combined suite **5,663 passed /
  0 failed**; mkdocs build clean. The merge queue is mechanical.
- **CI:** green on every PR that has checks (#73 gets CI only after #72
  merges and it retargets to main — stacked PR, workflow filters on
  base=main).
- **Version:** still `0.23.0`. The Unreleased CHANGELOG assumes v0.24.0.

## Merge Queue (John's clicks, IN ORDER)

1. **#70** — Gemini schema sanitization + loud empty-candidate warning
2. **#71** — integration sweep (CHANGELOG for everything, docs, handoff)
3. **#72** — ToolResult base + Artifact side-channel (closes #59)
4. **#73** — deferred confirmation flow (closes #58) — WAIT for its CI
   after #72 merges
5. **#74** — LiteLLMProvider (100+ models)
6. **#75** — RouterProvider (cost-optimized routing)
7. **#76** — A2A protocol (server + client)
8. **#77** — toolbox expansion (15 tools, 6 categories)
9. **#78** — UnifiedMemory (tiered, auto-promotion)
10. **#79** — cross-session search (4 backends)
11. Close **#66** with a pointer to #70 (model-side flash-lite issue;
    diagnosis in the PR body). #57/#58/#59/#60 auto-close.
12. Tag **v0.24.0** (CHANGELOG is ready; this is a hefty minor).

The auto-mode classifier blocks agent-side `gh pr merge`/`gh issue
comment`; allow-rule snippet is in the 2026-06-10 daily note.

## What Shipped (13 PRs)

Wave 1 (issues): #67 Anthropic prompt caching (merged), #68 Agent-as-API
(merged, P0 done), #69 KnowledgeBackend (merged), #70 Gemini fixes,
#72 ToolResult/Artifact, #73 deferred confirmation.

Wave 2 (roadmap): #74 LiteLLM, #75 Router, #76 A2A, #77 toolbox,
#78 UnifiedMemory, #79 session search. Plus #71 integration sweep.

Suite: 5,064 → 5,663 tests (~600 new). Blog draft about the session:
vault `01-projects/content/selectools-parallel-subagent-shipping-day.md`.

## Next Steps (after merge queue)

- **Sheriff adoption** (after v0.24.0 on PyPI): enable Anthropic prompt
  caching; replace its SupabaseSessionStore copy, knowledge.py shim, and
  pending_actions.py with the upstream versions. Each is a small PR in
  ~/projects/sheriff.
- **Remaining P2 items all need core.py / AgentConfig sign-off** (John's
  decision): tool result compression, agent-level HITL, planning-as-
  config, UnifiedMemory config wiring (`MemoryConfig(unified=True)`).
- **Remaining backlog:** prompt registry/versioning, durable execution,
  code sandbox, Bedrock-native provider (LiteLLM covers it meanwhile),
  P3 items.
- **v1.0.0 prep** is the next big arc: API freeze, migration guide,
  stability promotion of the new @beta surface after it bakes.

## Watch Out For

- **Venv fixed mid-session**: hypothesis/starlette/python-multipart were
  missing (silently skipping serve tests); httpx was added to the serve
  extras by #76. `.venv` has NO ruff (use system) and NO mkdocs (use
  ~/Library/Python/3.9/bin/mkdocs). Stale 0.8.0 editable .pth: always
  test with `PYTHONPATH=$PWD/src`.
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
- **examples numbering**: 97 agent-as-api, 98 knowledge backend (renamed
  in #71), 99 tool results, 100 deferred confirmation, 101 litellm,
  102 router, 103 a2a, 104 toolbox, 105 session search, 106 unified
  memory.
