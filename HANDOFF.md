# Session Handoff

## What I Was Doing (2026-06-10)

Six-track session via parallel subagents (worktree + branch + PR each):
all 5 open issues plus the Agent-as-API P0 roadmap item. The issue
tracker is now fully cleared — everything is merged or one click away.

## Current State

- **Branch:** `chore/v0.24-integration` (PR #71 — CHANGELOG, example
  renumber, AgentAPI trust-model docs, this handoff)
- **Main:** `0520b20` — PRs #67, #68, #69 squash-merged; external PR #65
  (entity-memory extraction-failure logging) also landed
- **Tests:** 5,250 passed / 0 failed on main; 5,358 passed on the #73
  stack (locally, `-k "not e2e"`); CI green everywhere it has run
- **Version:** still `0.23.0`, no new tag. Unreleased CHANGELOG section
  assumes the work ships as v0.24.0.

## Merge Queue (John's clicks, IN ORDER)

The auto-mode classifier blocks agent-side merges and issue comments;
allow rules for `gh pr merge` / `gh issue comment` are drafted in the
2026-06-10 daily note if wanted.

1. **PR #70** — Gemini schema sanitization + loud empty-candidate
   warning (refs #66). CI green.
2. **PR #71** — integration sweep (CHANGELOG for ALL of #67-#73,
   example renumber 97→98, SERVE.md trust model, this handoff).
3. **PR #72** — ToolResult base + Artifact side-channel (closes #59).
   CI green, 25 new tests.
4. **PR #73** — deferred confirmation flow (closes #58). STACKED on
   #72: CI cannot run until #72 merges (workflow only triggers on
   base=main); it retargets automatically. WAIT for its CI before
   merging. 84 new tests, full suite green locally.
5. Close **#66** with a comment pointing at PR #70 (diagnosis is in the
   PR body: model-side flash-lite FC unreliability, live-reproduced;
   schema conversion exonerated). #57/#58/#59/#60 auto-close.
6. Consider tagging **v0.24.0** (4 features + 1 fix bundle is a solid
   minor release; CHANGELOG is already written for it).

## What Shipped Today (6 PRs)

1. **#67 MERGED** (closes #57) — Anthropic prompt caching:
   `cache_system`/`cache_tools`, all 4 paths, cache token fields on
   `UsageStats`.
2. **#68 MERGED** — Agent-as-API: `serve/api.py` `AgentAPI` (chat, SSE,
   session CRUD, per-user namespaces, bearer auth, CLI `--api`).
   **Competitive P0 backlog now empty.**
3. **#69 MERGED** (closes #60) — `KnowledgeBackend` protocol +
   Supabase/Redis adapters for `KnowledgeMemory`; Sheriff-compatible
   via configurable table/key/data columns.
4. **#70 OPEN** — Gemini fixes (see merge queue).
5. **#72 OPEN** (closes #59) — `selectools.results`: `ToolResult` base
   (kind ClassVar + serializer re-injection — `asdict()` drops
   ClassVars), `Ambiguous`/`NotFound`, `Artifact` (sha256/size per
   review feedback), contextvar `emit_artifact()` →
   `AgentResult.artifacts`. Fixed 2 thread-pool context-propagation
   gaps in `_tool_executor.py`.
6. **#73 OPEN** (closes #58) — `selectools.pending`: full deferred
   confirmation stack per rpelevin's review spec (record binding,
   digest guards, exactly-once consume, PT/EN/ES parser,
   `ChannelAgent`). Sheriff/Clovis can delete their bespoke
   pending-action code.

## Next Steps (after merge queue)

- **Downstream adoption:** Sheriff can (a) enable Anthropic prompt
  caching, (b) swap its `SupabaseSessionStore` + knowledge.py shim +
  `pending_actions.py` for the upstream versions. Each is a small PR in
  ~/projects/sheriff.
- **Roadmap:** P0 done → P1: LiteLLM provider wrapper, cost-optimized
  RouterProvider, A2A protocol, toolbox expansion — or pivot to v1.0.0
  stable-release prep (API freeze, migration guide).
- Issue #66 stays informational (model-side); nothing left to build.

## Watch Out For

- **Shared venv was fixed mid-session:** `hypothesis`, `starlette`,
  `python-multipart` were missing and silently skipping serve tests.
  Also: `.venv` has NO ruff — use system `ruff`. And the editable
  install is stale (0.8.0 .pth) — always test with `PYTHONPATH=$PWD/src`.
- **`Tool._serialize_result` re-injects `kind`** for ToolResult
  instances (#72). ClassVar fields never survive `asdict()` — don't
  "simplify" that away.
- **Thread-pool tool execution now copies contextvars per submission**
  (#72): `Context.run` raises if one Context is entered concurrently —
  each submission needs its OWN copy.
- **`RedisPendingStore` requires Redis >= 6.2** (GETDEL atomic claim).
  Executor closures are never serialized; cross-process confirms need
  `register_executor_factory(kind, ...)`.
- **`AgentAPI` trust model:** `user_id` is self-asserted; deploy behind
  a backend that authenticates end users (documented in SERVE.md).
- **`UsageStats` gained 2 trailing Optional fields**; exhaustive field
  iteration may need updating.
- **Gemini schemas sanitized before send** (#70); flash-lite + tools is
  unreliable upstream — see docs/COMPATIBILITY.md, don't re-litigate.
- **examples numbering:** 97 (agent-as-api), 98 (knowledge backend,
  renamed in #71), 99 (tool results, #72), 100 (deferred confirmation,
  #73).
