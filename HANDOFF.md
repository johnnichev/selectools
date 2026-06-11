# Session Handoff

## What I Was Doing (2026-06-11)

Prepared the **v0.25.0 — Hardening & v1.0 Prep** release
(branch `release/v0.25.0`). This closes the v1.0 engineering arc:
everything v1.0 needs from the codebase is now merged. What remains
for the 1.0 tag is mechanical and happens at tag time, after a bake
window.

## Current State

- **Version:** `0.25.0` in both `src/selectools/__init__.py` and
  `pyproject.toml`.
- **CHANGELOG:** `## [0.25.0] - 2026-06-11 — Hardening & v1.0 Prep`
  finalized and synced to `docs/CHANGELOG.md`.
- **Suite:** 7,268 tests collected (7,016 passed / 27 skipped / 231
  e2e-deselected locally), 111 examples.
- **Public surface:** 433 unique public symbols, 100% stability-marked
  (205 stable / 228 beta), module-level `__stability__` on all 123
  public modules, architecture-test CI gate enforcing markers on every
  future public symbol (#95).
- **Quality gate:** ruff format + check clean on src/tests (the only
  ruff findings are pre-existing in `notebooks/getting_started.ipynb`,
  which CI does not lint), bandit clean at -ll, mkdocs build clean.
  mypy baseline: 5 accepted pre-existing errors (2 in
  `agent/_tool_executor.py`, 3 in `agent/core.py`).
- **Docs swept:** README (What's New v0.25, stats), ROADMAP (v0.25.0 ✅
  block + backlog items #86/#87/#88 marked shipped + v1.0 progress),
  CONTRIBUTING (both copies), landing/index.html counters, docs/llms.txt,
  this file.

## What Shipped in v0.25.0 (PRs #84-#88, #90-#95)

- **Features (`@beta`):** pending intent hooks (#85), knowledge
  pre-save sanitizers (#84), planning-as-config (#86), tool result
  compression (#87), agent-level HITL (#88).
- **v1.0 groundwork:** wart removal (#94 — public
  `clone_for_isolation()`, `__all__` reconciliation, **BREAKING**
  removal of `AgentConfig.hooks`), stability marking sweep + CI gate
  (#95), security audit + migration guide + compatibility refresh
  (#91), real-Redis Lua smoke tests (#93), docs count corrections
  (#92), CHANGELOG for the feature wave (#90).
- **Fix:** runtime-checkable Protocol isinstance regression on
  py3.9-3.11 (#95).

## The v1.0 Arc Is Code-Complete — Bake Window Starts Now

All v1.0 code work is merged: API freeze (warts removed), 100% stability
marking with a CI gate, published security audit, 0.x→1.0 migration
guide, compatibility matrix. v0.25.0 now bakes in the wild so the new
`@beta` surface and the 19 fresh `@stable` promotions get real-world
mileage before the 1.0 promise is stamped on them.

### July 1.0 Tag Checklist (in order)

1. **Drop Python 3.9** — dedicated PR: `requires-python = ">=3.10"`,
   remove 3.9 from the CI matrix, drop the py39 typing shims and the
   3.9-only Protocol workaround notes, retarget ruff
   (`target-version = "py310"`). This is the only remaining breaking
   change and it lands BEFORE the tag, not in it.
2. **Promote-after-bake review** — sweep the `@beta` surface for
   anything that baked cleanly through the window (v0.24 + v0.25
   betas; check issue tracker for API-shape complaints). Promotions
   are deliberate, not automatic.
3. **Classifier flip at tag** — `Development Status :: 5 -
   Production/Stable` in pyproject.toml, version `1.0.0`, CHANGELOG
   entry, tag. The release commit for 1.0 should be boring.

## Next Arcs (post-1.0 or parallel)

- **Sheriff adoption:** enable Anthropic prompt caching; replace its
  SupabaseSessionStore copy, knowledge.py shim, and pending_actions.py
  with upstream. Small PRs in ~/projects/sheriff.
- **UnifiedMemory config wiring** (`MemoryConfig(unified=True)`) —
  the one P2 deferred from the core.py trio.
- **Remaining backlog:** prompt registry/versioning, durable execution,
  code sandbox, Bedrock-native provider (LiteLLM covers it meanwhile),
  P3 items.

## Watch Out For

- **Venv quirks**: `.venv` has NO ruff (use system) and NO mkdocs (use
  ~/Library/Python/3.9/bin/mkdocs). The venv `bandit` shim has a broken
  shebang — run `python -m bandit`. Stale 0.8.0 editable .pth: always
  test with `PYTHONPATH=$PWD/src`.
- **`AgentConfig(hooks=...)` now raises TypeError** (#94). Migration
  mapping lives in `docs/MIGRATION_1.0.md`. Don't resurrect it for a
  "quick fix" — observers cover every hook point.
- **Stability gate is parametrized per symbol**
  (`tests/test_architecture.py::test_every_public_symbol_has_stability_marker`).
  Any new public symbol without a marker fails CI by design; mark it,
  don't exempt it.
- **Protocol classes + markers**: stability markers must not become
  structural members of runtime-checkable Protocols (py3.9-3.11
  regression fixed in #95) — keep markers in the registry for
  Protocols, not as attributes.
- **`tests/rag/test_property_based_rag.py::test_full_metadata_filter_always_matches`**
  flaked once during release prep (hypothesis draw), passed on re-run
  and in isolation. If it flakes again, pin and minimize the failing
  example instead of rerunning.
- **`RedisPendingStore` needs Redis >= 6.2** (GETDEL claim);
  `tighten_ttl` is an id-pinned atomic Lua rewrite — covered by
  real-Redis smoke tests (#93), keep a Redis running locally to
  exercise them.
- **`Tool._serialize_result` re-injects `kind`** for ToolResult
  subclasses (#72) — ClassVars never survive `asdict()`.
- **Gemini flash-lite + tools is unreliable upstream**
  (docs/COMPATIBILITY.md) — don't re-litigate.
