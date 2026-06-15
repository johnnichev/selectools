# Session Handoff

## What I Was Doing (2026-06-13)

Merged a large feature wave to `main` and continued the post-1.0 roadmap.
Cut the **v0.27.0 — Scheduling, Reasoning & New Backends** feature release,
shipping the post-1.0 backlog work merged this session. All additive (`@beta`),
no breaking changes (minor bump). Folded into the v1.0 train (John's call to
integrate now rather than hold to a post-tag minor — v1.0 is now materially
larger than the originally-baked v0.24-0.26 surface). Bake clock still ~mid-July.

## Current State

- **Version:** `0.27.2` in `src/selectools/__init__.py` and `pyproject.toml`
  (bug-hunt patch over 0.27.0 — 10 fixes, no API changes). CHANGELOG
  `## [0.27.1] - 2026-06-13 — Bug-Hunt Patch` synced to
  `docs/CHANGELOG.md`. Release = push tag `v0.27.1` (CI `publish-pypi` job
  fires on `refs/tags/v*`).
- **Suite:** ~7,728 tests collected; full non-e2e run 7,492 passed / 0
  failed. 115 examples, 115 source-verified models.
- **Quality gate:** ruff format + check clean, **mypy fully clean (0 errors —
  the 5 long-standing baseline errors were fixed in the tech-debt sweep)**,
  bandit clean. Architecture
  stability gate green (every new module registered).
- **Docs:** README (What's Included + capability table), ARCHITECTURE,
  QUICKSTART, SESSIONS, GUARDRAILS, BENCHMARKS/SCHEDULER/REASONING_TOOLS
  pages, mkdocs nav, landing/index.html, llms.txt/llms-full.txt, ROADMAP,
  and CHANGELOG all reconciled to the new surface (this sweep).

## Merged to main since v0.26.0 (2026-06-13, all `[Unreleased]`)

- **#108** v0.26.0 performance benchmarks published (`docs/modules/BENCHMARKS.md`)
  + post-1.0 backlog reconciled in ROADMAP.
- **#109** `recall` tool — completes the agentic-memory pair (`toolbox/memory_tools.py`).
- **#110** Toolbox +4 categories (Discord, S3, browser, image-gen): 48 → 56 tools.
- **#111** UnifiedMemory AgentConfig wiring — `MemoryConfig(unified=True, ...)`;
  also delivered episodic-retention config (`add_turn` auto-prunes).
- **#112** Cache-rate cost for OpenAI + Gemini (`calculate_cost_with_cached_input`,
  `cached_prompt_cost`); 24 rates re-verified live; gemini-embedding-2 documented
  as GA/recommended-for-new, default stays -001 (incompatible space).
- **#113** Cron/scheduled agents (`scheduler.py`: `AgentScheduler`, `cron`, `every`).
- **#114** Reasoning tools (`toolbox/reasoning_tools.py`: `think`/`analyze`, bounded).
- **#116** MongoSessionStore; **#117** DynamoDBSessionStore (sessions now 6 backends).
- **#118** `PromptInjectionGuardrail` — heuristic injection/jailbreak detection.

## Next Up — roadmap is decision-gated

All autonomously-buildable Future/Watch items are shipped. The rest need a
product call (see ROADMAP "needs a product decision"):
- Firestore session backend — only if there's real demand (adds a dep).
- Model-based guard (beyond the heuristic #118) — hosting decision.
- Multi-channel bot gateway — in-repo vs separate package.
- Learning system — needs a concrete spec.
- Shadow git checkpoints — only if steering toward coding agents.

Sheriff #303 (the prior "next up") is DONE — merged + deployed 2026-06-12,
prod logs clean on selectools 0.26.0.

## The v1.0 Bake Window Continues — July Tag Unchanged

All v1.0 code work remains merged and baking. v0.26.0 is a mid-bake
patch, not a new feature wave: the `@beta` surface got real-world
mileage, the bake hunt caught a real safety bug, and the fix shipped.
The July 1.0 tag plan is unchanged.

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

- **Sheriff adoption:** DONE — caching, parser, knowledge, and the pending
  store/sanitizer deletion all merged + deployed (Sheriff #300-#303).
- **UnifiedMemory config wiring:** DONE (#111).
- **Remaining backlog:** prompt registry/versioning, durable execution,
  code sandbox, Bedrock-native provider (LiteLLM covers it meanwhile),
  P3 items. Plus the decision-gated Future/Watch items above.

## Watch Out For

- **Venv**: `.venv` now has `ruff` and `mkdocs` installed (this session);
  `pymongo` is NOT installed (the Mongo backend tests inject a fake via
  `sys.modules`, the same pattern as the Redis/Mongo/Dynamo backend tests).
  Run `bandit` via `python -m bandit` if the shim shebang is broken. The
  editable `.pth` points at the main repo's `src`; pytest's
  `pythonpath = ["src", "."]` makes worktree runs use the worktree's own
  source.
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
