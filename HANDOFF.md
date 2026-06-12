# Session Handoff

## What I Was Doing (2026-06-12)

Prepared the **v0.26.0 — Safety Patch & Verified Registry** release
(branch `release/v0.26.0`). This is the planned mid-bake release:
the bake-window bug hunt (#103/#104) found a destructive-CONFIRM
safety bug in the `@beta` pending surface, and the registry refresh
(#105/#106) was ready — shipping them together gets the safety fix
to `selectools.pending` consumers immediately without disturbing the
v1.0 bake (the fixes ARE the bake working as intended).

## Current State

- **Version:** `0.26.0` in both `src/selectools/__init__.py` and
  `pyproject.toml`.
- **CHANGELOG:** `## [0.26.0] - 2026-06-12 — Safety Patch & Verified
  Registry` finalized and synced to `docs/CHANGELOG.md`.
- **Suite:** 7,420 tests collected (7,168 non-e2e passed / 27 skipped /
  231 e2e-deselected locally), 111 examples, 115 source-verified
  models.
- **Public surface:** 437 unique public symbols across 123 public
  modules, 100% stability-marked, architecture-test CI gate intact.
- **Quality gate:** ruff format + check clean, bandit clean at -ll
  with pyproject config, full non-e2e suite green. mypy baseline:
  5 accepted pre-existing errors (2 in `agent/_tool_executor.py`,
  3 in `agent/core.py`).
- **Docs swept:** README (What's New v0.26, stats), ROADMAP (v0.26.0 ✅
  block), CHANGELOG synced, landing/index.html version strings
  (status bar, footer, JSON-LD softwareVersion) + og-image.svg,
  this file.

## What Shipped in v0.26.0 (PRs #103-#106)

- **Safety fix:** `RegexConfirmParser` non-leading negation fired
  destructive CONFIRM ("se você não pode apagar, tudo bem" parsed as
  confirm). Negation token anywhere now vetoes the restated-verb
  branch (#103). All `selectools.pending` consumers should upgrade.
- **Registry refresh (#105):** 152 → 115 models, every entry
  source-verified; claude-fable-5/opus-4-8/opus-4-7, gpt-5.5 family,
  gemini-3.5 line added; retired-model constants REMOVED (BREAKING
  for direct registry references); opus-4-1 pricing fixed
  ($5/$25 → $15/$75); gpt-5 context specs corrected.
- **Cache-aware cost (#106):** `calculate_cost(...,
  cache_read_input_tokens=, cache_creation_input_tokens=)`;
  AnthropicProvider `UsageStats.cost_usd` now cache-accurate.
- **Fixes:** A2A -32602 instead of HTTP 500 on malformed
  `message.parts` (#103); bake-hunt tests gated on starlette (#104);
  Gemini embedding dimension constant 3072 (#106).

## Next Up

- **Sheriff #303** — bump Sheriff's selectools pin to 0.26.0 so its
  pending-confirmation flow picks up the negation-veto safety fix.
  This is the first post-release action.

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
