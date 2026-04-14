# Session Handoff

## What I Was Doing

Shipping v0.22.0: 3 rounds of competitive bug mining (9 repos, ~325k combined stars) + verification pass + cookbook expansion + doc sweep. 38 bugs fixed, 30 cookbook recipes, 94 examples. Now ready for P0 competitive features before tagging.

## Current State

- **Branch:** `v0.22.0-round4-features` (1 commit ahead of main)
- **Main:** `2238a2f` — squash-merged PR #55 with rounds 1-3 (34 bugs + cookbook + examples + docs)
- **Last commit:** `0614dfe` — BUG-36/37/38/39 verification-pass fixes
- **Tests passing:** yes — 5,064 passed, 3 skipped, 0 failed
- **Version:** `0.22.0` in `__init__.py` and `pyproject.toml` (no git tag yet)
- **Working tree:** clean

## What's Left for v0.22.0

### P0 Features (competitive gap closure — not started)

These are from the original competitive gap analysis (`project_competitive_gap_analysis.md`). They're what every competitor (Agno, PraisonAI) already has. Ship before tagging v0.22.0.

1. **Tool-call loop detection** (PraisonAI, P0, Low effort ~4h)
   - Detect agents stuck in repetitive tool-call cycles
   - 3 detectors: repeat (same tool+args N times), poll-no-progress (tool called but output unchanged), ping-pong (A->B->A->B)
   - stdlib only, no new deps
   - Target: new module `src/selectools/loop_detection.py` + wire into `agent/core.py` main loop
   - Use `/brainstorming` then `/writing-plans` then `/subagent-driven-development`

2. **Agentic memory (memory-as-tool)** (Agno, P0, Low effort ~4h)
   - Agent decides when to store/recall via tool calls vs always-on passive memory
   - Two new tools: `remember(key, value)` and `recall(query)` backed by existing `KnowledgeMemory` or `EntityMemory`
   - Target: new `src/selectools/toolbox/memory_tools.py` or extend `entity_memory.py`
   - Cookbook recipe + example already have placeholders in the 30-recipe cookbook

3. **Agent-as-API (production serve enhancement)** (Agno, P0, Medium effort ~8h)
   - Enhanced `selectools serve` with per-user session isolation, auth middleware, streaming SSE
   - Existing: `serve/app.py` + `serve/cli.py` + `serve/_starlette_app.py` already work
   - Gap: no per-user isolation, no auth, limited streaming
   - Target: extend `serve/` module

### Remaining Bugs (deferred — need architecture decisions)

4. **BUG-35: Parallel interrupt ID collision** (graph.py:1473)
   - Second child's interrupt state silently dropped in `ParallelGroupNode`
   - Root cause: `if child_interrupted and interrupted_child_state is None` — first-wins pattern
   - Fix options: (a) collect all interrupted states + merge all `_interrupt_responses` (minimal), (b) support multi-interrupt resume API (full), (c) raise clear error when >1 child interrupts (safety net)
   - Recommendation: option (a) for v0.22.0, option (b) for v0.23.0

5. **BUG-40: Pipeline `_is_subtype` weak validation** (pipeline.py:67)
   - `hasattr(output, "__origin__") -> return True` short-circuits on any generic
   - DX improvement, not correctness — mismatched pipelines fail at runtime, not composition time
   - Low priority, defer to v0.23.0

## Bugs Already Shipped (38 total across 3 rounds + verification)

### Round 1 (BUG-01 to BUG-22): Agno + PraisonAI
Streaming tool-call drops, typing.Literal crashes, asyncio.run re-entry, HITL interrupts (parallel/subgraph/multi-interrupt), ConversationMemory thread safety, think-tag stripping, RAG batch limits, MCP concurrent race, str->typed coercion, Union typing, GraphState fail-fast, session namespace, summary cap, cancelled-result persistence, AgentTrace lock, async observer logging, clone isolation, OTel/Langfuse locks, vector store dedup, Optional[T] handling.

### Round 2 (BUG-23 to BUG-26): LangChain + LlamaIndex
Reranker top_k=0 falsy fallback, _dedup_search_results text-only keying, in-memory filter operator-dict silent-ignore, Gemini provider or-0 usage metadata.

### Round 3 (BUG-27 to BUG-34): LiteLLM + Pydantic AI + Haystack
FallbackProvider retry list (529/504/408/522/524), Azure deployment-name family detection, bare list/dict tool schemas, pipeline.parallel shared input, malformed tool-call JSON silent drop, run_in_executor drops contextvars (5 sites), astream missing aclosing, max_iterations shared with structured-retry budget.

### Verification Pass (BUG-36 to BUG-39): Confirmed from rounds 2+3
Vector store ref mutation (defensive copy), Cohere embedder batch limit (96), Gemini streaming ToolCall dedup, Gemini parallel tool result grouping (same-role merge).

## Key Decisions Made

- **v0.22.0 scope expanded** to include P0 features because the tag hasn't been pushed yet. PR #55 was squash-merged to main; the new branch `v0.22.0-round4-features` continues from that merge commit.
- **Cookbook expanded from 7 to 30 recipes** covering all round-2/3 features + general gaps.
- **CLAUDE.md pitfalls 27-30 added** for: aclosing, contextvars propagation, malformed JSON recovery, structured retry budget.
- **Bug methodology validated**: "grep selectools source to confirm live" directive is mandatory for all future research prompts.
- **6 NOT-APPLICABLE verifications** closed permanently: time-travel resume, multi-producer join, Anthropic thinking+tool_choice, MCP pending-future, streaming merge collision, provider config round-trip.

## Research Saved (for future rounds)

All competitive research is in Claude memory files:
- Round 1: `project_competitive_agno.md`, `_praisonai.md`, `_superagent.md`
- Round 2: `project_competitive_langchain.md`, `_langgraph.md`, `_crewai.md`, `_n8n.md`, `_llamaindex.md`, `_autogen.md`
- Round 3: `project_competitive_litellm.md`, `_pydantic_ai.md`, `_haystack.md`
- Gap analyses: `project_competitive_gap_analysis.md`, `_round2.md`, `_round3.md`
- Backlog: `project_competitive_research_backlog.md` (rounds 4-7 targets)

Obsidian articles:
- `Clovis/01-projects/content/blog-draft-competitor-bug-hunt.md` (round 1)
- `Clovis/01-projects/content/selectools-round2-competitor-bug-hunt.md` (round 2)
- `Clovis/01-projects/content/selectools-round3-competitor-bug-hunt.md` (round 3)

## Watch Out For

- **`_parse_tool_call_arguments` return type changed** in BUG-31 from `dict` to `Tuple[Dict, Optional[str]]`. Ollama's override was updated. Any new provider that overrides this method MUST return the tuple.
- **`run_in_executor_copyctx`** is now the ONLY way to schedule sync work from async contexts. Raw `loop.run_in_executor()` is forbidden (pitfall #28). Test `test_bug32_five_executor_sites_use_contextvar_helper` catches violations.
- **`aclosing`** from `_async_utils.py` is a Python 3.9 backport. Replace with `contextlib.aclosing` when dropping 3.9.
- **Structured retry budget** (BUG-34): outer loop uses `while ctx.iteration < max_iterations + ctx.structured_retries`. Loop-detection feature must account for this.
- **`ToolCall.parse_error`** is a new field (BUG-31). It's `Optional[str]` default `None` — existing serialized data is fine, but code iterating ToolCall fields explicitly may need updating.
- **Gemini `_format_messages` now merges consecutive same-role Content** (BUG-39). Tests asserting position-based indexing must use content-presence checks instead.
- **`element_type`** is a new field on `ToolParameter` (BUG-29). `to_schema()` emits `items`/`additionalProperties` when populated.
