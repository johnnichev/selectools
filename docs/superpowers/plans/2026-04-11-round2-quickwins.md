# Round-2 Competitor Bug-Fix Quick Wins (v0.22.0 addendum)

> Four confirmed live bugs from round-2 competitive mining (LangChain, LangGraph, CrewAI, n8n, LlamaIndex, AutoGen). All verified from source before this plan was written. Scope: one commit per bug, TDD, regression test in `tests/agent/test_regression.py`.

**Goal:** Ship 4 confirmed live bugs as additional commits on the open `v0.22.0-competitor-bug-fixes` branch before tagging v0.22.0.

**Background:** Round 1 (Agno + PraisonAI) shipped 22 bugs as BUG-01 through BUG-22. Round 2 (LangChain/LangGraph/CrewAI/n8n/LlamaIndex/AutoGen) mined ~270k combined competitor stars and found 4 concretely live bugs in selectools. This plan ships those 4 as BUG-23 through BUG-26. Larger top-15 unverified candidate list is parked for a follow-up round.

**Tech stack:** Python 3.9+, pytest, existing `tests/agent/test_regression.py` convention.

---

## BUG-23 — Reranker `top_k=0` falsy fallback

**Source:** LlamaIndex #20880 (`alpha = query.alpha or 0.5` swallowed `alpha=0.0`). Same class, new instance.

**File:** `src/selectools/rag/reranker.py:122`
**Current:** `top_n=top_k or len(results),`
**Bug:** `top_k=0` → `or` short-circuits → `len(results)`. User asking for zero results gets everything.
**Fix:** `top_n=top_k if top_k is not None else len(results),`

**Test:** `test_bug23_reranker_top_k_zero_returns_empty` — CohereReranker.rerank with `top_k=0` must pass `top_n=0` to the Cohere API (assert on mock client call) and return an empty list.

---

## BUG-24 — `_dedup_search_results` keyed only on document text

**Source:** LlamaIndex #21033. Sync recursive retrieval dedup keyed on `node.hash`; async used `(hash, ref_doc_id)`. Dropped legitimately-distinct nodes.

**File:** `src/selectools/rag/vector_store.py:50-72`
**Current:** dedupe key is `r.document.text`.
**Bug:** Two documents with identical text but different sources (same snippet ingested from two files — common in legal/academic corpora) collapse into one result; second source's citation is lost forever.
**Fix:** Key on `(text, doc.metadata.get("source"))` — fall back to tuple-of-sorted-metadata-items when no `source` key present. When metadata is unhashable (nested dicts), fall back to id(doc) so we at least preserve distinct instances.

**Test:** `test_bug24_dedup_preserves_distinct_sources` — two `Document(text="snippet", metadata={"source":"a"})` / `{"source":"b"}` wrapped in SearchResults → `_dedup_search_results` returns both.

---

## BUG-25 — In-memory filter silently returns wrong results for operator-dict values

**Source:** LlamaIndex #20246 / #20237. Qdrant silently returned an empty filter for unsupported operators (`CONTAINS`, `ANY`, `ALL`), matching all documents. Security-adjacent: permission filters bypassed.

**Files:**
- `src/selectools/rag/stores/memory.py:220-234` (`InMemoryVectorStore._matches_filter`)
- `src/selectools/rag/bm25.py:388-395` (`BM25Retriever._matches_filter`)

**Current:** `if doc.metadata.get(key) != value: return False` — when `value` is an operator dict like `{"$in": [1,2]}`, the equality check fails for every doc → zero results, no indication of user error.
**Bug:** User expects `$in`/`$eq`/`$ne` semantics, gets silently empty result. Opposite direction to LlamaIndex's "all docs returned" but same root cause: operator dict silently mishandled.
**Fix:** Add an `_is_operator_dict(value)` helper that returns True when `value` is a dict with ≥1 key starting with `$`. When detected, raise `NotImplementedError("In-memory filter does not support operator syntax '{k}'. Use a vector store backend that supports operators (Chroma, Pinecone, Qdrant, pgvector) or upgrade to equality-only filters.")`. Literal dict values without `$`-prefixed keys still go through the equality check.

**Tests:**
- `test_bug25_memory_filter_operator_dict_raises` — `InMemoryVectorStore.search(query_emb, filter={"user_id": {"$in": [1,2]}})` must raise `NotImplementedError`.
- `test_bug25_bm25_filter_operator_dict_raises` — same for `BM25Retriever.search`.
- `test_bug25_memory_filter_literal_dict_still_works` — backward compat: `filter={"config": {"nested": "v"}}` where metadata has literal `{"config": {"nested": "v"}}` still matches.

---

## BUG-26 — Gemini usage metadata `or 0` pattern

**Source:** LangChain #36500. `token_usage.get("total_tokens") or fallback` silently replaces provider-reported `0`.

**File:** `src/selectools/providers/gemini_provider.py:158-159` (sync `complete`) and `505-506` (stream/astream)
**Current:** `prompt_tokens = (usage.prompt_token_count or 0) if usage else 0`
**Bug:** If the Gemini API ever returns `prompt_token_count=None` alongside a real `candidates_token_count`, the `or 0` conflates "unknown" with "zero" and under-reports total_tokens. Also round-1 pitfall #22 instance not yet swept.
**Fix:** `prompt_tokens = usage.prompt_token_count if usage and usage.prompt_token_count is not None else 0` (same for `candidates_token_count`). Apply to both sync and stream paths.

**Tests:**
- `test_bug26_gemini_usage_zero_preserved` — mock `usage_metadata` with `prompt_token_count=0, candidates_token_count=5`; assert `UsageStats.prompt_tokens == 0` and total_tokens == 5 (not 5 from `0 or 0`, verified via distinct path).
- Simpler version: verify the source code no longer contains the `or 0` pattern via `inspect.getsource`.

---

## Execution order

1. BUG-23 (reranker) — simplest, one-line fix, 1 test
2. BUG-24 (dedup) — small helper change, 1 test
3. BUG-26 (Gemini) — 4 line changes, 1-2 tests
4. BUG-25 (filter) — 2 files, 3 tests, involves NotImplementedError
5. Update CHANGELOG with round-2 quick-wins section
6. Push to `v0.22.0-competitor-bug-fixes`
7. PR #55 auto-updates

Not in scope: top-15 unverified candidates (park for v0.23.0 round-2 plan), LangGraph parallel-interrupt-ID collision (needs real fault injection), CrewAI ContextVar propagation (needs careful test setup).
