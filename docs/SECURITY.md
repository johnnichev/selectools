# Security Audit

**Version audited:** v0.19.2
**Audit date:** 2026-03-30
**Tool:** [Bandit](https://bandit.readthedocs.io/) 1.7 + manual review of all `# nosec` annotations
**Result:** Zero findings requiring remediation. All suppressed warnings are justified.
**SBOM:** [`sbom.json`](https://github.com/johnnichev/selectools/blob/main/sbom.json) — CycloneDX 1.6, 4 core production dependencies

To report a vulnerability, see [Vulnerability Reporting](#vulnerability-reporting) below.

---

## What Was Checked

1. Full `bandit -r src/ -ll -q -c pyproject.toml` scan (medium severity and above)
2. Manual review of every `# nosec` annotation in `src/selectools/` — 41 annotations across 15 files
3. Review of all SQL query construction for injection risk
4. Review of all file path operations for traversal risk
5. Review of all user-controlled content interpolated into LLM prompts

---

## Annotation-by-Annotation Review

### B110 — Bare exception swallowing (19 sites)

**Files:** `observer.py`, `agent/_lifecycle.py`, `agent/_memory_manager.py`, `agent/core.py`, `providers/fallback.py`, `mcp/client.py`, `mcp/multi.py`, `orchestration/supervisor.py`, `orchestration/checkpoint.py`, `evals/suite.py`, `evals/serve.py`

**Pattern:** `except Exception: # nosec B110` inside observer notification loops, best-effort cleanup paths, and MCP tool discovery.

**Justification:** These are all fire-and-forget callbacks. An observer failing to process an event (e.g., a logging handler throwing a network error) must not crash the agent run that triggered it. This is the intended contract and is documented in the `AgentObserver` protocol. The same applies to MCP tool listing — a single unavailable MCP server should not prevent other tools from loading.

**Risk:** None. The agent's own error handling is unaffected; only side-effect callbacks are silenced.

---

### B112 — Bare except with `continue` (1 site)

**File:** `env.py:26`

**Pattern:** Loading `.env` file; bare except continues to the next key.

**Justification:** The `.env` loader is fail-quiet by design — if a key cannot be read (malformed line, encoding issue), the rest of the file should still load. This matches the behavior of `python-dotenv` and similar tools.

**Risk:** None.

---

### B301 / B403 — Pickle usage (2 sites)

**File:** `cache_redis.py:11,68`

**Pattern:** `import pickle` and `pickle.loads(raw)` for Redis cache serialization.

**Justification:** Data stored in Redis by selectools is always serialized by the same process (via `pickle.dumps`). The cache is not a public ingestion endpoint — it is a process-local cache that happens to use Redis as a backend. No external or user-controlled bytes are deserialized. The annotation comment at line 11 states this explicitly.

**Risk:** None in the intended deployment model. If you expose your Redis instance to untrusted writers, do not use `RedisCache`. Use `InMemoryCache` or `SemanticCache` instead.

---

### B608 — SQL via f-string (7 sites)

**Files:** `checkpoint_postgres.py` (5 sites), `knowledge_graph.py` (1 site), `rag/stores/sqlite.py` (1 site), `observe/trace_store.py` (1 site)

**Pattern:** Table names constructed with f-strings (e.g., `f"SELECT ... FROM {self._table} ..."`).

**Justification:** In every case the interpolated value is either:

- A constructor argument that is a developer-controlled configuration value (e.g., `table_name="my_traces"`), validated at construction time, and never exposed to end users.
- A compile-time literal constant in the module.

All actual query parameters (IDs, metadata, content) are passed via parameterized placeholders (`?` for SQLite, `%s` for Postgres), not f-strings. Bandit flags f-strings in SQL regardless of what is interpolated; the annotation suppresses the false positive.

**Risk:** None for standard usage. If you allow untrusted users to specify table names at agent construction time, validate the table name before passing it to the constructor.

---

### B324 — MD5 usage (1 site)

**File:** `orchestration/graph.py:137`

**Pattern:** `hashlib.md5(payload.encode()).hexdigest()` for loop detection.

**Justification:** This is a non-cryptographic fingerprint used to detect whether graph state has changed between steps. MD5 collisions would at worst cause the loop detector to miss a repeated state — a minor correctness concern, not a security issue. No authentication, key derivation, or signature is involved.

**Risk:** None.

---

### B104 — Bind to 0.0.0.0 (3 sites)

**Files:** `serve/cli.py:30`, `serve/app.py:136,161`

**Pattern:** Default host `"0.0.0.0"` in `selectools serve`.

**Justification:** `selectools serve` is a development server. Binding to all interfaces is the expected default for a locally-run playground, matching the behavior of `uvicorn`, `flask run`, and `mkdocs serve`.

**Risk:** In production, place `selectools serve` behind a reverse proxy (nginx, Caddy) and bind to `127.0.0.1`:

```bash
selectools serve agent.yaml --host 127.0.0.1
```

---

### B405 — xml.etree import (1 site)

**File:** `evals/junit.py:5`

**Pattern:** `import xml.etree.ElementTree as ET`

**Justification:** JUnit XML is **generated**, not parsed. `ElementTree` is used only for `ET.Element`, `ET.SubElement`, and `ET.ElementTree` to build the output tree. No untrusted XML is ever parsed, so XXE vulnerabilities do not apply.

**Risk:** None.

---

### B105 — Hardcoded password string (1 site)

**File:** `evals/types.py:13`

**Pattern:** `PASS = "pass"  # nosec B105`

**Justification:** This is the string value of the `CaseVerdict.PASS` enum member. Bandit incorrectly identifies the string `"pass"` as a hardcoded password. It is not a credential of any kind.

**Risk:** None (false positive).

---

## Prompt Injection Mitigations

All LLM judge prompts in `evals/llm_evaluators.py` and the coherence checker in `coherence.py` fence user-controlled content with `<<<BEGIN_USER_CONTENT>>>` / `<<<END_USER_CONTENT>>>` delimiters to prevent test case inputs from hijacking the judge's scoring. This was implemented in v0.19.1 as part of the Ralph loop quality initiative.

---

## Path Traversal Mitigations

The following storage backends sanitize user-controlled names using `Path(name).name` to strip directory components before constructing file paths:

- `BaselineStore` (eval regression baselines)
- `SnapshotStore` (eval snapshots)
- `EvalHistory` (eval history files)
- `FileKnowledgeStore` (knowledge entries)
- `JsonFileSessionStore` (session files)

A session ID or suite name containing `../../etc/passwd` will be reduced to `passwd` before any file operation.

---

## Vulnerability Reporting

To report a security vulnerability privately:

- **Email:** support@nichevlabs.com
- **Response SLA:** 48 hours acknowledgement, 7 days for triage
- **Disclosure:** Coordinated disclosure — we will work with you on a fix before public announcement

Please do not open a public GitHub issue for security vulnerabilities.
