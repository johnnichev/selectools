# Security Audit — v0.25.0

**Version audited:** v0.25.0
**Audit date:** 2026-06-11
**Tools:** [Bandit](https://bandit.readthedocs.io/) 1.8.6 static analysis + manual review of all `# nosec` annotations + [pip-audit](https://pypi.org/project/pip-audit/) 2.9.0 dependency scan
**Result:** Zero Bandit findings at medium severity or above. All suppressed warnings are justified. Three known advisories in transitive (not directly declared) dependencies; fixed versions are released upstream but require Python >= 3.10 (see assessment below).
**SBOM:** [`sbom.json`](https://github.com/johnnichev/selectools/blob/main/sbom.json) — CycloneDX 1.6, regenerated for v0.25.0 with `cyclonedx-py` 7.3.0

This supersedes the [v0.19.2 audit](SECURITY.md). To report a vulnerability, see the
[Security Policy](https://github.com/johnnichev/selectools/blob/main/SECURITY.md).

---

## Scope and Methodology

1. Full `bandit -r src/ -ll -q -c pyproject.toml` scan of `src/selectools/`
2. Manual review of every `# nosec` annotation — **72 annotations across 34 files**
3. `pip-audit` scan of the four core production dependencies and their resolved transitive tree
4. SBOM regeneration and verification
5. Review of the framework's defensive layers (threat-model summary below)

### Bandit configuration

The scan uses the `[tool.bandit]` config in `pyproject.toml`:

- **Targets:** `src/` only; `tests/`, `examples/`, and `scripts/` are excluded (not shipped code)
- **Skips:** `B101`, `B104`, `B110`, `B310`, `B324`, `B601`, `B602` are skipped globally.
  Each skipped check corresponds to an intentional pattern documented below; individual
  sites additionally carry `# nosec` annotations so the justification survives config changes.
- **`-ll` flag:** report only **MEDIUM and HIGH severity** issues. Low-severity findings
  (e.g. `assert` usage, `try/except/pass`) are triaged via the annotation review instead
  of failing the build.

### CI gate

The same command runs as a dedicated job in `.github/workflows/ci.yml` on every push and
pull request. Any medium-or-higher finding fails CI, so no code with an unreviewed Bandit
finding can merge to `main`:

```yaml
- name: Run bandit
  run: bandit -r src/ -ll -q -c pyproject.toml
```

This is also a release gate: tags are not cut while the Bandit job is red.

---

## Current Results (v0.25.0)

```text
$ bandit -r src/ -ll -q -c pyproject.toml
$ echo $?
0
```

| Severity | Findings |
|----------|----------|
| High     | 0 |
| Medium   | 0 |
| Low (suppressed via config/annotations, all reviewed below) | 72 annotated sites |

---

## Annotation-by-Annotation Review

All 72 `# nosec` annotations were re-reviewed for this audit, grouped by check.

### B110 — Bare exception swallowing (27 sites)

**Files:** `token_estimation.py`, `agent/_lifecycle.py` (4), `agent/_memory_manager.py` (5), `agent/core.py` (2), `providers/fallback.py` (4), `providers/router.py`, `mcp/client.py` (3), `mcp/multi.py` (2), `orchestration/checkpoint.py`, `orchestration/state.py`, `orchestration/supervisor.py`, `evals/suite.py`, `evals/serve.py`

(One fewer site than the v0.24.0 audit: `observer.py` no longer carries a
`# nosec B110` annotation after the #84–#88 hardening passes.)

**Pattern:** `except Exception:  # nosec B110` inside observer notification loops, best-effort
cleanup paths, provider fallback bookkeeping, and MCP tool discovery.

**Justification:** These are fire-and-forget callbacks and best-effort side effects. An
observer failing to process an event (e.g., a logging handler hitting a network error)
must not crash the agent run that triggered it — this is the documented contract of the
`AgentObserver` protocol. Likewise, a single unavailable MCP server must not prevent other
tools from loading, and a fallback provider's failure accounting must not mask the
underlying provider error.

**Risk:** None. The agent's own error handling is unaffected; only side-effect callbacks
are silenced.

### B608 and SQL construction (19 sites)

**Files:** `checkpoint_postgres.py` (6), `sessions.py`, `knowledge.py`, `knowledge_graph.py`,
`observe/trace_store.py`, `rag/stores/pgvector.py` (8, several as bare `# nosec` with inline
justification), `rag/stores/sqlite.py` (1)

**Pattern:** Table names interpolated into SQL with f-strings
(e.g., `f"SELECT ... FROM {self._table}"`), or `IN (...)` placeholder lists built with
`",".join("?" * n)`.

**Justification:** In every case the interpolated value is either a developer-controlled
constructor argument (e.g., `table_name="my_traces"`) validated at construction time, or a
generated placeholder string. All actual query *parameters* (IDs, metadata, content) are
passed via parameterized placeholders (`?` for SQLite, `%s` for Postgres). Bandit flags
f-strings in SQL regardless of what is interpolated; the annotations suppress the false
positive.

**Risk:** None for standard usage. If you allow untrusted users to specify table names at
construction time, validate the name before passing it to the constructor.

### B101 — Assert usage (7 sites)

**Files:** `serve/api.py` (6), `a2a/server.py` (1)

**Pattern:** `assert x is not None  # narrowed above  # nosec B101`

**Justification:** These asserts are type-narrowing statements for mypy that follow
explicit runtime validation (the request has already been rejected with a 4xx response if
the value is missing). They are unreachable in practice; if Python runs with `-O` the
preceding validation still guards the path.

**Risk:** None.

### B105 — Hardcoded password string (4 sites)

**Files:** `evals/types.py`, `security.py`, `guardrails/pii.py`, `toolbox/slack_tools.py`

**Pattern / Justification:**

- `evals/types.py:16` — `PASS = "pass"` is the string value of the `CaseVerdict.PASS` enum member, not a credential.
- `security.py:151` and `guardrails/pii.py:77` — `compiled.search("a" * 100)` is a ReDoS smoke test that probes user-supplied regex patterns with a benign string before accepting them.
- `toolbox/slack_tools.py:26` — `_MISSING_TOKEN_ERROR` is a user-facing error *message* about a missing token, not a token.

**Risk:** None (all false positives).

### B602 / B603 / B607 / B404 — Subprocess and shell (7 sites)

**Files:** `toolbox/code_tools.py` (4), `serve/cli.py` (3)

**Pattern / Justification:**

- `toolbox/code_tools.py` — the built-in shell/code-execution tool runs commands with
  `shell=True` **by design**; that is the tool's documented purpose. It is opt-in: agents
  only gain it if the developer explicitly adds it to the toolbox, and it should be paired
  with `ToolPolicy` review rules (see threat model below) in any deployment that handles
  untrusted input.
- `serve/cli.py` — the `selectools serve` hot-reload path restarts the CLI by re-executing
  a command reconstructed from `sys.argv`. No untrusted input reaches the command line.

**Risk:** Acknowledged and intentional for the code-execution tool; gate it with
`ToolPolicy` and human-in-the-loop review in production. None for the CLI self-restart.

### B301 / B403 — Pickle usage (2 sites)

**File:** `cache_redis.py`

**Justification:** Data stored in Redis by selectools is always serialized by the same
process (`pickle.dumps`). The cache is a process-local cache that happens to use Redis as
a backend; no external or user-controlled bytes are deserialized.

**Risk:** None in the intended deployment model. If your Redis instance is writable by
untrusted parties, do not use `RedisCache`.

### B324 — MD5 usage (1 site)

**File:** `orchestration/graph.py:139`

**Justification:** Non-cryptographic fingerprint for graph-state loop detection. A
collision would at worst cause the loop detector to miss one repeated state. No
authentication, key derivation, or signature involved.

**Risk:** None.

### B310 — urllib urlopen (1 site)

**File:** `serve/app.py:1053`

**Justification:** Posts eval results to a webhook URL that the operator configures in
`eval_config`. The URL is never derived from request input.

**Risk:** None for standard usage; treat your eval webhook URL as configuration.

### B112 — Bare except with continue (1 site)

**File:** `env.py:26`

**Justification:** The `.env` loader is fail-quiet by design — a malformed line must not
prevent the rest of the file from loading. Matches `python-dotenv` behavior.

**Risk:** None.

### B104 — Bind to 0.0.0.0 (1 site)

**File:** `a2a/server.py:187`

**Justification:** Development-server default, matching `uvicorn`/`flask run` conventions.
`A2AServer` **logs an explicit warning** when started unauthenticated on a
non-loopback host, telling the operator to pass `auth_token=...`.

**Risk:** In production, bind to `127.0.0.1` behind a reverse proxy and always set
`auth_token`.

### B405 / B406 — XML imports (2 sites)

**Files:** `evals/junit.py`, `evals/badge.py`

**Justification:** JUnit XML and SVG badges are **generated**, never parsed — XXE does not
apply. Badge content is XML-escaped via `xml.sax.saxutils.escape()`.

**Risk:** None.

---

## Dependency Audit (pip-audit)

`pip-audit` 2.9.0 was run on 2026-06-11 against the four core production dependencies at
their currently-tested pins (`openai==1.109.1`, `anthropic==0.75.0`, `google-genai==1.47.0`,
`numpy==2.0.2`) including their fully resolved transitive tree:

```text
Found 3 known vulnerabilities in 2 packages
Name     Version ID                  Fix Versions
-------- ------- ------------------- ------------
requests 2.32.5  GHSA-gc5v-m9x4-r6x2 2.33.0
urllib3  2.6.3   PYSEC-2026-142      2.7.0
urllib3  2.6.3   PYSEC-2026-141      2.7.0
```

**Assessment:**

- **No advisories in selectools' directly declared dependencies** (`openai`, `anthropic`,
  `google-genai`, `numpy`).
- `requests` and `urllib3` are **transitive** dependencies pulled in by the provider SDKs.
  The fixed versions are **released on PyPI** (`requests` 2.33.0 on 2026-03-25, latest
  2.34.2; `urllib3` 2.7.0 on 2026-05-07) but both **require Python >= 3.10**. selectools
  supports Python 3.9, where the latest installable releases remain `requests` 2.32.5 and
  `urllib3` 2.6.3 — the table above reflects a Python 3.9 resolution, and the exposure
  stands on 3.9 environments. selectools does not pin or constrain either package; on
  Python >= 3.10, upgrade in place:
  `pip install -U "requests>=2.33.0" "urllib3>=2.7.0"`.

**Process:** dependency audits are run before every release tag with:

```bash
pip install pip-audit
pip-audit  # in the deployment environment, audits the resolved tree
```

---

## SBOM

[`sbom.json`](https://github.com/johnnichev/selectools/blob/main/sbom.json) was regenerated
for this audit (root component `selectools@0.25.0`, CycloneDX spec 1.6, 4 core production
dependencies). `requirements` mode against the four pinned core dependencies plus
`--pyproject` for the root-component metadata is the canonical method — `environment`
mode would inventory the whole venv (90+ components, no selectools root) and is the
wrong shape for this repo. To regenerate:

```bash
pip install cyclonedx-bom
pip freeze | grep -E "^(openai|anthropic|google-genai|numpy)==" > /tmp/core-reqs.txt
cyclonedx-py requirements /tmp/core-reqs.txt --pyproject pyproject.toml \
  --spec-version 1.6 --output-format JSON -o sbom.json
```

---

## Threat Model — Defensive Layers

selectools ships several independent layers that mitigate the main risks of running
LLM agents with tools. Each is summarized here with the module that implements it.

### Guardrails (`selectools.guardrails`)

`GuardrailsPipeline` composes input/output validators that run before the LLM sees user
input and after it produces output: PII detection/redaction (`pii.py`, with a ReDoS smoke
test on any custom patterns), topic blocking (`topic.py`), toxicity screening
(`toxicity.py`), and length/format enforcement (`length.py`, `format.py`). Guardrails are
the first line against both data leakage (PII out) and prompt-level abuse (blocked topics
in).

### Tool policy engine (`selectools.policy`)

`ToolPolicy` provides declarative **allow / review / deny** rules matched by glob patterns
against tool names (e.g. `deny=["delete_*"]`, `review=["send_*"]`). Evaluation order is
deny → review → allow, and the **default for an unmatched tool is `review`**, so new tools
are never silently auto-executed. Every decision is surfaced to observers via
`on_policy_decision`.

### Approval gates / human-in-the-loop (`selectools.pending`)

Tools can return a `PendingConfirmation` instead of executing immediately; the framework
materializes it as a `PendingAction` with an args digest (`compute_args_digest`) and an
expiry. The action only executes after explicit confirmation, and a changed-args digest or
an expired action is rejected — preventing both stale approvals and confirm-then-mutate
races. Combined with `ToolPolicy` review rules, this is the HITL gate for destructive
operations.

### Prompt-injection screening and sanitizers (`selectools.security`, `selectools.knowledge_sanitizers`)

`security.py` screens **tool outputs** after execution against 15+ built-in injection
patterns (role-override, jailbreak, instruction-smuggling), with Unicode normalization to
defeat homoglyph bypasses; custom patterns can be added via
`AgentConfig(output_screening_patterns=[...])`. `knowledge_sanitizers.py` defangs
delimiter sequences, strips surrogates, and rejects near-duplicates before retrieved
content enters memory. Separately, all LLM-judge prompts in the eval framework and the
coherence checker fence user content with `<<<BEGIN_USER_CONTENT>>>` /
`<<<END_USER_CONTENT>>>` delimiters so test inputs cannot hijack scoring.

### AgentAPI authentication and isolation (`selectools.serve.api`)

`AgentAPI` supports bearer-token auth (`auth_key=...`): every route except `/v1/health`
requires `Authorization: Bearer <key>` and fails closed with a structured 401. Sessions
are persisted per user through the `SessionStore` protocol, and request handling validates
and narrows all inputs before any agent work happens. Errors are returned in a fixed
`{"error": {"message", "type"}}` envelope rather than raw tracebacks.

### A2A server hardening (`selectools.a2a.server`)

`A2AServer` supports bearer-token auth on `POST /a2a` (the Agent Card discovery route is
intentionally public), enforces a request body size limit, and **logs a startup warning
when running unauthenticated on a non-loopback host**. Failed agent executions return only
the exception **type** to the remote caller — exception messages can contain internal URLs
and paths, so full detail goes to server logs only. Each task runs on an isolated agent
clone (`_clone_for_isolation`, public as `clone_for_isolation` at v1.0) so concurrent
remote tasks cannot share memory or race provider context.

### Path traversal mitigations

File-backed stores (`BaselineStore`, `SnapshotStore`, `EvalHistory`, `FileKnowledgeStore`,
`JsonFileSessionStore`) sanitize user-controlled names with `Path(name).name` before
constructing file paths, so a session ID of `../../etc/passwd` reduces to `passwd`.

---

## Vulnerability Reporting

To report a security vulnerability privately:

- **Email:** support@nichevlabs.com
- **Response SLA:** 48 hours acknowledgement, 7 days for triage
- **Disclosure:** Coordinated — we work with you on a fix before public announcement

Please do not open a public GitHub issue for security vulnerabilities. Full policy:
[SECURITY.md](https://github.com/johnnichev/selectools/blob/main/SECURITY.md).
