---
name: ralph-bug-hunt
description: Autonomous hunt-and-fix loop for a single selectools module. Finds bugs, auto-applies fixes, writes regression tests, verifies with pytest. Outputs RALPH_RESULT sentinel on the last line so the orchestration script can detect convergence.
argument-hint: [module] e.g. "agent", "providers", "tools", "rag", "memory", "evals", "security"
---

# Ralph Bug Hunt

Autonomous bug hunt + auto-fix loop for module: **$ARGUMENTS**

If no module is given, default to "rag".

## What makes this different from /bug-hunt

1. **Every finding is fixed immediately** — no "ask user" step.
2. **The last line of output is always a machine-parseable sentinel** so the
   orchestration script (`scripts/ralph_bug_hunt.sh`) can detect convergence.

---

## Step 1 — Scope Resolution

Map the module argument to its source paths:

| Module     | Source paths                                                                                  |
|------------|-----------------------------------------------------------------------------------------------|
| `agent`    | `src/selectools/agent/`                                                                       |
| `providers`| `src/selectools/providers/`                                                                   |
| `tools`    | `src/selectools/tools/`, `src/selectools/toolbox/`                                            |
| `rag`      | `src/selectools/rag/`                                                                         |
| `memory`   | `src/selectools/memory.py`, `src/selectools/entity_memory.py`, `src/selectools/knowledge*.py`, `src/selectools/sessions.py` |
| `evals`    | `src/selectools/evals/`                                                                       |
| `security` | `src/selectools/guardrails/`, `src/selectools/audit.py`, `src/selectools/security.py`, `src/selectools/coherence.py`, `src/selectools/policy.py` |

---

## Step 2 — Bug Hunt Analysis

Read all source files for the module. Hunt for bugs in these 6 categories:

### Category 1 — Correctness
- Type mismatches: function signatures don't match callers
- Async/sync inconsistency: `arun()`/`astream()` missing features that `run()` has
- None handling: using fields without `or ""` / `or []` guards
- Race conditions: shared mutable state without locks in concurrent paths
- Resource leaks: executors, file handles, DB connections created per-call

### Category 2 — API Contract
- Provider protocol: `stream()`/`astream()` not passing `tools` parameter
- ToolCall stringification in streaming paths
- Observer events missing `run_id` or not firing in all execution paths
- StepType using string literals instead of `StepType.ENUM_NAME`

### Category 3 — Security
- SQL injection: raw f-strings in SQLite queries
- Path traversal: user-controlled paths in file operations (test with `../../etc/passwd`)
- Prompt injection in evaluators: `case.input`/`case.reference` interpolated into LLM judge prompts without fencing
- IDOR: session IDs guessable or not validated

### Category 4 — Memory & Performance
- Unbounded growth: lists/dicts that grow without limits
- Non-atomic file writes: `path.write_text()` without `.tmp` → `os.replace()` pattern
- Blocking sync I/O in async paths
- `ThreadPoolExecutor()` created per-call instead of module singleton

### Category 5 — Edge Cases
- Empty inputs: 0 messages, 0 tools, empty strings, zero-length lists
- Zero/negative numeric parameters: max_iterations=0, budget=0, top_k=0
- Provider returns empty: empty string content, None tool_calls
- Concurrent tool execution modifying shared state

### Category 6 — Documentation Drift
- Docstrings describing behavior the code doesn't implement
- Type hints claiming `Optional` but value is never actually None (or vice versa)

---

## Step 3 — Fix Each Finding

For each bug found (Critical first, then High, Medium, Low):

1. **Read the affected file** in full.
2. **Apply the fix** using the Edit tool.
3. **Write a regression test** in the appropriate test file:
   - `tests/<module>/test_<bug_description>_regression.py`, or append to an
     existing regression file like `tests/rag/test_rag_regression_phase3.py`.
4. **Verify the fix** by running pytest on just that test file:
   ```
   pytest <test_file> -x -q
   ```
   - If the test passes → record as FIXED.
   - If the test fails → revert the code change (restore the original with Edit),
     leave the test commented out with a `# UNFIXED:` prefix, and record as UNFIXED.

---

## Step 4 — Suite Verification

After all individual fixes are applied and verified, run the full suite (minus e2e):

```
pytest tests/ -k "not e2e" -x -q
```

If the suite fails:
- Identify which fix broke the suite.
- Revert that specific fix.
- Re-run the suite and confirm it passes.
- Reclassify that finding as UNFIXED.

---

## Step 5 — Emit Sentinel (MANDATORY — this must be the LAST line of output)

Count findings and fixed vs unfixed:

If zero findings OR all findings were UNFIXED (meaning no net code change that broke anything):

```
RALPH_RESULT: CLEAN
```

If any findings were FIXED:

```
RALPH_RESULT: FOUND <N> (CRITICAL:<c> HIGH:<h> MEDIUM:<m> LOW:<l>) FIXED:<f>
```

Where:
- `<N>` = total findings
- `<c>`, `<h>`, `<m>`, `<l>` = count per severity
- `<f>` = number successfully fixed (test passed + suite passed)

**IMPORTANT**: The sentinel line must be the very last line printed. Do not add
any text after it. The orchestration script uses `tail -1` to detect convergence.

---

## False Positive Handling

Before fixing, check if an existing test explicitly asserts the current behavior.
A passing test that validates the "buggy" code is evidence the finding may be
wrong — investigate before changing.

## Why These Categories Catch What Tests Miss

- **Thread-safety**: tests run sequentially with mocks — races are invisible
- **Injection/path traversal**: tests use normal inputs — adversarial paths never tried
- **Non-atomic writes**: tests don't simulate process crashes
- **None content**: mocks return valid data — `None` content never exercised
