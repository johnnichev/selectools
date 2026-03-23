---
name: bug-hunt
description: Deploy parallel QA agents to hunt for bugs across selectools. Each agent audits a different subsystem. Use for pre-release quality gates or periodic sweeps.
argument-hint: [scope] e.g. "all", "agent", "providers", "rag", "tools", "memory", "evals", "security"
---

# Bug Hunt

Deploy parallel QA sub-agents to find bugs. Scope: **$ARGUMENTS**

If no scope, default to "all".

## Resolve Scope

- `all` → deploy 7 agents (one per subsystem below)
- `agent` → 1 agent: core loop, config, mixins, streaming, batch
- `providers` → 1 agent: all 5 providers, fallback, circuit breaker, _openai_compat
- `tools` → 1 agent: tool system, decorators, loader, registry, toolbox, MCP bridge
- `rag` → 1 agent: loaders, chunking, hybrid search, BM25, reranker, vector stores, embeddings
- `memory` → 1 agent: ConversationMemory, sessions, entity memory, knowledge graph, knowledge stores
- `evals` → 1 agent: evaluators, suite, report, regression, pairwise, snapshot, badges, CLI
- `security` → 1 agent: guardrails, audit, screening, coherence, policy, injection patterns

## Agent Template

Each agent should read the source files for its subsystem, then look for:

### 1. Correctness Bugs
- **Type mismatches** — function signatures don't match callers (e.g. missing `tools=None`)
- **Async/sync inconsistency** — `arun()`/`astream()` missing features that `run()` has (pitfall #12)
- **None handling** — `response_msg.content` used without `or ""` guard (pitfall #7)
- **Race conditions** — shared mutable state in `batch()`/`abatch()` without locks
- **Resource leaks** — `ThreadPoolExecutor()` created per call instead of shared (known bug in tools/base.py)

### 2. API Contract Violations
- **Provider protocol** — `stream()`/`astream()` not passing `tools` parameter (pitfall #1)
- **ToolCall stringification** — streaming paths converting ToolCall objects to strings (pitfall #2)
- **Observer events** — missing `run_id` in observer calls, or events not firing in all 3 loop methods
- **StepType consistency** — trace steps using string literals instead of `StepType.ENUM_NAME`
- **_effective_model** — any remaining `self.config.model` in `_provider_caller.py` (should all be `self._effective_model`)

### 3. Security Issues
- **SQL injection** — raw f-strings in SQLite queries (knowledge stores, sessions, checkpoints)
- **Path traversal** — user-controlled paths in file operations (knowledge directory, sessions)
- **Prompt injection** — tool outputs not screened when `screen_tool_output=True`
- **IDOR** — session IDs guessable or not validated
- **Missing input validation** — AgentConfig fields accepting invalid values silently

### 4. Memory & Performance
- **Unbounded growth** — lists/dicts that grow without limits (trace steps, tool history)
- **Deep copy overhead** — unnecessary `copy.deepcopy()` on large objects
- **Blocking in async** — sync I/O calls in `arun()`/`astream()` paths (file reads, SQLite)
- **Import-time side effects** — heavy initialization at import (should be lazy)

### 5. Edge Cases
- **Empty inputs** — agent with 0 messages, 0 tools (should it error?)
- **Max iterations = 0** — does the loop handle this?
- **Budget = 0** — does budget check fire immediately?
- **Cancelled before start** — pre-cancelled token behavior
- **Provider returns empty** — empty string content, None tool_calls
- **Concurrent tool execution** — parallel tools modifying shared state

### 6. Documentation Drift
- **Docstrings vs behavior** — function does something different than documented
- **Type hints** — `Optional` fields that are never actually None, or vice versa
- **Deprecated code** — hooks dict still referenced somewhere other than `_HooksAdapter`

## Output Format

Each agent reports findings as:

```
## [Subsystem] Bug Report

### CRITICAL (breaks core functionality or data loss)
| # | File | Line | Bug | Suggested Fix |
|---|------|------|-----|---------------|

### HIGH (incorrect behavior, security issue)
| # | File | Line | Bug | Suggested Fix |
|---|------|------|-----|---------------|

### MEDIUM (edge case, performance, DX issue)
| # | File | Line | Bug | Suggested Fix |
|---|------|------|-----|---------------|

### LOW (style, minor inconsistency)
| # | File | Line | Bug | Suggested Fix |
|---|------|------|-----|---------------|
```

## After All Agents Complete

Compile a master bug report:
1. Deduplicate findings across subsystems
2. Sort by severity (Critical > High > Medium > Low)
3. Count totals by subsystem and severity
4. Identify patterns (e.g., "async/sync parity issues found in 4/7 subsystems")
5. Cross-reference with CLAUDE.md "Common Pitfalls" — are there new pitfalls to add?
6. Present the report to the user

Ask the user if they want to proceed with fixes. Do NOT auto-fix or auto-commit.
