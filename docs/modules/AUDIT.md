---
description: "JSONL audit logging with 4 privacy levels and daily file rotation"
tags:
  - security
  - audit
---

# Audit Logging

**Added in:** v0.15.0

`AuditLogger` provides a JSONL append-only audit trail for every agent action. It records tool calls, LLM responses, policy decisions, and errors — all with configurable privacy controls and daily file rotation.

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.audit import AuditLogger, PrivacyLevel

@tool(description="Search the knowledge base")
def search(query: str) -> str:
    return f"Results for: {query}"

audit = AuditLogger(
    log_dir="./audit",
    privacy=PrivacyLevel.KEYS_ONLY,   # redact argument values
    daily_rotation=True,               # audit-2026-03-12.jsonl
)

agent = Agent(
    tools=[search],
    provider=OpenAIProvider(),
    config=AgentConfig(observers=[audit]),
)

result = agent.ask("Find articles about Python")
# Check ./audit/audit-2026-03-12.jsonl
```

Every event is one JSON line:

```json
{"event":"run_start","run_id":"abc123","message_count":1,"ts":"2026-03-12T18:30:00.000000+00:00"}
{"event":"tool_start","run_id":"abc123","call_id":"xyz","tool_name":"search","tool_args":{"query":"<redacted>"},"ts":"..."}
{"event":"tool_end","run_id":"abc123","call_id":"xyz","tool_name":"search","duration_ms":42.5,"success":true,"ts":"..."}
{"event":"llm_end","run_id":"abc123","model":"gpt-4o","prompt_tokens":150,"completion_tokens":50,"cost_usd":0.001,"ts":"..."}
{"event":"run_end","run_id":"abc123","iterations":2,"tool_name":"search","ts":"..."}
```

---

## Privacy Levels

Control how sensitive data appears in audit logs:

| Level | Behaviour | Example `{"query": "secret data"}` |
|---|---|---|
| `PrivacyLevel.FULL` | Log everything verbatim | `{"query": "secret data"}` |
| `PrivacyLevel.KEYS_ONLY` | Redact values | `{"query": "<redacted>"}` |
| `PrivacyLevel.HASHED` | SHA-256 hash (truncated) | `{"query": "2bb80d537b1da3..."}` |
| `PrivacyLevel.NONE` | Omit arguments entirely | `{}` |

```python
# Full logging (development)
AuditLogger(privacy=PrivacyLevel.FULL)

# Keys only (production default)
AuditLogger(privacy=PrivacyLevel.KEYS_ONLY)

# Hashed (compliance — can verify without exposing)
AuditLogger(privacy=PrivacyLevel.HASHED)

# No args (strictest privacy)
AuditLogger(privacy=PrivacyLevel.NONE)
```

---

## File Rotation

```python
# Daily rotation (default) — audit-2026-03-12.jsonl, audit-2026-03-13.jsonl, ...
AuditLogger(log_dir="./audit", daily_rotation=True)

# Single file — audit.jsonl
AuditLogger(log_dir="./audit", daily_rotation=False)
```

---

## Recorded Events

| Event | When | Key Fields |
|---|---|---|
| `run_start` | Agent starts processing | `run_id`, `message_count` |
| `run_end` | Agent finishes | `run_id`, `iterations`, `tool_name`, `total_cost_usd` |
| `tool_start` | Before tool execution | `run_id`, `call_id`, `tool_name`, `tool_args` |
| `tool_end` | After successful tool | `run_id`, `call_id`, `tool_name`, `duration_ms`, `success` |
| `tool_error` | Tool raised exception | `run_id`, `tool_name`, `error`, `error_type`, `duration_ms` |
| `llm_end` | After LLM response | `run_id`, `model`, `prompt_tokens`, `cost_usd` |
| `policy_decision` | Policy evaluated tool | `run_id`, `tool_name`, `decision`, `reason` |
| `error` | Unrecoverable error | `run_id`, `error`, `error_type` |

---

## Include LLM Response Content

By default, response content is **not** logged (privacy). Opt in:

```python
AuditLogger(include_content=True)
# llm_end events will include "response_length": 250
# tool_end events will include "result_length": 100
```

---

## Combining with Other Observers

`AuditLogger` is just an `AgentObserver` — combine it with others:

```python
from selectools.observer import LoggingObserver

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(
        observers=[
            AuditLogger(log_dir="./audit"),     # JSONL file
            LoggingObserver(),                   # Python logging
        ],
    ),
)
```

---

## Thread Safety

`AuditLogger` uses a `threading.Lock` for file writes, making it safe for concurrent `batch()` usage.

---

## API Reference

| Class / Enum | Description |
|---|---|
| `AuditLogger(log_dir, privacy, daily_rotation, include_content)` | JSONL audit logger (implements `AgentObserver`) |
| `PrivacyLevel.FULL` | Log all values |
| `PrivacyLevel.KEYS_ONLY` | Redact values to `"<redacted>"` |
| `PrivacyLevel.HASHED` | SHA-256 hash of values |
| `PrivacyLevel.NONE` | Omit tool_args entirely |
