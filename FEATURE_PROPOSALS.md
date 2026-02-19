# Selectools — Feature Proposals

> Improvement proposals for production safety, memory persistence, and developer experience.
> These items complement the existing [ROADMAP.md](./ROADMAP.md) and fill identified gaps.

---

## P0 — Critical Safety & Correctness

### 1. Tool Policy Engine (allow / review / deny)

**Problem:** Selectools currently trusts all LLM tool-call decisions unconditionally. In production, certain tools (e.g., `delete_record`, `send_email`) should require explicit permission or be outright blocked depending on context.

**Proposal:** A declarative policy system that classifies every tool call before execution.

```yaml
# policies/default.yaml
allow:  [search_*, read_*, get_*]
review: [send_*, create_*, update_*]
deny:   [delete_*, drop_*]

# Tools matching "review" that can skip human confirmation
auto_approve:
  - create_draft
  - update_preferences
```

**Implementation sketch:**

```python
from selectools import ToolPolicy

policy = ToolPolicy.from_yaml("policies/default.yaml")
# or inline:
policy = ToolPolicy(
    allow=["search_*", "read_*"],
    review=["send_*", "create_*"],
    deny=["delete_*"],
    auto_approve=["create_draft"],
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(tool_policy=policy),
)
```

**Evaluation order:** deny → review → allow → unknown defaults to review.

**Conditional rules** for argument-level control:

```yaml
deny_when:
  - tool: send_email
    arg: to
    pattern: "*@external.com"

review_when:
  - tool: upload_file
    arg: visibility
    pattern: "public"
```

**Touches:** `agent/core.py` (agent loop), new `policy.py` module, `AgentConfig`.

---

### 2. Human-in-the-Loop Tool Approval

**Problem:** Even with a policy engine, there's no mechanism to pause execution and ask the user "Are you sure you want to call `send_email(to='ceo@company.com')`?" before proceeding.

**Proposal:** A confirmation callback that the agent loop invokes for tools flagged as `review`.

```python
def my_confirmation_handler(tool_name: str, tool_args: dict, reason: str) -> bool:
    print(f"⚠ Tool '{tool_name}' requires approval: {reason}")
    print(f"  Args: {tool_args}")
    return input("Approve? (y/n): ").lower() == "y"

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(
        tool_policy=policy,
        confirm_action=my_confirmation_handler,
    ),
)
```

**Async variant** for web/API contexts:

```python
config = AgentConfig(
    tool_policy=policy,
    confirm_action=async_confirm_handler,  # async (name, args, reason) -> bool
    approval_timeout=60,                   # seconds; deny on timeout
)
```

**Agent loop behaviour:**
- `allow` → execute immediately
- `review` + `auto_approve` match → execute immediately
- `review` + `confirm_action` provided → call callback, execute if approved
- `review` + no callback → deny with error message to LLM
- `deny` → return error to LLM, never execute

**Touches:** `agent/core.py`, `AgentConfig`, new `approvals.py` module.

---

### 3. Tool-Pair-Aware Conversation Trimming

**Problem:** `ConversationMemory._enforce_limits()` uses naive sliding-window trimming. If the cut happens between an assistant `tool_use` message and its corresponding user `tool_result` message, the LLM receives a malformed conversation that violates provider API contracts.

**Current behaviour:**

```
[user] "Delete file X"
[assistant] tool_use: delete_file(path="X")   ← might get trimmed here
[user] tool_result: "File deleted"             ← orphaned result
[assistant] "Done, I deleted file X"
```

**Proposal:** Replace naive trimming with tool-pair-aware trimming that:
1. Never splits an assistant `tool_use` from its matching `tool_result`
2. Ensures the trimmed conversation always starts with a valid user text message
3. Preserves at least one complete exchange

```python
class ConversationMemory:
    def _enforce_limits(self) -> None:
        # After naive trim, scan forward to find the first safe boundary
        # (a user text message that isn't a tool_result)
        ...
```

**Touches:** `memory.py` only — small, self-contained fix.

---

## P1 — Persistent Memory & Sessions

### 4. Persistent Conversation Sessions

**Problem:** `ConversationMemory` is in-memory only. If the process restarts, all conversation history is lost. Users building chat applications need sessions that survive restarts.

**Proposal:** A `SessionStore` protocol with pluggable backends.

```python
from selectools.memory import SessionStore, JsonFileSessionStore

store = JsonFileSessionStore(directory="./sessions")
# or: SqliteSessionStore(path="sessions.db")
# or: RedisSessionStore(url="redis://localhost")

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(session_store=store, session_id="user-123"),
)

# Sessions auto-persist after each turn
result = agent.run("What was my last question?")

# Session management
store.list_sessions()                     # → [SessionInfo(...), ...]
store.resume_session("user-123", "abc")   # switch active session
store.new_session("user-123")             # start fresh, archive old
```

**Session features:**
- Auto-save after each turn
- TTL-based expiry (configurable)
- Tool-pair-preserving trim on load (see P0.3)
- Title auto-generated from first user message

**Touches:** New `sessions.py` module, `AgentConfig`, `agent/core.py` integration.

---

### 5. Cross-Session Knowledge Memory

**Problem:** Even with persistent sessions, each session is isolated. There's no way for an agent to "remember" facts across conversations (e.g., user preferences, prior decisions).

**Proposal:** A file-based or DB-backed knowledge memory with two layers:

| Layer | Purpose | Persistence |
|-------|---------|-------------|
| **Daily log** | Append-only entries from the current day | `memory/YYYY-MM-DD.md` |
| **Long-term** | Curated facts that persist indefinitely | `MEMORY.md` or DB table |

```python
from selectools.memory import KnowledgeMemory

knowledge = KnowledgeMemory(
    directory="./workspace",
    recent_days=2,           # inject last 2 days into system prompt
    max_context_chars=5000,  # cap memory injection size
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(knowledge_memory=knowledge),
)
```

**Built-in `remember` tool** that the agent can call:

```python
@tool
def remember(text: str, category: str = "general") -> str:
    """Save a piece of information to memory for future reference."""
    return knowledge.remember(text, category)
    # Appends: "- [14:32] **general**: User prefers dark mode"
```

**System prompt auto-injection:**

```
You have access to the following memories from recent conversations:

# Memory — 2026-02-15
- [09:15] **preference**: User prefers metric units
- [14:32] **context**: User is working on Project Alpha

# Long-Term Memory
- User's timezone is Europe/Oslo
- Preferred communication style: concise, no emojis
```

**Touches:** New `knowledge.py` module, `PromptBuilder` integration, built-in tool in `toolbox/`.

---

### 6. Summarize-on-Trim

**Problem:** When conversation history exceeds limits, old messages are silently dropped. Important context from early in the conversation is lost.

**Proposal:** Before trimming, summarize the messages being removed and inject the summary as a system-level context message.

```python
memory = ConversationMemory(
    max_messages=30,
    summarize_on_trim=True,
    summarize_provider=provider,  # uses same LLM to summarize
)
```

**Behaviour:**
1. When `len(messages) > max_messages`, take the oldest N messages to be trimmed
2. Call the provider with: "Summarize this conversation so far in 2-3 sentences"
3. Replace trimmed messages with a single `[system]` message containing the summary
4. Keep the most recent messages intact

**Touches:** `memory.py`, provider integration for summarization.

---

## P2 — Security Hardening

### 7. Tool Output Screening

**Problem:** Tools that return untrusted content (web scraping, email reading, file parsing) can contain prompt injection payloads that hijack the agent's next action.

**Proposal:** An optional `classify_output=True` flag on tools that screens results before feeding them back to the LLM.

```python
@tool(classify_output=True)
def fetch_webpage(url: str) -> str:
    """Fetch and return content from a URL."""
    ...
```

**Screening options:**
- Pattern-based: regex check for common injection phrases
- Classifier-based: lightweight local model (optional dependency)
- LLM-based: secondary call to a fast model (e.g., Haiku)

**When blocked:** The tool result is replaced with a safe message:
> "Output from 'fetch_webpage' was blocked by the security classifier. The content may contain prompt injection."

**Touches:** `agent/core.py` (post-tool-execution hook), new `security.py` module, `Tool` class.

---

### 8. Coherence Checking

**Problem:** Prompt injection in tool outputs can cause the agent to call completely unrelated tools. A user asks "summarize my emails" and an injected email body causes the agent to call `send_email` instead.

**Proposal:** An optional LLM-based check that verifies tool calls match the user's original intent.

```python
config = AgentConfig(
    coherence_check=True,           # enable coherence checking
    coherence_model="haiku",        # fast model for checking
)
```

**Flow:**
1. User says: "Summarize my inbox"
2. Agent calls `read_email` → returns content with injection
3. Agent wants to call `send_email` next
4. **Coherence check**: "Does calling `send_email` match the user's request to 'summarize my inbox'?" → **No** → Block

**Touches:** `agent/core.py`, `AgentConfig`, uses existing provider infrastructure.

---

### 9. Privacy-Preserving Audit Logging

**Problem:** No built-in way to audit what tools were called, with what arguments, and whether they succeeded. Important for compliance, debugging, and cost analysis.

**Proposal:** JSONL-based append-only audit log with privacy controls.

```python
from selectools import AuditLogger

audit = AuditLogger(
    path="agent_audit.jsonl",
    hash_inputs=True,        # SHA-256 hash of inputs (never raw text)
    log_arg_keys=True,       # log argument names but not values
    log_timing=True,         # execution duration per tool
    rotate_daily=True,       # one file per day
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(audit_logger=audit),
)
```

**Log entry format:**

```json
{
  "ts": "2026-02-15T14:32:01Z",
  "event": "tool_execution",
  "tool": "search_database",
  "arg_keys": ["query", "limit"],
  "input_hash": "a1b2c3...",
  "success": true,
  "duration_ms": 142,
  "output_length": 1523
}
```

**Queryable:** Integrate with existing `AgentAnalytics` or provide a standalone query API.

**Touches:** New `audit.py` module, hooks into `agent/core.py` via existing observability callbacks.

---

## P3 — Developer Experience

### 10. Declarative Tool Configuration (YAML)

**Problem:** Tool definitions are currently code-only (`@tool` decorator). Non-developers or ops teams may want to configure available tools without editing Python.

**Proposal:** Optional YAML-based tool registration alongside the existing `@tool` decorator.

```yaml
# tools.yaml
tools:
  search_docs:
    function: myapp.tools.search_docs
    description: "Search internal documentation"
    parameters:
      query:
        type: string
        required: true
      limit:
        type: integer
        default: 10

  send_notification:
    function: myapp.tools.send_notification
    requires_approval: true
    timeout: 30
    description: "Send a push notification"
    parameters:
      user_id:
        type: string
        required: true
      message:
        type: string
        required: true
```

```python
registry = ToolRegistry.from_yaml("tools.yaml")
agent = Agent(tools=registry.tools, provider=provider)
```

**Touches:** `tools/registry.py`, new YAML loader.

---

## Implementation Priority Matrix

| # | Feature | Effort | Impact | Dependencies |
|---|---------|--------|--------|-------------|
| 3 | Tool-pair-aware trimming | Small | High | None (bug fix) |
| 1 | Tool Policy Engine | Medium | High | None |
| 2 | Human-in-the-loop approval | Medium | High | #1 (policy engine) |
| 4 | Persistent sessions | Medium | High | #3 (safe trimming) |
| 6 | Summarize-on-trim | Small | Medium | None |
| 5 | Cross-session knowledge memory | Medium | Medium | #4 (sessions) |
| 9 | Audit logging | Small | Medium | None |
| 7 | Tool output screening | Medium | Medium | None |
| 8 | Coherence checking | Medium | Medium | #7 (screening infra) |
| 10 | YAML tool configuration | Small | Low | None |

**Suggested implementation order:**

```
Phase 1 (v0.13.0): #3 → Structured output → Fallback providers → Batch → #1 → #2
                    (Routing infrastructure + safety foundation)

Phase 2 (v0.14.0): Multi-agent graphs → handoffs → shared state → supervisor
                    (Orchestration — the full classify-then-delegate pattern)

Phase 3 (v0.15.0): MCP client → MCP server → FastAPI/Flask integrations
                    (Ecosystem interoperability)

Phase 4 (v0.16.0): #4 → #6 → #5 + entity/KG memory
                    (Memory persistence & advanced memory types)

Phase 5 (v1.0.0):  Guardrails → #9 → #7 → #8 → #10
                    (Enterprise readiness)
```
