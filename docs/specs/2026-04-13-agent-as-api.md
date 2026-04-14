# Agent-as-API (Serve Enhancement)

**Stack:** Python 3.9+, src-layout, pytest, starlette (optional dep), stdlib HTTP fallback
**Date:** 2026-04-13
**Status:** Draft

## Problem

Selectools ships a working `selectools serve` with `/invoke`, `/stream`, `/health`, and `/schema` endpoints — plus a visual builder and playground. But the serve module lacks three things every production deployment needs and every competitor (Agno's `AgentOS`, PraisonAI's agent-as-API) already provides:

1. **Per-user session isolation** — all requests share a single agent instance and conversation history. User A's messages leak into User B's context.
2. **API-key authentication for the invoke endpoint** — current auth is cookie-based builder auth only. There's no way for programmatic API clients to authenticate with a bearer token or API key.
3. **Configurable streaming** — SSE works but there's no heartbeat keepalive, no configurable chunk batching, and no per-event typing (tool calls vs. text chunks vs. final result aren't typed in the SSE stream).

## Solution

Enhance the existing `serve/` module (not rewrite) with three targeted additions:

### 1. Session Manager

A pluggable `SessionManager` that maps `(user_id, session_id)` → isolated agent state (conversation history, entity memory, knowledge memory). Default: in-memory with TTL-based eviction. Optional: file-backed or Redis-backed (via protocol, no hard dep).

### 2. API-Key Auth Middleware

A middleware layer (works with both stdlib and Starlette) that validates `Authorization: Bearer <key>` headers against a configurable key store. Extracts `user_id` from the key for session routing. Falls back to existing cookie auth for browser-based playground/builder.

### 3. Typed SSE Events

Enrich the existing SSE stream with typed event categories and a heartbeat:

```
event: chunk
data: {"content": "Hello", "type": "text"}

event: tool_call
data: {"tool": "search", "args": {"q": "weather"}, "call_id": "tc_1"}

event: tool_result
data: {"tool": "search", "call_id": "tc_1", "result": "Sunny, 72°F"}

event: done
data: {"content": "The weather is sunny.", "iterations": 2, "usage": {...}}

: heartbeat
```

## Acceptance Criteria

### Session Isolation
- [ ] New `SessionManager` protocol in `serve/sessions.py` with `get(user_id, session_id) -> SessionState` and `put(user_id, session_id, state) -> None` and `delete(user_id, session_id) -> None`
- [ ] `SessionState` holds: `ConversationMemory`, `Optional[EntityMemory]`, `Optional[KnowledgeMemory]`, `created_at`, `last_accessed`, `metadata: Dict`
- [ ] `InMemorySessionManager` implementation with configurable `max_sessions` (default 1000) and `ttl_seconds` (default 3600)
- [ ] Session ID from request: `X-Session-ID` header or `session_id` field in JSON body; auto-generated UUID if absent
- [ ] User ID extracted from auth (API key → user_id mapping) or `X-User-ID` header for trusted internal networks
- [ ] Each `/invoke` and `/stream` request clones the agent's base config, injects the session's memory, runs, then saves state back
- [ ] Sessions are isolated — User A's history never appears in User B's context
- [ ] Stale sessions evicted on `get()` check (lazy eviction), plus periodic sweep every 60s in Starlette mode

### API-Key Authentication
- [ ] `serve/auth.py` with `ApiKeyAuth` class: validates `Authorization: Bearer <key>` headers
- [ ] Key store: `Dict[str, KeyInfo]` where `KeyInfo` has `user_id`, `role`, `rate_limit`, `created_at`
- [ ] Keys loadable from: `~/.selectools/api_keys.json` file, `SELECTOOLS_API_KEYS` env var (JSON), or passed programmatically
- [ ] New CLI flag: `--api-key <key>` for single-key quick-start (maps to user_id `"default"`)
- [ ] Existing cookie auth continues to work for playground/builder — API key auth is additive
- [ ] Unauthenticated requests to protected endpoints return `401` with `{"error": "unauthorized", "message": "..."}`
- [ ] Invalid key returns `403` with `{"error": "forbidden", "message": "..."}`

### Typed SSE Streaming
- [ ] SSE events use `event:` field to categorize: `chunk`, `tool_call`, `tool_result`, `error`, `done`
- [ ] `chunk` events include `{"content": str, "type": "text"|"reasoning"}`
- [ ] `tool_call` events include `{"tool": str, "args": dict, "call_id": str}`
- [ ] `tool_result` events include `{"tool": str, "call_id": str, "result": str}`
- [ ] `done` event includes `{"content": str, "iterations": int, "usage": dict}`
- [ ] Heartbeat comment (`: heartbeat\n\n`) sent every 15s (configurable) to keep connection alive
- [ ] Backward compatible: clients that ignore `event:` field and only read `data:` still work

### Integration
- [ ] `AgentRouter` constructor gains optional `session_manager` and `auth` params
- [ ] `AgentServer` constructor gains optional `session_manager`, `auth`, and `api_keys` params
- [ ] `selectools serve` CLI gains `--api-key` flag
- [ ] Starlette app (`create_builder_app`) gains session + auth support when available
- [ ] `/health` endpoint enhanced: includes `active_sessions` count and `uptime_seconds`

### General
- [ ] No new required dependencies — `starlette`/`uvicorn` remain optional
- [ ] Stdlib HTTP server supports all features (sessions, API key auth, typed SSE)
- [ ] ≥90% test coverage on new code
- [ ] Stability marker: `@beta` on all new public classes
- [ ] Existing serve tests continue to pass with no modifications

## Non-Goals

- **WebSocket support** — SSE is sufficient for server→client streaming. WebSocket adds bidirectional complexity not needed for agent responses.
- **JWT/OAuth2 implementation** — API key auth is the scope. Users needing JWT can wrap the server in a reverse proxy (nginx, Caddy) or implement a custom `AuthProvider`.
- **Redis/database session backend** — define the protocol, ship in-memory only. Redis adapter is a future add-on.
- **Rate limiting** — the `KeyInfo` dataclass will carry a `rate_limit` field for future use, but enforcement is not in v0.22.0 scope.
- **Multi-agent routing** — one agent per server instance. Multi-agent orchestration happens at the `AgentGraph` level, not the HTTP level.
- **Graceful shutdown / drain** — stdlib HTTP server doesn't support it. Starlette/uvicorn handles this natively.
- **OpenTelemetry integration** — trace context propagation is a separate concern.

## Technical Approach

### New files

| File | Purpose |
|------|---------|
| `serve/sessions.py` | `SessionManager` protocol, `SessionState` dataclass, `InMemorySessionManager` |
| `serve/auth.py` | `ApiKeyAuth` class, `KeyInfo` dataclass, key loading helpers |

### Modified files

| File | Change |
|------|--------|
| `serve/app.py` | `AgentRouter`: accept `session_manager` + `auth`; `handle_invoke`/`handle_stream`: extract user/session, load state, run, save state; typed SSE events in `handle_stream`; heartbeat thread |
| `serve/app.py` | `AgentServer`: pass session_manager + auth to router; auth check in handler |
| `serve/cli.py` | `--api-key` flag; load `api_keys.json`; pass to server constructors |
| `serve/models.py` | `InvokeRequest` gains optional `session_id` field; `StreamEvent` dataclass for typed SSE |
| `serve/_starlette_app.py` | Wire session + auth into ASGI routes; async session eviction background task |
| `serve/__init__.py` | Export `SessionManager`, `InMemorySessionManager`, `ApiKeyAuth` |
| `__init__.py` | Top-level exports with `@beta` |

### Session lifecycle

```
Request arrives
  → Auth middleware extracts user_id (from API key or cookie)
  → Session ID from X-Session-ID header or body, or auto-generate
  → session_manager.get(user_id, session_id) → SessionState
  → Clone agent's base ConversationMemory → inject session's history
  → Run agent (run or astream)
  → Extract updated history from agent → session_manager.put(...)
  → Return response
```

Key design: the agent itself is NOT cloned. Only its memory state is swapped per-request. The agent's tools, config, provider, and system prompt are shared (they're stateless). `ConversationMemory` is the only mutable per-user state.

### Auth flow

```
Authorization: Bearer sk_sel_abc123
  → auth.validate("sk_sel_abc123") → KeyInfo(user_id="alice", role="editor")
  → user_id piped to session manager
  → role checked against RBAC (existing _has_permission)

No Authorization header + builder_session cookie present
  → Existing cookie auth path (unchanged)

No auth at all + auth is configured
  → 401 Unauthorized

No auth at all + auth is NOT configured
  → Allowed (backward compatible, no auth = open access)
```

### Typed SSE format

Enhance `AgentRouter.handle_stream()` to yield typed events. The current implementation yields `data: {"type": "chunk", ...}\n\n`. The new format adds the SSE `event:` field:

```
event: chunk\ndata: {"content": "Hel"}\n\n
event: chunk\ndata: {"content": "lo!"}\n\n
event: tool_call\ndata: {"tool": "search", "args": {...}, "call_id": "tc_1"}\n\n
event: tool_result\ndata: {"tool": "search", "call_id": "tc_1", "result": "..."}\n\n
event: done\ndata: {"content": "...", "iterations": 2}\n\n
```

Clients using `EventSource` API automatically get event routing. Clients using raw `fetch()` that only parse `data:` lines continue to work — the `event:` line is a separate SSE field they'll simply ignore.

## Dependencies

- `ConversationMemory` — for session state cloning (uses `.branch()` and `.add_many()`)
- `EntityMemory` and `KnowledgeMemory` — optional per-session state
- `AgentRouter` and `AgentServer` — existing classes being extended
- Existing RBAC model (`ROLES`, `_has_permission`) — reused for API key roles
- `serve/models.py` — existing request/response dataclasses

## Risks

| Risk | Mitigation |
|------|-----------|
| Memory pressure from many concurrent sessions | `InMemorySessionManager` has `max_sessions` cap + TTL eviction. Document memory-per-session guidance. |
| Thread safety of session manager in stdlib server | `InMemorySessionManager` uses `threading.Lock` around the session dict. Each request gets its own memory clone. |
| Agent state beyond ConversationMemory (e.g., entity_memory is mutated in-place) | `SessionState` holds separate `EntityMemory` and `KnowledgeMemory` instances per session. Clone on first use. |
| Breaking existing API clients | All new params are optional with `None` defaults. No auth configured = open access (current behavior). SSE `event:` field is additive. |
| API key security — keys stored in plaintext | Keys hashed with SHA-256 in the key store. Raw key only needed on first load. Document that `api_keys.json` should have `600` permissions. |
| Heartbeat thread management | Use daemon thread (stdlib) or `asyncio.create_task` (Starlette). Thread dies with server. |
