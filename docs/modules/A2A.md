# A2A Protocol

Agent-to-agent communication over the A2A protocol (the Google-backed
emerging standard). One selectools agent exposes itself over HTTP; other
agents — selectools or not — discover it and send it tasks.

> **Stability:** `@beta` — first release, API may evolve.

```bash
pip install selectools[serve]
```

## Serving an agent

```python
from selectools.serve import A2AServer

server = A2AServer(agent=my_agent, auth_token="sk-...")
server.serve(port=8000)
```

`A2AServer` is itself an ASGI app, so you can also run it with any ASGI
server directly:

```bash
uvicorn my_module:server --port 8000
```

### Endpoints

| Route | Method | Auth | Purpose |
|---|---|---|---|
| `/.well-known/agent.json` | GET | never | Agent Card discovery |
| `/a2a` | POST | bearer (optional) | JSON-RPC 2.0 message handler |

### Trust model (v1)

- **Single shared tenant.** One bearer token grants full access: any
  authenticated caller can read every stored task — including other
  callers' inputs and outputs — via `tasks/get`. Do not share one server
  between mutually untrusting parties.
- **Do not run unauthenticated on a public interface.** Without
  `auth_token`, anyone who can reach the port can run the agent and read
  all stored tasks. `serve()` logs a warning when started without a
  token on a non-loopback host.
- **Per-request agent isolation.** Each `message/send` runs on an
  isolated clone of the agent (fresh history and usage, memory dropped),
  so callers never see each other's conversation context. A2A v1 has no
  session model: every request starts from a clean slate, with no
  cross-request memory.
- **Bounded resources.** The in-memory task store keeps at most
  `max_tasks` tasks (default 2000, FIFO eviction — an evicted id returns
  `-32001` from `tasks/get`), and bodies larger than `max_body_bytes`
  (default 1 MiB) are rejected with `-32600` before parsing.
- **Sanitized errors.** When the agent raises, remote callers only see
  the exception type (`Agent execution failed (RuntimeError)`); the full
  detail goes to the server log at warning level.

### Agent Card

The card is auto-generated from the agent's metadata: `config.name`,
the server `description`, and one A2A *skill* per tool (name +
description). The agent's system prompt is intentionally **not** used as
the default description, since the card is served unauthenticated.

```json
{
  "protocolVersion": "0.2.6",
  "name": "researcher",
  "description": "Research agent that summarizes topics",
  "url": "",
  "version": "0.21.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {"id": "summarize", "name": "summarize", "description": "Summarize a text", "tags": ["tool"]}
  ]
}
```

### JSON-RPC methods

| Method | Description |
|---|---|
| `message/send` | Run a task through the agent, return the final Task |
| `tasks/send` | Legacy alias for `message/send` |
| `tasks/get` | Fetch a previously created task by id |
| `tasks/cancel` | Cancel a task (always `-32002` in v1 — tasks finish within the request) |

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Research quantum computing trends"}],
      "messageId": "..."
    }
  }
}
```

Response (`result` is an A2A Task object):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "9f1c...",
    "contextId": "ab42...",
    "kind": "task",
    "status": {"state": "completed", "timestamp": "2026-06-10T12:00:00+00:00"},
    "artifacts": [
      {"artifactId": "...", "name": "response", "parts": [{"kind": "text", "text": "..."}]}
    ],
    "history": [ ... ]
  }
}
```

### Task lifecycle

`submitted` → `working` → `input-required` → `completed` / `failed` /
`canceled`.

This synchronous v1 runs the agent inside the request, so a task moves
submitted→working→completed (or failed) before the response is sent. The
`status.state` field is modeled anyway so async backends can slot in
later without changing the wire format.

### Error mapping

| Condition | Result |
|---|---|
| Invalid bearer token | HTTP `401` (before any JSON-RPC handling) |
| Body is not valid JSON | JSON-RPC error `-32700` |
| Body larger than `max_body_bytes` | JSON-RPC error `-32600` (checked before parsing) |
| Not a valid JSON-RPC 2.0 request | JSON-RPC error `-32600` |
| Unknown method | JSON-RPC error `-32601` |
| Missing/invalid `message` or no text part | JSON-RPC error `-32602` |
| Unknown (or evicted) task id (`tasks/get`/`tasks/cancel`) | JSON-RPC error `-32001` |
| Task not cancelable | JSON-RPC error `-32002` |
| **Agent raises during the run** | Task with `status.state = "failed"` and the exception *type* (not its message) in `status.message` — *not* a transport-level error |

### Text-first v1

`file` and `data` parts are accepted and never rejected: `data` parts
are appended to the prompt as JSON, `file` parts are acknowledged but
their content is not forwarded to the agent. Full multimodal pass-through
is a follow-up.

## Consuming a remote agent

```python
from selectools.a2a import A2AClient

client = A2AClient("https://other-agent.example.com", auth_token="sk-...")

card = await client.discover()          # reads /.well-known/agent.json
print(card.name, [s.id for s in card.skills])

result = await client.send_task("Research quantum computing trends")
print(result.state)                     # "completed"
print(result.text)                      # the agent's answer
```

`send_task` returns an `A2ATask` (`id`, `context_id`, `state`, `text`,
`error`, `raw`). Agent failures come back as `state == "failed"` with
the detail in `.error`; protocol and transport failures raise `A2AError`.

Sync code can use the wrappers:

```python
card = client.discover_sync()
task = client.send_task_sync("hello")
```

### In-process testing

Pass an `httpx.ASGITransport` to talk to an `A2AServer` without a socket:

```python
import httpx
from selectools.a2a import A2AClient

transport = httpx.ASGITransport(app=server)
client = A2AClient("http://testserver", transport=transport)
```

See [`examples/103_a2a_protocol.py`](https://github.com/johnnichev/selectools/blob/main/examples/103_a2a_protocol.py)
for a complete offline round-trip.
