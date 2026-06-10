#!/usr/bin/env python3
"""
Agent-as-API — auto-generated production REST endpoints for any Agent.

Demonstrates AgentAPI: one line turns an Agent (or several) into a
production-ready Starlette ASGI app with standardized JSON schemas,
session persistence via any SessionStore backend, per-user isolation
through the user_id header, and optional bearer-token auth.

Endpoints generated:

    POST   /v1/chat           — single-turn completion (JSON)
    POST   /v1/chat/stream    — streaming completion (SSE)
    POST   /v1/sessions       — create session
    GET    /v1/sessions/{id}  — get session history
    DELETE /v1/sessions/{id}  — delete session
    GET    /v1/health         — health check (never requires auth)

No API key needed. Runs entirely offline with the built-in LocalProvider
and Starlette's TestClient. In production, deploy with uvicorn:

    # api.py
    from selectools.serve import AgentAPI
    app = AgentAPI(agents=[my_agent], auth_key="sk-...")
    # uvicorn api:app --port 8000

Or straight from a YAML config:

    selectools serve agent.yaml --api --port 8000

Prerequisites: pip install selectools[serve]
Run: python examples/97_agent_as_api.py
"""

from __future__ import annotations

import json

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider
from selectools.serve import AgentAPI

AUTH_KEY = "sk-demo-key"


@tool()
def word_count(text: str) -> str:
    """Count the words in a text."""
    return f"{len(text.split())} words"


def build_app() -> AgentAPI:
    support = Agent(
        tools=[word_count],
        provider=LocalProvider(),
        config=AgentConfig(name="support", model="local-model"),
    )
    billing = Agent(
        tools=[word_count],
        provider=LocalProvider(),
        config=AgentConfig(name="billing", model="local-model"),
    )
    # Multi-agent: route requests with the optional "agent" field.
    # session_store accepts any SessionStore (JSON file, SQLite, Redis,
    # Supabase); the default is an in-memory store.
    return AgentAPI(agents=[support, billing], auth_key=AUTH_KEY)


def main() -> None:
    from starlette.testclient import TestClient

    app = build_app()
    client = TestClient(app)
    auth = {"Authorization": f"Bearer {AUTH_KEY}"}

    print("=== GET /v1/health (no auth required) ===")
    print(json.dumps(client.get("/v1/health").json(), indent=2))

    print("\n=== POST /v1/chat without auth -> 401 ===")
    r = client.post("/v1/chat", json={"input": "hello"})
    print(r.status_code, json.dumps(r.json()))

    print("\n=== POST /v1/chat (default agent, new session) ===")
    r = client.post(
        "/v1/chat",
        json={"input": "How many words is 'the quick brown fox'?"},
        headers={**auth, "user_id": "alice"},
    )
    body = r.json()
    session_id = body["session_id"]
    print(json.dumps(body, indent=2))

    print("\n=== POST /v1/chat (continue session, route to 'billing') ===")
    r = client.post(
        "/v1/chat",
        json={"input": "And my invoice?", "session_id": session_id, "agent": "billing"},
        headers={**auth, "user_id": "alice"},
    )
    print(json.dumps(r.json(), indent=2))

    print("\n=== GET /v1/sessions/{id} (alice sees her history) ===")
    r = client.get(f"/v1/sessions/{session_id}", headers={**auth, "user_id": "alice"})
    print(json.dumps(r.json(), indent=2))

    print("\n=== GET /v1/sessions/{id} as bob -> 404 (per-user isolation) ===")
    r = client.get(f"/v1/sessions/{session_id}", headers={**auth, "user_id": "bob"})
    print(r.status_code, json.dumps(r.json()))

    print("\n=== POST /v1/chat/stream (SSE) ===")
    r = client.post(
        "/v1/chat/stream",
        json={"input": "stream me"},
        headers={**auth, "user_id": "alice"},
    )
    for line in r.text.splitlines():
        if line.startswith("data: "):
            print(line)

    print("\n=== DELETE /v1/sessions/{id} ===")
    r = client.delete(f"/v1/sessions/{session_id}", headers={**auth, "user_id": "alice"})
    print(r.status_code, json.dumps(r.json()))


if __name__ == "__main__":
    main()
