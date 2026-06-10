"""Tests for serve/api.py — AgentAPI auto-generated production REST endpoints.

Covers: chat round-trip, session CRUD, per-user isolation, bearer auth,
SSE streaming, multi-agent routing, and error schema.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

starlette = pytest.importorskip("starlette", reason="starlette not installed")
from starlette.testclient import TestClient  # noqa: E402

from selectools.agent import Agent, AgentConfig  # noqa: E402
from selectools.serve import AgentAPI  # noqa: E402
from selectools.tools import Tool, ToolParameter  # noqa: E402
from tests.conftest import SharedFakeProvider  # noqa: E402

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _noop_tool() -> Tool:
    def _fn(text: str) -> str:
        return text

    return Tool(
        name="noop",
        description="No-op tool",
        function=_fn,
        parameters=[ToolParameter(name="text", param_type=str, description="text")],
    )


def make_agent(name: str = "assistant", responses: Optional[List[str]] = None) -> Agent:
    provider = SharedFakeProvider(responses=responses or ["hello from agent"])
    return Agent(
        tools=[_noop_tool()],
        provider=provider,
        config=AgentConfig(name=name, model="fake-model"),
    )


def make_client(app: AgentAPI) -> TestClient:
    return TestClient(app)


def sse_events(text: str) -> List[dict]:
    """Parse SSE body into a list of JSON events (excluding [DONE])."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))
    return events


# ─── Health ──────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.get("/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["agents"] == ["assistant"]

    def test_health_requires_no_auth(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        r = client.get("/v1/health")
        assert r.status_code == 200


# ─── Chat ────────────────────────────────────────────────────────────────────


class TestChat:
    def test_chat_round_trip(self):
        client = make_client(AgentAPI(agents=[make_agent(responses=["42 is the answer"])]))
        r = client.post("/v1/chat", json={"input": "what is the answer?"})
        assert r.status_code == 200
        data = r.json()
        assert data["output"] == "42 is the answer"
        assert data["agent"] == "assistant"
        assert data["session_id"]
        assert "usage" in data
        assert data["usage"]["total_tokens"] >= 0

    def test_chat_missing_input_422(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat", json={})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["type"] == "validation_error"
        assert "input" in err["message"]

    def test_chat_non_string_input_422(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat", json={"input": 42})
        assert r.status_code == 422

    def test_chat_invalid_json_422(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post(
            "/v1/chat", content=b"not json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422
        assert r.json()["error"]["type"] == "validation_error"

    def test_chat_unknown_session_404(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat", json={"input": "hi", "session_id": "nope"})
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    def test_chat_continues_session_history(self):
        agent = make_agent(responses=["first reply", "second reply"])
        client = make_client(AgentAPI(agents=[agent]))
        r1 = client.post("/v1/chat", json={"input": "first question"})
        sid = r1.json()["session_id"]
        r2 = client.post("/v1/chat", json={"input": "second question", "session_id": sid})
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid
        # Provider must have seen the prior turn in the second call
        provider = agent.provider
        contents = [m.content or "" for m in provider.last_messages]
        assert any("first question" in c for c in contents)
        assert any("first reply" in c for c in contents)
        assert any("second question" in c for c in contents)


# ─── Auth ────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_missing_bearer_rejected(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        r = client.post("/v1/chat", json={"input": "hi"})
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "unauthorized"

    def test_wrong_bearer_rejected(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        r = client.post(
            "/v1/chat", json={"input": "hi"}, headers={"Authorization": "Bearer sk-wrong"}
        )
        assert r.status_code == 401

    def test_correct_bearer_accepted(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        r = client.post(
            "/v1/chat", json={"input": "hi"}, headers={"Authorization": "Bearer sk-secret"}
        )
        assert r.status_code == 200

    def test_sessions_also_protected(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        assert client.post("/v1/sessions").status_code == 401
        assert client.get("/v1/sessions/abc").status_code == 401
        assert client.delete("/v1/sessions/abc").status_code == 401

    def test_no_auth_key_means_open(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat", json={"input": "hi"})
        assert r.status_code == 200


# ─── Sessions CRUD ───────────────────────────────────────────────────────────


class TestSessions:
    def test_create_session(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/sessions")
        assert r.status_code == 201
        assert r.json()["session_id"]

    def test_get_session_history(self):
        client = make_client(AgentAPI(agents=[make_agent(responses=["pong"])]))
        sid = client.post("/v1/sessions").json()["session_id"]
        r = client.get(f"/v1/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["messages"] == []

        client.post("/v1/chat", json={"input": "ping", "session_id": sid})
        r = client.get(f"/v1/sessions/{sid}")
        msgs = r.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "ping"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "pong"

    def test_get_unknown_session_404(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.get("/v1/sessions/missing")
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    def test_delete_session(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        sid = client.post("/v1/sessions").json()["session_id"]
        r = client.delete(f"/v1/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert client.get(f"/v1/sessions/{sid}").status_code == 404

    def test_delete_unknown_session_404(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        assert client.delete("/v1/sessions/missing").status_code == 404


# ─── Per-user isolation ──────────────────────────────────────────────────────


class TestUserIsolation:
    def test_user_cannot_read_another_users_session(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat", json={"input": "hi"}, headers={"user_id": "alice"})
        sid = r.json()["session_id"]

        # Alice can read her session
        assert client.get(f"/v1/sessions/{sid}", headers={"user_id": "alice"}).status_code == 200
        # Bob cannot
        assert client.get(f"/v1/sessions/{sid}", headers={"user_id": "bob"}).status_code == 404
        # Anonymous cannot
        assert client.get(f"/v1/sessions/{sid}").status_code == 404

    def test_user_cannot_delete_another_users_session(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        sid = client.post("/v1/sessions", headers={"user_id": "alice"}).json()["session_id"]
        assert client.delete(f"/v1/sessions/{sid}", headers={"user_id": "bob"}).status_code == 404
        assert client.delete(f"/v1/sessions/{sid}", headers={"user_id": "alice"}).status_code == 200

    def test_user_cannot_chat_into_another_users_session(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        sid = client.post("/v1/sessions", headers={"user_id": "alice"}).json()["session_id"]
        r = client.post(
            "/v1/chat", json={"input": "hi", "session_id": sid}, headers={"user_id": "bob"}
        )
        assert r.status_code == 404


# ─── Streaming (SSE) ─────────────────────────────────────────────────────────


class TestChatStream:
    def test_stream_yields_chunks_and_result(self):
        client = make_client(AgentAPI(agents=[make_agent(responses=["streamed answer"])]))
        r = client.post("/v1/chat/stream", json={"input": "stream it"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in r.text

        events = sse_events(r.text)
        chunk_text = "".join(e["content"] for e in events if e["type"] == "chunk")
        assert chunk_text == "streamed answer"
        result = [e for e in events if e["type"] == "result"]
        assert len(result) == 1
        assert result[0]["output"] == "streamed answer"
        assert result[0]["session_id"]

    def test_stream_persists_session(self):
        client = make_client(AgentAPI(agents=[make_agent(responses=["streamed answer"])]))
        r = client.post("/v1/chat/stream", json={"input": "stream it"})
        sid = [e for e in sse_events(r.text) if e["type"] == "result"][0]["session_id"]
        msgs = client.get(f"/v1/sessions/{sid}").json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["content"] == "streamed answer"

    def test_stream_requires_auth(self):
        client = make_client(AgentAPI(agents=[make_agent()], auth_key="sk-secret"))
        r = client.post("/v1/chat/stream", json={"input": "hi"})
        assert r.status_code == 401

    def test_stream_validation_error(self):
        client = make_client(AgentAPI(agents=[make_agent()]))
        r = client.post("/v1/chat/stream", json={})
        assert r.status_code == 422

    def test_stream_sync_only_provider_falls_back(self):
        provider = SharedFakeProvider(responses=["sync only"], supports_async=False)
        agent = Agent(
            tools=[_noop_tool()],
            provider=provider,
            config=AgentConfig(name="assistant", model="fake-model"),
        )
        client = make_client(AgentAPI(agents=[agent]))
        r = client.post("/v1/chat/stream", json={"input": "hi"})
        assert r.status_code == 200
        events = sse_events(r.text)
        chunk_text = "".join(e["content"] for e in events if e["type"] == "chunk")
        assert chunk_text == "sync only"
        assert [e for e in events if e["type"] == "result"][0]["output"] == "sync only"


# ─── Multi-agent routing ─────────────────────────────────────────────────────


class TestMultiAgent:
    def _app(self) -> AgentAPI:
        a = make_agent(name="alpha", responses=["from alpha"])
        b = make_agent(name="beta", responses=["from beta"])
        return AgentAPI(agents=[a, b])

    def test_route_by_name(self):
        client = make_client(self._app())
        r = client.post("/v1/chat", json={"input": "hi", "agent": "beta"})
        assert r.status_code == 200
        assert r.json()["output"] == "from beta"
        assert r.json()["agent"] == "beta"

    def test_default_is_first_agent(self):
        client = make_client(self._app())
        r = client.post("/v1/chat", json={"input": "hi"})
        assert r.json()["output"] == "from alpha"
        assert r.json()["agent"] == "alpha"

    def test_unknown_agent_404(self):
        client = make_client(self._app())
        r = client.post("/v1/chat", json={"input": "hi", "agent": "gamma"})
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"
        assert "gamma" in r.json()["error"]["message"]

    def test_stream_routes_by_name(self):
        client = make_client(self._app())
        r = client.post("/v1/chat/stream", json={"input": "hi", "agent": "beta"})
        events = sse_events(r.text)
        chunk_text = "".join(e["content"] for e in events if e["type"] == "chunk")
        assert chunk_text == "from beta"

    def test_health_lists_all_agents(self):
        client = make_client(self._app())
        assert client.get("/v1/health").json()["agents"] == ["alpha", "beta"]

    def test_duplicate_agent_names_rejected(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            AgentAPI(agents=[make_agent(name="same"), make_agent(name="same")])

    def test_single_agent_not_in_list(self):
        client = make_client(AgentAPI(agents=make_agent(responses=["solo"])))
        r = client.post("/v1/chat", json={"input": "hi"})
        assert r.json()["output"] == "solo"


# ─── Custom session store ────────────────────────────────────────────────────


class TestSessionStoreParam:
    def test_json_file_store(self, tmp_path):
        from selectools.sessions import JsonFileSessionStore

        store = JsonFileSessionStore(directory=str(tmp_path))
        client = make_client(
            AgentAPI(agents=[make_agent(responses=["persisted"])], session_store=store)
        )
        r = client.post("/v1/chat", json={"input": "hi"}, headers={"user_id": "alice"})
        sid = r.json()["session_id"]
        assert client.get(f"/v1/sessions/{sid}", headers={"user_id": "alice"}).status_code == 200
        # Files actually written to disk
        assert any(tmp_path.iterdir())
