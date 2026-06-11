"""Tests for a2a/ — A2A protocol server (Agent Card + JSON-RPC) and client.

Covers: agent card shape, happy-path message/send, JSON-RPC error codes,
bearer auth accept/reject, task lifecycle state on success and failure,
tasks/get + tasks/cancel, and the A2AClient discover/send_task round-trip
against the real server app in-process via httpx ASGITransport.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

starlette = pytest.importorskip("starlette", reason="starlette not installed")
httpx = pytest.importorskip("httpx", reason="httpx not installed")
from starlette.testclient import TestClient  # noqa: E402

from selectools.a2a import A2AClient, A2AServer  # noqa: E402
from selectools.a2a.types import A2AError, AgentCard, TaskState  # noqa: E402
from selectools.agent import Agent, AgentConfig  # noqa: E402
from selectools.tools import Tool, ToolParameter  # noqa: E402
from tests.conftest import SharedErrorProvider, SharedFakeProvider  # noqa: E402

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _echo_tool() -> Tool:
    def _fn(text: str) -> str:
        return text

    return Tool(
        name="echo",
        description="Echo the input text",
        function=_fn,
        parameters=[ToolParameter(name="text", param_type=str, description="text")],
    )


def make_agent(name: str = "researcher", responses: Optional[List[str]] = None) -> Agent:
    provider = SharedFakeProvider(responses=responses or ["agent reply"])
    return Agent(
        tools=[_echo_tool()],
        provider=provider,
        config=AgentConfig(name=name, model="fake-model"),
    )


def make_failing_agent() -> Agent:
    # RuntimeError escapes agent.run (ProviderError is gracefully degraded
    # into an AgentResult by the agent loop, so it would NOT hit the
    # failed-task path).
    provider = SharedErrorProvider(exception=RuntimeError("kaboom"))
    return Agent(
        tools=[_echo_tool()],
        provider=provider,
        config=AgentConfig(name="boom", model="fake-model", max_retries=0),
    )


def make_server(**kwargs) -> A2AServer:
    return A2AServer(agent=kwargs.pop("agent", None) or make_agent(), **kwargs)


def rpc(client: TestClient, method: str, params: Optional[dict] = None, **kwargs):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return client.post("/a2a", json=payload, **kwargs)


def send_text(client: TestClient, text: str, **kwargs):
    return rpc(
        client,
        "message/send",
        {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}},
        **kwargs,
    )


# ─── Agent Card ──────────────────────────────────────────────────────────────


class TestAgentCard:
    def test_card_shape(self):
        client = TestClient(make_server(description="Research agent"))
        r = client.get("/.well-known/agent.json")
        assert r.status_code == 200
        card = r.json()
        assert card["name"] == "researcher"
        assert card["description"] == "Research agent"
        assert card["protocolVersion"]
        assert card["version"]
        assert card["capabilities"] == {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        }
        assert card["defaultInputModes"] == ["text/plain"]
        assert card["defaultOutputModes"] == ["text/plain"]

    def test_card_lists_tools_as_skills(self):
        client = TestClient(make_server())
        skills = client.get("/.well-known/agent.json").json()["skills"]
        assert len(skills) == 1
        assert skills[0]["id"] == "echo"
        assert skills[0]["name"] == "echo"
        assert skills[0]["description"] == "Echo the input text"

    def test_card_never_requires_auth(self):
        client = TestClient(make_server(auth_token="sk-secret"))
        r = client.get("/.well-known/agent.json")
        assert r.status_code == 200

    def test_card_default_description_not_system_prompt(self):
        agent = Agent(
            tools=[_echo_tool()],
            provider=SharedFakeProvider(),
            config=AgentConfig(name="a", model="fake", system_prompt="SECRET INSTRUCTIONS"),
        )
        client = TestClient(A2AServer(agent=agent))
        card = client.get("/.well-known/agent.json").json()
        assert "SECRET" not in card["description"]


# ─── message/send happy path + lifecycle ─────────────────────────────────────


class TestMessageSend:
    def test_happy_path_returns_completed_task(self):
        client = TestClient(make_server(agent=make_agent(responses=["42"])))
        r = send_text(client, "what is the answer?")
        assert r.status_code == 200
        body = r.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1
        task = body["result"]
        assert task["kind"] == "task"
        assert task["id"]
        assert task["contextId"]
        assert task["status"]["state"] == TaskState.COMPLETED == "completed"
        assert task["status"]["timestamp"]
        parts = task["artifacts"][0]["parts"]
        assert parts == [{"kind": "text", "text": "42"}]

    def test_tasks_send_alias(self):
        client = TestClient(make_server())
        r = rpc(
            client,
            "tasks/send",
            {"message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}},
        )
        assert r.json()["result"]["status"]["state"] == "completed"

    def test_context_id_echoed(self):
        client = TestClient(make_server())
        r = rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "contextId": "ctx-1",
                    "parts": [{"kind": "text", "text": "hi"}],
                }
            },
        )
        assert r.json()["result"]["contextId"] == "ctx-1"

    def test_file_and_data_parts_accepted(self):
        # v1 is text-first: file/data parts must not be rejected.
        client = TestClient(make_server())
        r = rpc(
            client,
            "message/send",
            {
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": "hi"},
                        {"kind": "data", "data": {"k": "v"}},
                        {"kind": "file", "file": {"uri": "https://x/y.png"}},
                    ],
                }
            },
        )
        assert r.json()["result"]["status"]["state"] == "completed"

    def test_agent_failure_returns_failed_task_not_rpc_error(self):
        client = TestClient(make_server(agent=make_failing_agent()))
        r = send_text(client, "boom")
        assert r.status_code == 200
        body = r.json()
        assert "error" not in body
        task = body["result"]
        assert task["status"]["state"] == TaskState.FAILED == "failed"
        detail = task["status"]["message"]["parts"][0]["text"]
        assert detail == "Agent execution failed (RuntimeError)"

    def test_agent_failure_does_not_leak_exception_detail(self):
        # Provider/tool exception strings can contain internal URLs or paths;
        # remote callers must only see the exception type, not its message.
        client = TestClient(make_server(agent=make_failing_agent()))
        task = send_text(client, "boom").json()["result"]
        detail = task["status"]["message"]["parts"][0]["text"]
        assert "kaboom" not in detail


# ─── Per-request agent isolation ─────────────────────────────────────────────


class TestRequestIsolation:
    def test_no_cross_caller_context_leak(self):
        # Two sequential requests from different callers: the second
        # provider call must NOT see the first caller's text (A2A v1 has
        # no session model, so every request starts from a clean slate).
        agent = make_agent(responses=["reply A", "reply B"])
        provider = agent.provider
        client = TestClient(A2AServer(agent=agent))
        r1 = send_text(client, "caller-A-secret-payload")
        assert r1.json()["result"]["status"]["state"] == "completed"
        r2 = send_text(client, "caller-B question")
        assert r2.json()["result"]["status"]["state"] == "completed"
        contents = [m.content or "" for m in provider.last_messages]
        assert any("caller-B question" in c for c in contents)
        assert not any("caller-A-secret-payload" in c for c in contents)

    def test_shared_agent_history_not_mutated(self):
        agent = make_agent()
        client = TestClient(A2AServer(agent=agent))
        send_text(client, "hi")
        send_text(client, "hi again")
        assert agent._history == []


# ─── Task store bounds + body size guard ─────────────────────────────────────


class TestTaskStoreBounds:
    def test_fifo_eviction(self):
        client = TestClient(make_server(max_tasks=2))
        ids = [send_text(client, f"msg {i}").json()["result"]["id"] for i in range(3)]
        # Oldest task evicted first.
        r0 = rpc(client, "tasks/get", {"id": ids[0]})
        assert r0.json()["error"]["code"] == -32001
        for tid in ids[1:]:
            assert rpc(client, "tasks/get", {"id": tid}).json()["result"]["id"] == tid

    def test_oversized_body_rejected(self):
        client = TestClient(make_server(max_body_bytes=200))
        r = send_text(client, "x" * 1000)
        body = r.json()
        assert body["error"]["code"] == -32600
        assert "body" in body["error"]["message"].lower()

    def test_body_within_limit_accepted(self):
        client = TestClient(make_server(max_body_bytes=10_000))
        assert send_text(client, "hi").json()["result"]["status"]["state"] == "completed"


# ─── serve() open-endpoint warning ───────────────────────────────────────────


class TestServeWarning:
    def _fake_uvicorn(self, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("uvicorn")
        fake.run = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "uvicorn", fake)

    def test_warns_when_unauthenticated_on_public_host(self, monkeypatch, caplog):
        import logging

        self._fake_uvicorn(monkeypatch)
        server = make_server()
        with caplog.at_level(logging.WARNING, logger="selectools.a2a.server"):
            server.serve(host="0.0.0.0")  # nosec B104
        assert any("unauthenticated" in r.getMessage() for r in caplog.records)

    def test_no_warning_on_loopback_or_with_token(self, monkeypatch, caplog):
        import logging

        self._fake_uvicorn(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="selectools.a2a.server"):
            make_server().serve(host="127.0.0.1")
            make_server(auth_token="sk-secret").serve(host="0.0.0.0")  # nosec B104
        assert not any("unauthenticated" in r.getMessage() for r in caplog.records)


# ─── JSON-RPC error codes ────────────────────────────────────────────────────


class TestJsonRpcErrors:
    def test_invalid_json_parse_error(self):
        client = TestClient(make_server())
        r = client.post("/a2a", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.json()["error"]["code"] == -32700

    def test_non_object_body_invalid_request(self):
        client = TestClient(make_server())
        r = client.post("/a2a", json=[1, 2, 3])
        assert r.json()["error"]["code"] == -32600

    def test_missing_method_invalid_request(self):
        client = TestClient(make_server())
        r = client.post("/a2a", json={"jsonrpc": "2.0", "id": 1})
        assert r.json()["error"]["code"] == -32600

    def test_wrong_jsonrpc_version_invalid_request(self):
        client = TestClient(make_server())
        r = client.post("/a2a", json={"jsonrpc": "1.0", "id": 1, "method": "message/send"})
        assert r.json()["error"]["code"] == -32600

    def test_unknown_method(self):
        client = TestClient(make_server())
        r = rpc(client, "nope/nothing")
        assert r.json()["error"]["code"] == -32601

    def test_missing_message_invalid_params(self):
        client = TestClient(make_server())
        r = rpc(client, "message/send", {})
        assert r.json()["error"]["code"] == -32602

    def test_no_text_part_invalid_params(self):
        client = TestClient(make_server())
        r = rpc(client, "message/send", {"message": {"role": "user", "parts": []}})
        assert r.json()["error"]["code"] == -32602

    def test_error_echoes_request_id(self):
        client = TestClient(make_server())
        r = client.post("/a2a", json={"jsonrpc": "2.0", "id": "abc", "method": "nope"})
        assert r.json()["id"] == "abc"


# ─── tasks/get + tasks/cancel ────────────────────────────────────────────────


class TestTaskRetrieval:
    def test_tasks_get_returns_stored_task(self):
        client = TestClient(make_server())
        task = send_text(client, "hi").json()["result"]
        r = rpc(client, "tasks/get", {"id": task["id"]})
        assert r.json()["result"]["id"] == task["id"]
        assert r.json()["result"]["status"]["state"] == "completed"

    def test_tasks_get_unknown_id(self):
        client = TestClient(make_server())
        r = rpc(client, "tasks/get", {"id": "nope"})
        assert r.json()["error"]["code"] == -32001

    def test_tasks_cancel_terminal_task_not_cancelable(self):
        client = TestClient(make_server())
        task = send_text(client, "hi").json()["result"]
        r = rpc(client, "tasks/cancel", {"id": task["id"]})
        assert r.json()["error"]["code"] == -32002


# ─── Auth ────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_post_rejected_without_token(self):
        client = TestClient(make_server(auth_token="sk-secret"))
        r = send_text(client, "hi")
        assert r.status_code == 401

    def test_post_rejected_with_wrong_token(self):
        client = TestClient(make_server(auth_token="sk-secret"))
        r = send_text(client, "hi", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_post_accepted_with_token(self):
        client = TestClient(make_server(auth_token="sk-secret"))
        r = send_text(client, "hi", headers={"Authorization": "Bearer sk-secret"})
        assert r.status_code == 200
        assert r.json()["result"]["status"]["state"] == "completed"

    def test_no_auth_configured_open(self):
        client = TestClient(make_server())
        assert send_text(client, "hi").status_code == 200


# ─── A2AClient round-trip (httpx ASGITransport against the real app) ─────────


def _client_for(server: A2AServer, auth_token: Optional[str] = None) -> A2AClient:
    transport = httpx.ASGITransport(app=server)
    return A2AClient("http://testserver", auth_token=auth_token, transport=transport)


class TestA2AClient:
    @pytest.mark.asyncio
    async def test_discover(self):
        client = _client_for(make_server(description="Research agent"))
        card = await client.discover()
        assert isinstance(card, AgentCard)
        assert card.name == "researcher"
        assert card.description == "Research agent"
        assert card.skills[0].id == "echo"
        assert card.raw["capabilities"]["streaming"] is False

    @pytest.mark.asyncio
    async def test_send_task_round_trip(self):
        server = make_server(agent=make_agent(responses=["quantum is hot"]))
        client = _client_for(server)
        task = await client.send_task("Research quantum computing trends")
        assert task.state == TaskState.COMPLETED
        assert task.text == "quantum is hot"
        assert task.id

    @pytest.mark.asyncio
    async def test_send_task_with_auth(self):
        server = make_server(auth_token="sk-secret")
        task = await _client_for(server, auth_token="sk-secret").send_task("hi")
        assert task.state == TaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_send_task_auth_rejected(self):
        server = make_server(auth_token="sk-secret")
        with pytest.raises(A2AError) as exc_info:
            await _client_for(server, auth_token="wrong").send_task("hi")
        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_task_failure_state(self):
        server = make_server(agent=make_failing_agent())
        task = await _client_for(server).send_task("boom")
        assert task.state == TaskState.FAILED
        assert task.error

    @pytest.mark.asyncio
    async def test_rpc_error_raises_a2a_error(self):
        server = make_server()
        client = _client_for(server)
        with pytest.raises(A2AError) as exc_info:
            await client._call("nope/method", {})
        assert exc_info.value.code == -32601

    def test_sync_wrappers(self):
        server = make_server(agent=make_agent(responses=["sync reply"]))
        client = _client_for(server)
        card = client.discover_sync()
        assert card.name == "researcher"
        task = client.send_task_sync("hello")
        assert task.text == "sync reply"


# ─── Exports ─────────────────────────────────────────────────────────────────


class TestExports:
    def test_a2a_server_exposed_from_serve(self):
        from selectools.serve import A2AServer as FromServe

        assert FromServe is A2AServer
