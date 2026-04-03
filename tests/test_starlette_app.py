"""Tests for the Starlette ASGI builder app."""

import json

import pytest
from starlette.testclient import TestClient

from selectools.serve._starlette_app import create_builder_app

# ─── Health ──────────────────────────────────────────────────────────────────


class TestStarletteHealth:
    def test_health_endpoint(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_no_auth_required(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200


# ─── Builder HTML ────────────────────────────────────────────────────────────


class TestStarletteBuilder:
    def test_builder_html_served(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/builder")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "selectools" in r.text

    def test_root_serves_builder(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "selectools" in r.text


# ─── Auth ────────────────────────────────────────────────────────────────────


class TestStarletteAuth:
    def test_auth_redirects_to_login(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app, follow_redirects=False)
        r = client.get("/builder")
        assert r.status_code == 302
        assert "/login" in r.headers["location"]

    def test_login_page_renders(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app)
        r = client.get("/login")
        assert r.status_code == 200
        assert "login" in r.text.lower() or "unlock" in r.text.lower()

    def test_no_auth_when_token_not_set(self):
        app = create_builder_app(auth_token=None)
        client = TestClient(app)
        r = client.get("/builder")
        assert r.status_code == 200

    def test_login_post_correct_token(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app, follow_redirects=False)
        r = client.post("/login", data={"token": "secret"})
        assert r.status_code == 302
        assert "/builder" in r.headers["location"]
        assert "builder_session" in r.headers.get("set-cookie", "")

    def test_login_post_wrong_token(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app)
        r = client.post("/login", data={"token": "wrong"})
        assert r.status_code == 200
        assert "invalid" in r.text.lower() or "try again" in r.text.lower()

    def test_authed_cookie_grants_access(self):
        """After successful login, the cookie should grant access to /builder."""
        app = create_builder_app(auth_token="secret")
        client = TestClient(app, follow_redirects=False)
        # Login first
        login_resp = client.post("/login", data={"token": "secret"})
        assert login_resp.status_code == 302
        # Now access /builder with the session cookie (TestClient carries cookies)
        builder_resp = client.get("/builder")
        assert builder_resp.status_code == 200
        assert "selectools" in builder_resp.text

    def test_protected_post_returns_401_without_auth(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app)
        r = client.post("/ai-build", json={"description": "test"})
        assert r.status_code == 401
        assert r.json()["error"] == "unauthorized"


# ─── AI Build ────────────────────────────────────────────────────────────────


class TestStarletteAiBuild:
    def test_ai_build_fallback(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/ai-build", json={"description": "simple chatbot"})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 3  # START + agent + END

    def test_ai_build_requires_description(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/ai-build", json={"description": ""})
        assert r.status_code == 400

    def test_ai_build_researcher_writer(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/ai-build", json={"description": "research and write a report"})
        data = r.json()
        names = [n["name"] for n in data["nodes"] if n["type"] == "agent"]
        assert any("research" in n.lower() for n in names)
        assert any("writ" in n.lower() for n in names)


# ─── AI Refine ───────────────────────────────────────────────────────────────


class TestStarletteAiRefine:
    def test_ai_refine_requires_message(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/ai-refine", json={"message": ""})
        assert r.status_code == 400

    def test_ai_refine_no_api_key(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/ai-refine", json={"message": "add a tool", "api_key": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["patch"] is None
        assert "API key" in data["explanation"]


# ─── Estimate Cost ───────────────────────────────────────────────────────────


class TestStarletteEstimateCost:
    def test_estimate_empty_nodes(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/estimate-run-cost", json={"nodes": [], "input": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert "total_tokens" in data
        assert "total_cost_usd" in data

    def test_estimate_with_agent_node(self):
        app = create_builder_app()
        client = TestClient(app)
        nodes = [
            {
                "type": "agent",
                "model": "gpt-4o-mini",
                "system_prompt": "You are helpful.",
                "max_tokens": 200,
            }
        ]
        r = client.post("/estimate-run-cost", json={"nodes": nodes, "input": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_tokens"] > 0


# ─── Smart Route ─────────────────────────────────────────────────────────────


class TestStarletteSmartRoute:
    def test_smart_route_returns_model(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/smart-route", json={"prompt": "hello", "system_prompt": "be helpful"})
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert isinstance(data["model"], str)
        assert len(data["model"]) > 0


# ─── Provider Health ─────────────────────────────────────────────────────────


class TestStarletteProviderHealth:
    def test_provider_health_returns_dict(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/provider-health")
        assert r.status_code == 200
        data = r.json()
        assert "openai" in data
        assert "anthropic" in data

    def test_provider_health_requires_auth(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app, follow_redirects=False)
        r = client.get("/provider-health")
        assert r.status_code == 302


# ─── Eval Dashboard ─────────────────────────────────────────────────────────


class TestStarletteEvalDashboard:
    def test_eval_dashboard_returns_html(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/eval-dashboard")
        assert r.status_code == 200
        assert "eval" in r.text.lower()

    def test_eval_dashboard_requires_auth(self):
        app = create_builder_app(auth_token="secret")
        client = TestClient(app, follow_redirects=False)
        r = client.get("/eval-dashboard")
        assert r.status_code == 302


# ─── Runs ────────────────────────────────────────────────────────────────────


class TestStarletteRuns:
    def test_runs_endpoint(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/runs", json={})
        assert r.status_code == 200
        data = r.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)


# ─── Feedback ────────────────────────────────────────────────────────────────


class TestStarletteFeedback:
    def test_feedback_endpoint(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.post("/feedback", json={"run_id": "test-123", "score": 1})
        assert r.status_code == 200
        assert r.json()["ok"] is True


# ─── Run SSE ─────────────────────────────────────────────────────────────────


class TestStarletteRunSSE:
    def test_run_mock_mode(self):
        """Without api_key the /run endpoint should use mock mode."""
        app = create_builder_app()
        client = TestClient(app)
        nodes = [
            {"id": "s1", "type": "start", "name": "START"},
            {
                "id": "a1",
                "type": "agent",
                "name": "Agent",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "be helpful",
                "tools": "",
                "frozen": False,
            },
            {"id": "e1", "type": "end", "name": "END"},
        ]
        edges = [
            {"id": "ed1", "from": "s1", "to": "a1", "label": ""},
            {"id": "ed2", "from": "a1", "to": "e1", "label": ""},
        ]
        r = client.post("/run", json={"input": "hello", "nodes": nodes, "edges": edges})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "run_start" in body
        assert "[DONE]" in body


# ─── CORS ────────────────────────────────────────────────────────────────────


class TestStarletteCORS:
    def test_cors_headers_on_preflight(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 200
        assert "access-control-allow-origin" in r.headers

    def test_cors_headers_on_normal_request(self):
        app = create_builder_app()
        client = TestClient(app)
        r = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"
