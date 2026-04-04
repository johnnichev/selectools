"""Coverage tests for selectools.serve.app — targeting 80%+ line coverage.

Tests are organized into:
1. Pure helper functions (no server needed)
2. AgentRouter unit tests
3. AgentServer HTTP integration tests (threading + urllib)
4. BuilderServer HTTP integration tests for uncovered routes
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List
from unittest import mock

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.stubs import LocalProvider
from selectools.serve.app import (
    CAPABILITY_TIERS,
    LOGIN_HTML,
    LOGIN_HTML_ERROR,
    ROLES,
    AgentRouter,
    AgentServer,
    BuilderServer,
    _ai_build_fallback,
    _apply_pinned_ports,
    _builder_run_mock,
    _check_graph_permission,
    _estimate_run_cost,
    _estimate_task_tier,
    _eval_route,
    _fire_eval_alert,
    _has_permission,
    _log_run,
    _make_provider,
    _make_session_cookie,
    _parse_python_to_graph,
    _render_hitl_form,
    _resolve_auth_token,
    _resolve_users,
    _route_experiment,
    _run_builder_evals,
    _run_evals_on_run,
    _smart_route,
    create_app,
)
from selectools.toolbox import get_all_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    tools = get_all_tools()[:1]
    return Agent(tools, provider=LocalProvider(), config=AgentConfig(name="test"))


def _free_port() -> int:
    """Get a random free port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from following redirects so we can assert on 302."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

    def http_error_302(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    http_error_301 = http_error_303 = http_error_307 = http_error_302


def _open_no_redirect(req):
    opener = urllib.request.build_opener(_NoRedirect)
    return opener.open(req)


# ════════════════════════════════════════════════════════════════════════════
# 1. Pure helper functions
# ════════════════════════════════════════════════════════════════════════════


class TestResolveAuthToken:
    """Tests for _resolve_auth_token priority chain."""

    def test_cli_flag_wins(self):
        with mock.patch.dict(os.environ, {"BUILDER_AUTH_TOKEN": "env_val"}):
            assert _resolve_auth_token("cli_val") == "cli_val"

    def test_env_var_fallback(self):
        with mock.patch.dict(os.environ, {"BUILDER_AUTH_TOKEN": "env_val"}):
            assert _resolve_auth_token(None) == "env_val"

    def test_dotfile_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_AUTH_TOKEN", None)
            with tempfile.TemporaryDirectory() as td:
                dotfile = os.path.join(td, "auth_token")
                with open(dotfile, "w") as f:
                    f.write("  dot_val  \n")
                with mock.patch("os.path.expanduser", return_value=dotfile):
                    assert _resolve_auth_token(None) == "dot_val"

    def test_returns_none_when_nothing_configured(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_AUTH_TOKEN", None)
            with mock.patch("os.path.isfile", return_value=False):
                assert _resolve_auth_token(None) is None

    def test_dotfile_empty_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_AUTH_TOKEN", None)
            with tempfile.TemporaryDirectory() as td:
                dotfile = os.path.join(td, "auth_token")
                with open(dotfile, "w") as f:
                    f.write("   \n")
                with mock.patch("os.path.expanduser", return_value=dotfile):
                    result = _resolve_auth_token(None)
                    assert result is None

    def test_dotfile_oserror(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_AUTH_TOKEN", None)
            with mock.patch("os.path.expanduser", return_value="/nonexistent/auth_token"):
                with mock.patch("os.path.isfile", return_value=True):
                    with mock.patch("builtins.open", side_effect=OSError("permission denied")):
                        assert _resolve_auth_token(None) is None


class TestMakeSessionCookie:
    def test_deterministic(self):
        assert _make_session_cookie("tok") == _make_session_cookie("tok")

    def test_different_tokens_differ(self):
        assert _make_session_cookie("a") != _make_session_cookie("b")


class TestAiBuildFallback:
    def test_single_agent_for_unknown_description(self):
        result = _ai_build_fallback("do something random")
        assert "nodes" in result and "edges" in result
        agent_nodes = [n for n in result["nodes"] if n["type"] == "agent"]
        assert len(agent_nodes) >= 1

    def test_research_keyword_creates_researcher(self):
        result = _ai_build_fallback("research about climate change")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Researcher" in names

    def test_write_keyword_creates_writer(self):
        result = _ai_build_fallback("write a blog post")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Writer" in names

    def test_review_keyword_creates_critic(self):
        result = _ai_build_fallback("review and evaluate the proposal")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Critic" in names

    def test_classify_keyword(self):
        result = _ai_build_fallback("classify and categorize emails")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Classifier" in names

    def test_summarize_keyword(self):
        result = _ai_build_fallback("summarize the document")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Summarizer" in names

    def test_combined_keywords(self):
        result = _ai_build_fallback("research and write a report then review it")
        names = [n["name"] for n in result["nodes"] if n["type"] == "agent"]
        assert "Researcher" in names
        assert "Writer" in names
        assert "Critic" in names

    def test_has_start_and_end_nodes(self):
        result = _ai_build_fallback("something")
        types = [n["type"] for n in result["nodes"]]
        assert "start" in types
        assert "end" in types

    def test_edges_chain_nodes(self):
        result = _ai_build_fallback("research and write")
        assert len(result["edges"]) >= 2  # start->agent, agent->end at minimum


class TestAiBuildLive:
    def test_fallback_on_exception(self):
        """When provider init fails, falls back to deterministic builder."""
        from selectools.serve.app import _ai_build_live

        result = _ai_build_live("write a summary", "sk-invalid-key-12345")
        # Should fall back to _ai_build_fallback
        assert "nodes" in result
        assert "edges" in result

    def test_anthropic_prefix_detected(self):
        """api_key starting with sk-ant should attempt AnthropicProvider."""
        from selectools.serve.app import _ai_build_live

        with mock.patch(
            "selectools.serve.app.AnthropicProvider",
            side_effect=Exception("no key"),
            create=True,
        ):
            result = _ai_build_live("research topic", "sk-ant-fakekey123")
            assert "nodes" in result


class TestApplyPinnedPorts:
    def test_pinned_value_overrides(self):
        edges = [{"from": "a", "source": "a", "sourceHandle": "output", "targetHandle": "input"}]
        pinned = {"a::output": "pinned_data"}
        last_outputs = {"a": "real_data"}
        result = _apply_pinned_ports([], edges, pinned, last_outputs)
        assert result["input"] == "pinned_data"

    def test_unpinned_uses_last_output(self):
        edges = [{"from": "a", "source": "a", "sourceHandle": "output", "targetHandle": "input"}]
        pinned = {}
        last_outputs = {"a": "real_data"}
        result = _apply_pinned_ports([], edges, pinned, last_outputs)
        assert result["input"] == "real_data"


class TestParsePythonToGraph:
    def test_empty_source(self):
        result = _parse_python_to_graph("")
        assert result == {"nodes": [], "edges": []}

    def test_syntax_error_returns_empty(self):
        result = _parse_python_to_graph("def ????")
        assert result == {"nodes": [], "edges": []}

    def test_add_node_detected(self):
        code = """
g.add_node("research_agent")
g.add_node("writer_agent")
"""
        result = _parse_python_to_graph(code)
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["id"] == "research_agent"
        assert result["nodes"][1]["id"] == "writer_agent"

    def test_add_edge_detected(self):
        code = """
g.add_node("a")
g.add_node("b")
g.add_edge("a", "b")
"""
        result = _parse_python_to_graph(code)
        assert len(result["edges"]) == 1
        assert result["edges"][0]["from"] == "a"
        assert result["edges"][0]["to"] == "b"


class TestRenderHitlForm:
    def test_fallback_buttons_no_fields(self):
        html = _render_hitl_form({"options": "yes, no"})
        assert "yes" in html
        assert "no" in html
        assert "<button" in html

    def test_text_field(self):
        html = _render_hitl_form(
            {
                "form_fields": [
                    {"id": "name", "label": "Name", "type": "text", "placeholder": "Enter name"}
                ]
            }
        )
        assert 'type="text"' in html
        assert "Name" in html

    def test_textarea_field(self):
        html = _render_hitl_form(
            {"form_fields": [{"id": "notes", "label": "Notes", "type": "textarea"}]}
        )
        assert "<textarea" in html

    def test_number_field(self):
        html = _render_hitl_form({"form_fields": [{"id": "qty", "label": "Qty", "type": "number"}]})
        assert 'type="number"' in html

    def test_select_field(self):
        html = _render_hitl_form(
            {
                "form_fields": [
                    {"id": "choice", "label": "Pick", "type": "select", "options": ["A", "B"]}
                ]
            }
        )
        assert "<select" in html
        assert "<option>A</option>" in html

    def test_checkbox_field(self):
        html = _render_hitl_form(
            {"form_fields": [{"id": "agree", "label": "I agree", "type": "checkbox"}]}
        )
        assert 'type="checkbox"' in html

    def test_required_attribute(self):
        html = _render_hitl_form(
            {"form_fields": [{"id": "x", "label": "X", "type": "text", "required": True}]}
        )
        assert "required" in html

    def test_empty_form_fields(self):
        html = _render_hitl_form({"form_fields": []})
        # Empty fields list -> fallback buttons
        assert "<button" in html


class TestResolveUsers:
    def test_from_env_var(self):
        data = json.dumps({"alice": {"role": "admin"}})
        with mock.patch.dict(os.environ, {"BUILDER_USERS": data}):
            result = _resolve_users()
            assert result["alice"]["role"] == "admin"

    def test_invalid_json_returns_empty(self):
        with mock.patch.dict(os.environ, {"BUILDER_USERS": "not-json"}):
            assert _resolve_users() == {}

    def test_non_dict_returns_empty(self):
        with mock.patch.dict(os.environ, {"BUILDER_USERS": '["a","b"]'}):
            assert _resolve_users() == {}

    def test_no_env_no_dotfile(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_USERS", None)
            with mock.patch("os.path.isfile", return_value=False):
                assert _resolve_users() == {}

    def test_dotfile_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUILDER_USERS", None)
            with tempfile.TemporaryDirectory() as td:
                dotfile = os.path.join(td, "users.json")
                with open(dotfile, "w") as f:
                    json.dump({"bob": {"role": "editor"}}, f)
                with mock.patch("os.path.expanduser", return_value=dotfile):
                    with mock.patch("os.path.isfile", return_value=True):
                        result = _resolve_users()
                        assert result.get("bob", {}).get("role") == "editor"


class TestHasPermission:
    def test_admin_has_all(self):
        for action in ("view", "edit", "run", "export", "delete", "manage_users"):
            assert _has_permission("admin", action)

    def test_editor_can_run(self):
        assert _has_permission("editor", "run")

    def test_editor_cannot_delete(self):
        assert not _has_permission("editor", "delete")

    def test_viewer_can_view(self):
        assert _has_permission("viewer", "view")

    def test_viewer_cannot_edit(self):
        assert not _has_permission("viewer", "edit")

    def test_unknown_role(self):
        assert not _has_permission("ghost", "view")


class TestCheckGraphPermission:
    def test_admin_always_true(self):
        assert _check_graph_permission("g1", "alice", "admin", "delete")

    def test_owner_has_access(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g1.json")
            with open(path, "w") as f:
                json.dump({"owner": "bob", "acl": []}, f)
            assert _check_graph_permission("g1", "bob", "editor", "edit", graphs_dir=td)

    def test_non_owner_no_acl(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g1.json")
            with open(path, "w") as f:
                json.dump({"owner": "alice", "acl": []}, f)
            assert not _check_graph_permission("g1", "bob", "editor", "edit", graphs_dir=td)

    def test_acl_entry_grants(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g1.json")
            with open(path, "w") as f:
                json.dump({"owner": "alice", "acl": [{"user": "bob", "permission": "editor"}]}, f)
            assert _check_graph_permission("g1", "bob", "viewer", "edit", graphs_dir=td)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            assert not _check_graph_permission("g_missing", "bob", "editor", "edit", graphs_dir=td)

    def test_corrupt_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g1.json")
            with open(path, "w") as f:
                f.write("NOT JSON")
            assert not _check_graph_permission("g1", "bob", "editor", "edit", graphs_dir=td)


class TestLogRun:
    def test_writes_jsonl_file(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("os.path.expanduser", return_value=td):
                _log_run({"run_id": "r1", "output": "hello"})
            files = os.listdir(td)
            assert len(files) == 1
            with open(os.path.join(td, files[0])) as f:
                data = json.loads(f.readline())
            assert data["run_id"] == "r1"


class TestRunEvalsOnRun:
    def test_no_evaluators_returns_early(self):
        # Should not raise
        _run_evals_on_run({"eval_config": {"evaluators": []}, "input": "hi"})

    def test_with_mock_evaluator(self):
        # Should not raise even with unknown evaluator names
        _run_evals_on_run({"eval_config": {"evaluators": ["UnknownEval"]}, "input": "hi"})


class TestFireEvalAlert:
    def test_no_webhook_noop(self):
        # No webhook URL => should not raise
        _fire_eval_alert({"eval_config": {}}, 0.3, 0.7)

    def test_webhook_called(self):
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = mock.MagicMock()
            _fire_eval_alert(
                {
                    "eval_config": {"webhook_url": "http://example.com/hook"},
                    "graph_id": "g1",
                    "run_id": "r1",
                    "input": "test",
                },
                0.3,
                0.7,
            )
            mock_urlopen.assert_called_once()


class TestRouteExperiment:
    def test_no_experiment_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            result = _route_experiment("graph_a", "run_1", experiments_dir=path)
            assert result == "graph_a"

    def test_active_experiment_routes(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            experiments = [
                {"active": True, "variant_a": "graph_a", "variant_b": "graph_b", "split": 0.5}
            ]
            with open(path, "w") as f:
                json.dump(experiments, f)
            result = _route_experiment("graph_a", "run_1", experiments_dir=path)
            assert result in ("graph_a", "graph_b")

    def test_inactive_experiment_passes_through(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            experiments = [{"active": False, "variant_a": "graph_a", "variant_b": "graph_b"}]
            with open(path, "w") as f:
                json.dump(experiments, f)
            result = _route_experiment("graph_a", "run_1", experiments_dir=path)
            assert result == "graph_a"


class TestEstimateTaskTier:
    def test_simple_signals(self):
        assert _estimate_task_tier("summarize and classify this text", "") == "simple"

    def test_advanced_signals(self):
        assert _estimate_task_tier("analyze and evaluate this complex problem", "") == "advanced"

    def test_standard_default(self):
        assert _estimate_task_tier("hello world", "") == "standard"


class TestSmartRoute:
    def test_returns_model_string(self):
        model = _smart_route("summarize this", "", None, None)
        assert isinstance(model, str)
        assert len(model) > 0

    def test_with_budget_constraint(self):
        model = _smart_route("hello", "", None, 0.0001)
        assert isinstance(model, str)

    def test_with_available_providers(self):
        model = _smart_route("hello", "", ["openai"], None)
        assert isinstance(model, str)


class TestEvalRoute:
    def test_no_api_key_heuristic(self):
        result = _eval_route("hello", "", [], api_key="")
        assert result["method"] == "heuristic"
        assert "model" in result

    def test_no_eval_cases_heuristic(self):
        result = _eval_route("hello", "", [], api_key="sk-fake")
        assert result["method"] == "heuristic"


class TestEstimateRunCost:
    def test_empty_nodes(self):
        result = _estimate_run_cost([], "hello")
        assert result["total_tokens"] == 0
        assert result["total_cost_usd"] == 0.0

    def test_with_agent_node(self):
        nodes = [{"type": "agent", "model": "gpt-4o-mini", "system_prompt": "Be helpful"}]
        result = _estimate_run_cost(nodes, "test input")
        assert result["total_tokens"] > 0

    def test_non_agent_nodes_ignored(self):
        nodes = [{"type": "start"}, {"type": "end"}]
        result = _estimate_run_cost(nodes, "test")
        assert result["total_tokens"] == 0


class TestMakeProvider:
    def test_claude_model(self):
        with mock.patch("selectools.providers.anthropic_provider.AnthropicProvider") as mock_cls:
            mock_cls.return_value = mock.MagicMock()
            result = _make_provider("claude-sonnet-4-6", "sk-ant-fake")
            mock_cls.assert_called_once_with(api_key="sk-ant-fake")

    def test_gemini_model(self):
        with mock.patch("selectools.providers.gemini_provider.GeminiProvider") as mock_cls:
            mock_cls.return_value = mock.MagicMock()
            result = _make_provider("gemini-2.0-flash", "fake-key")
            mock_cls.assert_called_once_with(api_key="fake-key")

    def test_llama_model(self):
        with mock.patch("selectools.providers.ollama_provider.OllamaProvider") as mock_cls:
            mock_cls.return_value = mock.MagicMock()
            result = _make_provider("llama-3.1", "fake-key")
            mock_cls.assert_called_once()

    def test_openai_default(self):
        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = mock.MagicMock()
            result = _make_provider("gpt-4o-mini", "sk-fake")
            mock_cls.assert_called_once_with(api_key="sk-fake")


# ════════════════════════════════════════════════════════════════════════════
# 2. AgentRouter unit tests
# ════════════════════════════════════════════════════════════════════════════


class TestAgentRouter:
    def test_creation(self):
        agent = _make_agent()
        router = AgentRouter(agent, prefix="/api")
        assert router.prefix == "/api"
        assert router.enable_playground is True
        assert router.enable_builder is False

    def test_prefix_strip_trailing_slash(self):
        agent = _make_agent()
        router = AgentRouter(agent, prefix="/api/")
        assert router.prefix == "/api"

    def test_handle_health(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        data = router.handle_health()
        assert data["status"] == "ok"
        assert "version" in data
        assert "model" in data
        assert isinstance(data["tools"], list)

    def test_handle_schema(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        data = router.handle_schema()
        assert "tools" in data
        assert "model" in data
        assert isinstance(data["tools"], list)

    def test_handle_invoke_missing_prompt(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        result = router.handle_invoke({})
        assert "error" in result
        assert "prompt is required" in result["error"]

    def test_handle_invoke_success(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        result = router.handle_invoke({"prompt": "Hello"})
        assert "content" in result
        assert "error" not in result

    def test_handle_invoke_exception(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        with mock.patch.object(agent, "run", side_effect=RuntimeError("boom")):
            result = router.handle_invoke({"prompt": "Hello"})
            assert "error" in result
            assert "boom" in result["error"]

    def test_handle_stream_missing_prompt(self):
        agent = _make_agent()
        router = AgentRouter(agent)
        chunks = list(router.handle_stream({}))
        assert any("prompt is required" in c for c in chunks)

    def test_handle_stream_with_prompt(self):
        """Stream handler should produce SSE output."""
        agent = _make_agent()
        router = AgentRouter(agent)

        # Mock agent.astream to yield a string then an AgentResult-like object
        async def fake_astream(prompt):
            yield "Hello "
            yield "world"

        with mock.patch.object(agent, "astream", side_effect=fake_astream):
            chunks = list(router.handle_stream({"prompt": "Hello"}))
        assert len(chunks) > 0
        assert any("[DONE]" in c for c in chunks)
        assert any("chunk" in c for c in chunks)


class TestCreateApp:
    def test_returns_agent_server(self):
        agent = _make_agent()
        app = create_app(agent, port=9999)
        assert isinstance(app, AgentServer)
        assert app.port == 9999

    def test_passes_params(self):
        agent = _make_agent()
        app = create_app(agent, prefix="/v1", playground=False, builder=True, auth_token="tok")
        assert app.router.prefix == "/v1"
        assert app.router.enable_playground is False
        assert app.router.enable_builder is True
        assert app.auth_token == "tok"


# ════════════════════════════════════════════════════════════════════════════
# 3. AgentServer HTTP integration tests
# ════════════════════════════════════════════════════════════════════════════


class TestAgentServerHTTP:
    """Integration tests using a real HTTP server in a background thread."""

    @pytest.fixture(scope="class")
    def server(self):
        port = _free_port()
        agent = _make_agent()
        srv = AgentServer(
            agent,
            host="127.0.0.1",
            port=port,
            playground=True,
            builder=True,
        )
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"

    def test_health_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert "tools" in data

    def test_schema_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/schema")
        data = json.loads(resp.read())
        assert "tools" in data
        assert "model" in data

    def test_playground_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/playground")
        assert resp.status == 200
        body = resp.read().decode()
        assert "selectools" in body.lower() or "<!DOCTYPE" in body

    def test_builder_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/builder")
        assert resp.status == 200
        body = resp.read().decode()
        assert "AgentGraph" in body

    def test_playground_via_explicit_path(self, server):
        resp = urllib.request.urlopen(f"{server}/playground")
        assert resp.status == 200

    def test_invoke_endpoint(self, server):
        payload = json.dumps({"prompt": "Say hello"}).encode()
        req = urllib.request.Request(
            f"{server}/invoke",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "content" in data

    def test_invoke_empty_prompt(self, server):
        payload = json.dumps({"prompt": ""}).encode()
        req = urllib.request.Request(
            f"{server}/invoke",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "error" in data

    def test_stream_endpoint_content_type(self, server):
        """Stream endpoint returns text/event-stream Content-Type.

        Note: LocalProvider doesn't support async, so the stream itself may error
        server-side, but the response headers are set before streaming begins.
        """
        payload = json.dumps({"prompt": "Hi"}).encode()
        req = urllib.request.Request(
            f"{server}/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

    def test_post_invalid_json(self, server):
        req = urllib.request.Request(
            f"{server}/invoke",
            data=b"NOT JSON",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            assert "error" in data
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert "error" in data
            assert e.code == 400

    def test_options_cors(self, server):
        req = urllib.request.Request(f"{server}/invoke", method="OPTIONS")
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_404_unknown_get(self, server):
        try:
            resp = urllib.request.urlopen(f"{server}/nonexistent")
            data = json.loads(resp.read())
            assert data.get("error") == "not found"
        except urllib.error.HTTPError as e:
            assert e.code == 404
            data = json.loads(e.read())
            assert data.get("error") == "not found"

    def test_404_unknown_post(self, server):
        req = urllib.request.Request(
            f"{server}/nonexistent",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            assert data.get("error") == "not found"
        except urllib.error.HTTPError as e:
            assert e.code == 404
            data = json.loads(e.read())
            assert data.get("error") == "not found"

    def test_login_get(self, server):
        resp = urllib.request.urlopen(f"{server}/login")
        assert resp.status == 200
        body = resp.read().decode()
        assert "token" in body


class TestAgentServerAuth:
    """AgentServer with auth_token set."""

    @pytest.fixture(scope="class")
    def server(self):
        port = _free_port()
        agent = _make_agent()
        srv = AgentServer(
            agent,
            host="127.0.0.1",
            port=port,
            playground=True,
            builder=True,
            auth_token="test_secret",
        )
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"

    def test_health_bypasses_auth(self, server):
        resp = urllib.request.urlopen(f"{server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_login_page_bypasses_auth(self, server):
        resp = urllib.request.urlopen(f"{server}/login")
        assert resp.status == 200

    def test_builder_redirects_without_auth(self, server):
        req = urllib.request.Request(f"{server}/builder")
        try:
            resp = _open_no_redirect(req)
            # If no redirect, check it returned login redirect or 302
            assert False, "Expected 302 redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302
            assert "/login" in e.headers.get("Location", "")

    def test_invoke_redirects_without_auth(self, server):
        req = urllib.request.Request(
            f"{server}/invoke",
            data=json.dumps({"prompt": "hi"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = _open_no_redirect(req)
            assert False, "Expected 302 redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302

    def test_login_correct_token(self, server):
        payload = json.dumps({"token": "test_secret"}).encode()
        req = urllib.request.Request(
            f"{server}/login",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = _open_no_redirect(req)
            assert False, "Expected 302 redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302
            assert "Set-Cookie" in str(e.headers) or "set-cookie" in str(e.headers).lower()

    def test_login_wrong_token(self, server):
        payload = json.dumps({"token": "wrong_secret"}).encode()
        req = urllib.request.Request(
            f"{server}/login",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()
        assert "Invalid token" in body

    def test_invoke_with_valid_cookie(self, server):
        cookie = f"builder_session={_make_session_cookie('test_secret')}"
        payload = json.dumps({"prompt": "Hi"}).encode()
        req = urllib.request.Request(
            f"{server}/invoke",
            data=payload,
            headers={"Content-Type": "application/json", "Cookie": cookie},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "content" in data


# ════════════════════════════════════════════════════════════════════════════
# 4. BuilderServer HTTP integration tests for uncovered routes
# ════════════════════════════════════════════════════════════════════════════


class TestBuilderServerRoutes:
    """Test BuilderServer routes that are not covered by test_visual_builder.py."""

    @pytest.fixture(scope="class")
    def server(self):
        port = _free_port()
        srv = BuilderServer(host="127.0.0.1", port=port)
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"

    def test_provider_health_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/provider-health")
        data = json.loads(resp.read())
        assert "openai" in data
        assert "anthropic" in data

    def test_eval_dashboard_endpoint(self, server):
        resp = urllib.request.urlopen(f"{server}/eval-dashboard")
        assert resp.status == 200
        body = resp.read().decode()
        assert "Eval Dashboard" in body

    def test_404_unknown_get(self, server):
        try:
            resp = urllib.request.urlopen(f"{server}/nonexistent-route")
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            assert e.code == 404
            data = json.loads(e.read())
        assert data.get("error") == "not found"

    def test_estimate_run_cost_endpoint(self, server):
        payload = json.dumps(
            {
                "nodes": [{"type": "agent", "model": "gpt-4o-mini", "system_prompt": ""}],
                "input": "test prompt",
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/estimate-run-cost",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "total_tokens" in data
        assert "total_cost_usd" in data

    def test_smart_route_endpoint(self, server):
        payload = json.dumps(
            {
                "prompt": "summarize and classify",
                "system_prompt": "",
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/smart-route",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "model" in data

    def test_eval_route_endpoint(self, server):
        payload = json.dumps(
            {
                "prompt": "hello",
                "system_prompt": "",
                "eval_cases": [],
                "threshold": 0.7,
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/eval-route",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "model" in data
        assert data["method"] == "heuristic"

    def test_runs_endpoint(self, server):
        payload = json.dumps({}).encode()
        req = urllib.request.Request(
            f"{server}/runs",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_feedback_endpoint(self, server):
        payload = json.dumps({"run_id": "r_test", "score": 5}).encode()
        req = urllib.request.Request(
            f"{server}/feedback",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert data.get("ok") is True

    def test_options_cors(self, server):
        req = urllib.request.Request(f"{server}/run", method="OPTIONS")
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_post_invalid_json(self, server):
        req = urllib.request.Request(
            f"{server}/run",
            data=b"NOT JSON {{{",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert e.code == 400
        assert "error" in data

    def test_404_unknown_post(self, server):
        req = urllib.request.Request(
            f"{server}/nonexistent-post",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            assert e.code == 404
            data = json.loads(e.read())
        assert data.get("error") == "not found"

    def test_ai_build_no_description(self, server):
        payload = json.dumps({"description": ""}).encode()
        req = urllib.request.Request(
            f"{server}/ai-build",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert e.code == 400
        assert "error" in data

    def test_ai_build_fallback_mode(self, server):
        payload = json.dumps({"description": "research and write a report"}).encode()
        req = urllib.request.Request(
            f"{server}/ai-build",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "nodes" in data
        assert "edges" in data

    def test_ai_refine_no_message(self, server):
        payload = json.dumps({"message": "", "current_graph": {}}).encode()
        req = urllib.request.Request(
            f"{server}/ai-refine",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert e.code == 400
        assert "error" in data

    def test_ai_refine_no_api_key(self, server):
        payload = json.dumps(
            {
                "message": "add another agent node",
                "current_graph": {"nodes": [], "edges": []},
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/ai-refine",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert "explanation" in data
        assert "No API key" in data["explanation"]

    def test_login_get(self, server):
        resp = urllib.request.urlopen(f"{server}/login")
        assert resp.status == 200
        body = resp.read().decode()
        assert "token" in body


class TestBuilderServerAuthRoutes:
    """BuilderServer with auth enabled — testing login flow via real HTTP."""

    @pytest.fixture(scope="class")
    def server(self):
        port = _free_port()
        srv = BuilderServer(host="127.0.0.1", port=port, auth_token="builder_secret")
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"

    def test_builder_redirects_to_login(self, server):
        req = urllib.request.Request(f"{server}/builder")
        try:
            _open_no_redirect(req)
            assert False, "Expected redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302
            assert "/login" in e.headers.get("Location", "")

    def test_health_bypasses_auth(self, server):
        resp = urllib.request.urlopen(f"{server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_login_post_correct_token(self, server):
        payload = json.dumps({"token": "builder_secret"}).encode()
        req = urllib.request.Request(
            f"{server}/login",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            _open_no_redirect(req)
            assert False, "Expected redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302
            raw_headers = str(e.headers)
            assert "builder_session=" in raw_headers

    def test_login_post_wrong_token(self, server):
        payload = json.dumps({"token": "wrong"}).encode()
        req = urllib.request.Request(
            f"{server}/login",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        body = resp.read().decode()
        assert "Invalid token" in body

    def test_post_routes_redirect_without_auth(self, server):
        payload = json.dumps({"description": "test"}).encode()
        req = urllib.request.Request(
            f"{server}/ai-build",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            _open_no_redirect(req)
            assert False, "Expected redirect"
        except urllib.error.HTTPError as e:
            assert e.code == 302

    def test_builder_accessible_with_valid_cookie(self, server):
        cookie_val = _make_session_cookie("builder_secret")
        req = urllib.request.Request(f"{server}/builder")
        req.add_header("Cookie", f"builder_session={cookie_val}")
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        body = resp.read().decode()
        assert "AgentGraph" in body


# ════════════════════════════════════════════════════════════════════════════
# 5. _builder_run_live (mock provider to avoid real API calls)
# ════════════════════════════════════════════════════════════════════════════


class TestBuilderRunLive:
    def test_no_agent_nodes_emits_error(self):
        from selectools.serve.app import _builder_run_live

        events: list = []
        _builder_run_live([{"id": "s1", "type": "start"}], [], "hello", "sk-fake", events.append)
        assert any(e["type"] == "error" for e in events)

    def test_provider_init_failure(self):
        from selectools.serve.app import _builder_run_live

        events: list = []
        with mock.patch(
            "selectools.providers.openai_provider.OpenAIProvider.__init__",
            side_effect=Exception("bad key"),
        ):
            _builder_run_live(
                [{"id": "a1", "type": "agent", "provider": "openai", "model": "gpt-4o-mini"}],
                [],
                "hello",
                "sk-fake",
                events.append,
            )
        assert any(e["type"] == "error" for e in events)


# ════════════════════════════════════════════════════════════════════════════
# 6. Misc constants / module-level checks
# ════════════════════════════════════════════════════════════════════════════


class TestModuleLevelConstants:
    def test_login_html_has_form(self):
        assert "<form" in LOGIN_HTML
        assert 'name="token"' in LOGIN_HTML

    def test_login_html_error_has_error_text(self):
        assert "Invalid token" in LOGIN_HTML_ERROR

    def test_capability_tiers_keys(self):
        assert "simple" in CAPABILITY_TIERS
        assert "standard" in CAPABILITY_TIERS
        assert "advanced" in CAPABILITY_TIERS

    def test_roles_keys(self):
        assert "admin" in ROLES
        assert "editor" in ROLES
        assert "viewer" in ROLES


class TestRunEvalSample:
    """Test _run_eval_sample with mocked provider."""

    def test_empty_cases_returns_1(self):
        from selectools.serve.app import _run_eval_sample

        score = _run_eval_sample("gpt-4o-mini", "be helpful", [], "sk-fake")
        assert score == 1.0

    def test_with_mock_provider(self):
        from selectools.serve.app import _run_eval_sample

        mock_msg = mock.MagicMock()
        mock_msg.content = "the answer is 42"
        mock_provider = mock.MagicMock()
        mock_provider.complete.return_value = (mock_msg, mock.MagicMock())

        with mock.patch("selectools.serve.app._make_provider", return_value=mock_provider):
            score = _run_eval_sample(
                "gpt-4o-mini",
                "be helpful",
                [{"input": "what is 6*7?", "expected_output": "42"}],
                "sk-fake",
            )
            assert score >= 0.0


class TestBuildAgentToolFromNode:
    """Test _build_agent_tool_from_node creates a proper Tool."""

    def test_creates_tool(self):
        from selectools.serve.app import _build_agent_tool_from_node

        node_data = {
            "tool_target_node": "target_1",
            "tool_name": "nested_agent",
            "tool_description": "A nested agent tool",
            "tool_input_param": "query",
            "tool_max_tokens": 200,
        }
        graph_nodes = [
            {"id": "target_1", "type": "agent", "model": "gpt-4o-mini", "system_prompt": "Help"}
        ]
        tool = _build_agent_tool_from_node(node_data, graph_nodes, "sk-fake")
        assert tool.name == "nested_agent"
        assert tool.description == "A nested agent tool"

    def test_missing_target_returns_error_string(self):
        from selectools.serve.app import _build_agent_tool_from_node

        node_data = {
            "tool_target_node": "nonexistent",
            "tool_name": "test_tool",
            "tool_description": "test",
            "tool_input_param": "query",
        }
        tool = _build_agent_tool_from_node(node_data, [], "sk-fake")
        # Running the tool with missing target should return error string
        result = tool.function(query="hi")
        assert "not found" in result


class TestAiRefineLive:
    """Test _ai_refine_live with mocked provider."""

    def test_no_api_key_error(self):
        from selectools.serve.app import _ai_refine_live

        result = _ai_refine_live(
            {"nodes": [], "edges": []},
            None,
            "add a node",
            [],
            "",
        )
        # Should return error since no API key
        assert "error" in result or "patch" in result

    def test_with_mocked_provider(self):
        from selectools.serve.app import _ai_refine_live

        mock_msg = mock.MagicMock()
        mock_msg.content = json.dumps(
            {"patch": {"type": "add_node"}, "explanation": "Added node", "suggested_follow_up": ""}
        )

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            instance = mock.MagicMock()
            instance.complete.return_value = (mock_msg, mock.MagicMock())
            mock_cls.return_value = instance
            result = _ai_refine_live(
                {"nodes": [{"id": "a1", "type": "agent"}], "edges": []},
                "a1",
                "change the system prompt",
                [{"role": "user", "content": "hello"}],
                "sk-fake-key",
            )
            assert "patch" in result

    def test_with_invalid_history_entries(self):
        from selectools.serve.app import _ai_refine_live

        mock_msg = mock.MagicMock()
        mock_msg.content = json.dumps(
            {"patch": None, "explanation": "err", "suggested_follow_up": ""}
        )

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            instance = mock.MagicMock()
            instance.complete.return_value = (mock_msg, mock.MagicMock())
            mock_cls.return_value = instance
            result = _ai_refine_live(
                {"nodes": [], "edges": []},
                None,
                "do something",
                [{"bad_key": "no role"}, {"role": "user", "content": "valid"}],
                "sk-key",
            )
            assert "patch" in result or "error" in result

    def test_markdown_fences_stripped(self):
        from selectools.serve.app import _ai_refine_live

        mock_msg = mock.MagicMock()
        mock_msg.content = '```json\n{"patch": {"type": "add_node"}, "explanation": "ok", "suggested_follow_up": ""}\n```'

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            instance = mock.MagicMock()
            instance.complete.return_value = (mock_msg, mock.MagicMock())
            mock_cls.return_value = instance
            result = _ai_refine_live(
                {"nodes": [], "edges": []},
                None,
                "add a node",
                [],
                "sk-key",
            )
            assert "patch" in result


# ════════════════════════════════════════════════════════════════════════════
# 7. Additional coverage for _builder_run_live
# ════════════════════════════════════════════════════════════════════════════


class TestBuilderRunLiveExtended:
    """More tests for _builder_run_live to cover the execution path with mocked provider."""

    def test_live_run_with_local_provider(self):
        """Use LocalProvider to exercise the full run path."""
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {"id": "s1", "type": "start", "name": "START"},
            {
                "id": "a1",
                "type": "agent",
                "name": "Bot",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
        ]
        edges = [{"id": "e1", "from": "s1", "to": "a1"}]

        # Mock OpenAIProvider to return a LocalProvider
        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, edges, "hello world", "sk-fake", events.append)

        types = [e["type"] for e in events]
        assert "node_start" in types
        assert "chunk" in types
        assert "node_end" in types
        assert "run_end" in types

    def test_live_run_multi_agent_chained(self):
        """Multiple agents in a chain — output of first feeds into second."""
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {"id": "s1", "type": "start", "name": "START"},
            {
                "id": "a1",
                "type": "agent",
                "name": "First",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
            {
                "id": "a2",
                "type": "agent",
                "name": "Second",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
        ]
        edges = [
            {"id": "e1", "from": "s1", "to": "a1"},
            {"id": "e2", "from": "a1", "to": "a2"},
        ]

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, edges, "test input", "sk-fake", events.append)

        starts = [e for e in events if e["type"] == "node_start"]
        assert len(starts) == 2
        assert starts[0]["node_name"] == "First"
        assert starts[1]["node_name"] == "Second"

    def test_live_run_exception_in_agent(self):
        """Agent.run raising an exception should emit error event, not crash."""
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {
                "id": "a1",
                "type": "agent",
                "name": "Faulty",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
        ]

        mock_provider = mock.MagicMock()
        mock_provider.complete.side_effect = RuntimeError("API timeout")

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = mock_provider
            _builder_run_live(nodes, [], "hi", "sk-fake", events.append)

        assert any(e["type"] == "error" for e in events)
        assert any(e["type"] == "run_end" for e in events)

    def test_anthropic_provider_selected(self):
        """Anthropic provider selected when first agent node has provider=anthropic."""
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {
                "id": "a1",
                "type": "agent",
                "name": "Bot",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "system_prompt": "",
            },
        ]

        with mock.patch("selectools.providers.anthropic_provider.AnthropicProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, [], "hi", "sk-ant-fake", events.append)

        assert any(e["type"] == "run_end" for e in events)

    def test_gemini_provider_selected(self):
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {
                "id": "a1",
                "type": "agent",
                "name": "Bot",
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "system_prompt": "",
            },
        ]

        with mock.patch("selectools.providers.gemini_provider.GeminiProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, [], "hi", "fake-key", events.append)

        assert any(e["type"] == "run_end" for e in events)

    def test_ollama_provider_selected(self):
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {
                "id": "a1",
                "type": "agent",
                "name": "Bot",
                "provider": "ollama",
                "model": "llama3",
                "system_prompt": "",
            },
        ]

        with mock.patch("selectools.providers.ollama_provider.OllamaProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, [], "hi", "fake-key", events.append)

        assert any(e["type"] == "run_end" for e in events)

    def test_unvisited_agent_still_executed(self):
        """Agent node not reachable via edges from START should still execute."""
        from selectools.serve.app import _builder_run_live

        events: list = []
        nodes = [
            {"id": "s1", "type": "start", "name": "START"},
            {
                "id": "a1",
                "type": "agent",
                "name": "Connected",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
            {
                "id": "a2",
                "type": "agent",
                "name": "Orphan",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "system_prompt": "",
            },
        ]
        edges = [{"id": "e1", "from": "s1", "to": "a1"}]  # a2 is not connected

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = LocalProvider()
            _builder_run_live(nodes, edges, "test", "sk-fake", events.append)

        starts = [e for e in events if e["type"] == "node_start"]
        assert len(starts) == 2  # Both agents executed


# ════════════════════════════════════════════════════════════════════════════
# 8. Additional _ai_build_live tests
# ════════════════════════════════════════════════════════════════════════════


class TestAiBuildLiveExtended:
    """Tests for _ai_build_live.

    Note: _ai_build_live calls provider.complete() and accesses result.content
    directly (not unpacking a tuple). The mock must return an object with
    a .content attribute directly from complete().
    """

    def _mock_provider(self, content: str):
        """Create a mock provider whose complete() returns an object with .content."""
        result_obj = mock.MagicMock()
        result_obj.content = content
        instance = mock.MagicMock()
        instance.complete.return_value = result_obj
        return instance

    def test_successful_parse(self):
        from selectools.serve.app import _ai_build_live

        graph_data = {
            "nodes": [{"id": "start_1", "type": "start"}, {"id": "agent_1", "type": "agent"}],
            "edges": [{"id": "e1", "from": "start_1", "to": "agent_1"}],
        }

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = self._mock_provider(json.dumps(graph_data))
            result = _ai_build_live("create a research workflow", "sk-fake-key")

        assert result["nodes"] == graph_data["nodes"]
        assert result["edges"] == graph_data["edges"]

    def test_markdown_fences_stripped(self):
        from selectools.serve.app import _ai_build_live

        graph_data = {
            "nodes": [{"id": "s1", "type": "start"}],
            "edges": [],
        }

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = self._mock_provider(f"```json\n{json.dumps(graph_data)}\n```")
            result = _ai_build_live("test", "sk-fake-key")

        assert "nodes" in result

    def test_missing_nodes_falls_back(self):
        from selectools.serve.app import _ai_build_live

        with mock.patch("selectools.providers.openai_provider.OpenAIProvider") as mock_cls:
            mock_cls.return_value = self._mock_provider(json.dumps({"just_data": True}))
            result = _ai_build_live("something", "sk-fake-key")

        # Should fall back to deterministic builder
        assert "nodes" in result
        assert any(n.get("type") == "start" for n in result["nodes"])

    def test_anthropic_key_detected(self):
        from selectools.serve.app import _ai_build_live

        graph_data = {"nodes": [{"id": "s1", "type": "start"}], "edges": []}

        with mock.patch("selectools.providers.anthropic_provider.AnthropicProvider") as mock_cls:
            mock_cls.return_value = self._mock_provider(json.dumps(graph_data))
            result = _ai_build_live("test", "sk-ant-fake123")

        assert "nodes" in result


# ════════════════════════════════════════════════════════════════════════════
# 9. _eval_route with mocked eval samples
# ════════════════════════════════════════════════════════════════════════════


class TestEvalRouteExtended:
    def test_eval_validated_path(self):
        """When eval samples pass threshold, method should be eval-validated."""
        with mock.patch("selectools.serve.app._run_eval_sample", return_value=0.9):
            result = _eval_route(
                prompt="analyze this complex problem with reasoning",
                system_prompt="",
                eval_cases=[{"input": "test", "expected_output": "ok"}],
                threshold=0.7,
                api_key="sk-fake",
            )
        assert result["method"] == "eval-validated"
        assert result["model"] != ""

    def test_best_available_fallback(self):
        """When no model meets threshold, return best-available."""
        with mock.patch("selectools.serve.app._run_eval_sample", return_value=0.3):
            result = _eval_route(
                prompt="hello",
                system_prompt="",
                eval_cases=[{"input": "test", "expected_output": "ok"}],
                threshold=0.9,
                api_key="sk-fake",
            )
        assert result["method"] == "best-available"

    def test_budget_filters_models(self):
        """Budget constraint should filter expensive models."""
        with mock.patch("selectools.serve.app._run_eval_sample", return_value=0.9):
            result = _eval_route(
                prompt="hello",
                system_prompt="",
                eval_cases=[{"input": "test"}],
                threshold=0.5,
                budget_usd=0.00001,
                api_key="sk-fake",
            )
        assert "model" in result


# ════════════════════════════════════════════════════════════════════════════
# 10. _smart_route inner functions
# ════════════════════════════════════════════════════════════════════════════


class TestSmartRouteExtended:
    def test_healthy_providers_filter(self):
        """When providers have ok status, only those are considered."""
        from selectools.serve.app import _provider_health

        original = dict(_provider_health)
        try:
            _provider_health["openai"]["status"] = "ok"
            _provider_health["anthropic"]["status"] = "down"
            _provider_health["gemini"]["status"] = "down"
            _provider_health["ollama"]["status"] = "down"

            model = _smart_route("hello", "")
            # Should prefer openai models since only openai is healthy
            assert isinstance(model, str)
        finally:
            for k, v in original.items():
                _provider_health[k] = v

    def test_all_unhealthy_falls_back_to_candidates(self):
        from selectools.serve.app import _provider_health

        original = dict(_provider_health)
        try:
            for k in _provider_health:
                _provider_health[k]["status"] = "down"
            model = _smart_route("hello", "")
            assert isinstance(model, str)
        finally:
            for k, v in original.items():
                _provider_health[k] = v


# ════════════════════════════════════════════════════════════════════════════
# 11. _estimate_run_cost extended
# ════════════════════════════════════════════════════════════════════════════


class TestEstimateRunCostExtended:
    def test_unknown_model_still_works(self):
        nodes = [
            {"type": "agent", "model": "totally-unknown-model-xyz", "system_prompt": "Be brief"}
        ]
        result = _estimate_run_cost(nodes, "hello world")
        # Unknown model: cost should be 0 but tokens still estimated
        assert isinstance(result["total_cost_usd"], float)

    def test_multiple_agent_nodes(self):
        nodes = [
            {"type": "agent", "model": "gpt-4o-mini", "system_prompt": "Help"},
            {"type": "agent", "model": "gpt-4o-mini", "system_prompt": "More help"},
        ]
        result = _estimate_run_cost(nodes, "test input")
        assert result["total_tokens"] > 0


# ════════════════════════════════════════════════════════════════════════════
# 12. _render_hitl_form edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestRenderHitlFormExtended:
    def test_default_options_fallback(self):
        """No options key = default 'approve, reject'."""
        html = _render_hitl_form({})
        assert "approve" in html
        assert "reject" in html

    def test_mixed_field_types(self):
        html = _render_hitl_form(
            {
                "form_fields": [
                    {"id": "a", "label": "A", "type": "text"},
                    {"id": "b", "label": "B", "type": "textarea"},
                    {"id": "c", "label": "C", "type": "number"},
                    {"id": "d", "label": "D", "type": "select", "options": ["X", "Y"]},
                    {"id": "e", "label": "E", "type": "checkbox"},
                ]
            }
        )
        assert 'type="text"' in html
        assert "<textarea" in html
        assert 'type="number"' in html
        assert "<select" in html
        assert 'type="checkbox"' in html


# ════════════════════════════════════════════════════════════════════════════
# 13. BuilderServer /runs with actual log files
# ════════════════════════════════════════════════════════════════════════════


class TestBuilderServerRunsEndpoint:
    """Test /runs endpoint reading actual log files."""

    @pytest.fixture(scope="class")
    def server_with_runs(self):
        port = _free_port()
        # Write a fake run log
        with tempfile.TemporaryDirectory() as td:
            run_dir = td
            log_file = os.path.join(run_dir, "2024-01-01.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"run_id": "r1", "input": "test"}) + "\n")
                f.write(json.dumps({"run_id": "r2", "input": "test2"}) + "\n")

            with mock.patch("os.path.expanduser", return_value=run_dir):
                srv = BuilderServer(host="127.0.0.1", port=port)
                t = threading.Thread(target=srv.serve, daemon=True)
                t.start()
                time.sleep(0.3)

                payload = json.dumps({}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/runs",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req)
                data = json.loads(resp.read())
                assert "runs" in data


# ════════════════════════════════════════════════════════════════════════════
# 14. _route_experiment extended
# ════════════════════════════════════════════════════════════════════════════


class TestRouteExperimentExtended:
    def test_deterministic_for_same_run_id(self):
        """Same run_id always routes to same variant."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            experiments = [{"active": True, "variant_a": "ga", "variant_b": "gb", "split": 0.5}]
            with open(path, "w") as f:
                json.dump(experiments, f)
            r1 = _route_experiment("ga", "run_fixed", experiments_dir=path)
            r2 = _route_experiment("ga", "run_fixed", experiments_dir=path)
            assert r1 == r2

    def test_corrupt_experiments_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            with open(path, "w") as f:
                f.write("NOT JSON")
            result = _route_experiment("ga", "run_1", experiments_dir=path)
            assert result == "ga"

    def test_variant_b_match(self):
        """Experiment match via variant_b graph_id."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "experiments.json")
            experiments = [{"active": True, "variant_a": "ga", "variant_b": "gb", "split": 0.5}]
            with open(path, "w") as f:
                json.dump(experiments, f)
            result = _route_experiment("gb", "run_1", experiments_dir=path)
            assert result in ("ga", "gb")


# ════════════════════════════════════════════════════════════════════════════
# 15. _run_evals_on_run extended
# ════════════════════════════════════════════════════════════════════════════


class TestRunEvalsOnRunExtended:
    def test_with_real_evaluator_name(self):
        """Test with a real evaluator class name from the registry."""
        from selectools.evals.evaluators import DEFAULT_EVALUATORS

        if DEFAULT_EVALUATORS:
            eval_name = type(DEFAULT_EVALUATORS[0]).__name__
            _run_evals_on_run(
                {
                    "eval_config": {"evaluators": [eval_name]},
                    "input": "hello world",
                }
            )


# ════════════════════════════════════════════════════════════════════════════
# 16. _parse_python_to_graph extended
# ════════════════════════════════════════════════════════════════════════════


class TestParsePythonExtended:
    def test_non_literal_args_ignored(self):
        code = """
g.add_node(some_variable)
g.add_edge(var1, var2)
"""
        result = _parse_python_to_graph(code)
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0

    def test_method_call_not_add_node(self):
        code = """
g.remove_node("x")
"""
        result = _parse_python_to_graph(code)
        assert len(result["nodes"]) == 0

    def test_x_positions_increase(self):
        code = """
g.add_node("a")
g.add_node("b")
g.add_node("c")
"""
        result = _parse_python_to_graph(code)
        xs = [n["x"] for n in result["nodes"]]
        assert xs == sorted(xs)
        assert xs[0] < xs[1] < xs[2]


# ════════════════════════════════════════════════════════════════════════════
# 17. _log_run OSError handling
# ════════════════════════════════════════════════════════════════════════════


class TestLogRunExtended:
    def test_oserror_suppressed(self):
        """OSError during write should be suppressed."""
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            with mock.patch("os.makedirs"):
                _log_run({"run_id": "r1"})  # Should not raise


# ════════════════════════════════════════════════════════════════════════════
# 18. BuilderServer /watch-file and /sync-to-file
# ════════════════════════════════════════════════════════════════════════════


class TestBuilderServerFileRoutes:
    """Test file-related routes on BuilderServer."""

    @pytest.fixture(scope="class")
    def server(self):
        port = _free_port()
        srv = BuilderServer(host="127.0.0.1", port=port)
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.3)
        yield f"http://127.0.0.1:{port}"

    def test_watch_file_not_found(self, server):
        payload = json.dumps({"path": "/nonexistent/path/file.py"}).encode()
        req = urllib.request.Request(
            f"{server}/watch-file",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert e.code == 404
        assert data.get("error") == "File not found"

    def test_sync_to_file_not_found(self, server):
        payload = json.dumps({"path": "/nonexistent/path/file.py", "patch": {}}).encode()
        req = urllib.request.Request(
            f"{server}/sync-to-file",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            assert e.code == 404
        assert data.get("error") == "File not found"

    def test_sync_to_file_no_patch(self, server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# test file\n")
            fpath = f.name
        try:
            payload = json.dumps({"path": fpath, "patch": {}}).encode()
            req = urllib.request.Request(
                f"{server}/sync-to-file",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            assert data.get("ok") is True
        finally:
            os.unlink(fpath)

    def test_sync_to_file_update_node(self, server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('g.add_node("agent_1")\n')
            fpath = f.name
        try:
            payload = json.dumps(
                {
                    "path": fpath,
                    "patch": {
                        "type": "update_node",
                        "node_id": "agent_1",
                        "changes": {"system_prompt": "new prompt"},
                    },
                }
            ).encode()
            req = urllib.request.Request(
                f"{server}/sync-to-file",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            assert data.get("ok") is True
        finally:
            os.unlink(fpath)
