"""Tests for the visual agent builder (v0.20.0)."""

import threading
import time
import urllib.request

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.stubs import LocalProvider
from selectools.serve.app import AgentRouter, AgentServer, BuilderServer, create_app
from selectools.serve.builder import BUILDER_HTML
from selectools.toolbox import get_all_tools


def _make_agent() -> Agent:
    tools = get_all_tools()[:1]
    return Agent(tools, provider=LocalProvider(), config=AgentConfig(name="test"))


class TestBuilderHtml:
    def test_builder_js_syntax(self):
        """JS block must have zero syntax errors (catches escaped newline bugs)."""
        import os
        import subprocess
        import tempfile

        script_start = BUILDER_HTML.find("<script>") + len("<script>")
        script_end = BUILDER_HTML.rfind("</script>")
        js = BUILDER_HTML[script_start:script_end]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js)
            js_file = f.name
        try:
            result = subprocess.run(["node", "--check", js_file], capture_output=True, text=True)
            assert result.returncode == 0, f"JS syntax error:\n{result.stderr}"
        finally:
            os.unlink(js_file)

    def test_builder_html_is_nonempty(self):
        assert len(BUILDER_HTML) > 1000

    def test_builder_html_has_required_elements(self):
        assert "selectools" in BUILDER_HTML
        assert "builder" in BUILDER_HTML
        assert "AgentGraph" in BUILDER_HTML
        assert "loadExample" in BUILDER_HTML
        assert "genPython" in BUILDER_HTML
        assert "genYaml" in BUILDER_HTML

    def test_builder_html_has_no_external_deps(self):
        # Must be fully self-contained — no CDN links
        assert "cdn." not in BUILDER_HTML
        assert "unpkg.com" not in BUILDER_HTML
        assert "jsdelivr" not in BUILDER_HTML
        assert "<script src=" not in BUILDER_HTML
        assert '<link rel="stylesheet" href="http' not in BUILDER_HTML


class TestAgentRouterBuilder:
    def _make_router(self, builder: bool = False) -> AgentRouter:
        return AgentRouter(_make_agent(), enable_builder=builder)

    def test_builder_disabled_by_default(self):
        router = self._make_router(builder=False)
        assert not router.enable_builder

    def test_builder_enabled(self):
        router = self._make_router(builder=True)
        assert router.enable_builder


class TestBuilderEndpoint:
    """Integration test: BuilderServer endpoint returns the HTML."""

    @pytest.fixture()
    def server(self):
        srv = BuilderServer(host="127.0.0.1", port=18766)
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.15)
        yield "http://127.0.0.1:18766"

    def test_builder_route_returns_html(self, server):
        resp = urllib.request.urlopen(f"{server}/builder")
        assert resp.status == 200
        body = resp.read().decode()
        assert "selectools" in body
        assert "AgentGraph" in body

    def test_builder_content_type(self, server):
        resp = urllib.request.urlopen(f"{server}/builder")
        ct = resp.headers.get("Content-Type", "")
        assert "text/html" in ct

    def test_builder_health_endpoint(self, server):
        import json

        resp = urllib.request.urlopen(f"{server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["mode"] == "builder"


class TestCreateAppBuilder:
    def test_create_app_accepts_builder_flag(self):
        app = create_app(_make_agent(), playground=False, builder=True)
        assert app.router.enable_builder
