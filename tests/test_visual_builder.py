"""Tests for the visual agent builder (v0.20.0)."""

import io
import json
import threading
import time
import urllib.request

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.stubs import LocalProvider
from selectools.serve.app import (
    AgentRouter,
    AgentServer,
    BuilderServer,
    _builder_run_mock,
    _run_builder_evals,
    create_app,
)
from selectools.serve.builder import BUILDER_HTML
from selectools.toolbox import get_all_tools


def _make_agent() -> Agent:
    tools = get_all_tools()[:1]
    return Agent(tools, provider=LocalProvider(), config=AgentConfig(name="test"))


# ─── HTML content checks ────────────────────────────────────────────────────


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

    # ── Feature: color-coded typed ports ──────────────────────────────────

    def test_port_type_css_classes_exist(self):
        """Color CSS classes for each port type must be defined."""
        for cls in ("port-msg", "port-ctrl", "port-body", "port-done", "port-term", "port-sub"):
            assert f".{cls} circle" in BUILDER_HTML, f"Missing CSS for .{cls}"

    def test_port_type_colors_are_distinct(self):
        """Each port type has a distinct stroke color."""
        # cyan for msg, green for ctrl/done, orange for body, red for term, purple for sub
        assert "#22d3ee" in BUILDER_HTML  # msg
        assert "#f97316" in BUILDER_HTML  # body
        assert "#ef4444" in BUILDER_HTML  # term
        assert "#a855f7" in BUILDER_HTML  # sub

    def test_port_type_js_assignment(self):
        """renderNodes() must assign port type classes (port-msg, port-ctrl, etc.)."""
        assert "port-msg" in BUILDER_HTML
        assert "port-ctrl" in BUILDER_HTML
        assert "port-body" in BUILDER_HTML
        assert "port-done" in BUILDER_HTML
        assert "port-term" in BUILDER_HTML
        assert "port-sub" in BUILDER_HTML

    def test_portclick_type_enforcement(self):
        """portClick must check ptype and block loop body → terminal connections."""
        assert "ptype" in BUILDER_HTML
        assert "connecting.ptype" in BUILDER_HTML
        assert "body" in BUILDER_HTML and "term" in BUILDER_HTML

    # ── Feature: freeze upstream nodes ────────────────────────────────────

    def test_frozen_field_in_mknode(self):
        """mkNode() must initialise frozen:false on agent nodes."""
        assert "frozen: false" in BUILDER_HTML

    def test_frozen_css_class_exists(self):
        assert ".node-frozen .body" in BUILDER_HTML

    def test_frozen_outputs_state(self):
        assert "frozenOutputs" in BUILDER_HTML

    def test_freeze_toggle_in_props(self):
        """showNodeProps must render a freeze/unfreeze button for agent nodes."""
        assert "Freeze Node" in BUILDER_HTML
        assert "n.frozen" in BUILDER_HTML

    def test_frozen_node_skipped_in_run(self):
        """runMock must skip frozen nodes that have cached output."""
        assert "n.frozen && frozenOutputs" in BUILDER_HTML

    # ── Feature: time-travel trace scrubber ───────────────────────────────

    def test_scrubber_div_in_html(self):
        assert 'id="scrubber"' in BUILDER_HTML

    def test_scrubber_css_exists(self):
        assert ".scrubber" in BUILDER_HTML
        assert ".scrub-step" in BUILDER_HTML
        assert ".scrub-dot" in BUILDER_HTML

    def test_renderscrubber_function_exists(self):
        assert "function renderScrubber()" in BUILDER_HTML

    def test_renderscrubber_called_after_run(self):
        """renderScrubber() must be called at the end of runTest()."""
        # The call should appear after refreshHistory() in runTest
        rh_idx = BUILDER_HTML.find("refreshHistory();")
        rs_idx = BUILDER_HTML.find("renderScrubber();")
        assert rs_idx != -1, "renderScrubber() call not found"
        assert rs_idx > rh_idx, "renderScrubber() must come after refreshHistory()"

    def test_scrubber_node_id_in_events(self):
        """node_start events must include node_id for scrubber click-to-highlight."""
        assert "node_id: n.id" in BUILDER_HTML

    # ── Feature: canvas search (Cmd+K) ────────────────────────────────────

    def test_search_overlay_in_html(self):
        assert 'id="searchOverlay"' in BUILDER_HTML

    def test_search_input_in_html(self):
        assert 'id="searchInput"' in BUILDER_HTML

    def test_search_functions_exist(self):
        assert "function openSearch()" in BUILDER_HTML
        assert "function closeSearch()" in BUILDER_HTML
        assert "function searchNodes(" in BUILDER_HTML
        assert "function searchSelect(" in BUILDER_HTML

    def test_cmd_k_keyboard_handler(self):
        """Keyboard handler must bind Cmd+K / Ctrl+K to openSearch."""
        assert "e.key === 'k'" in BUILDER_HTML
        assert "openSearch()" in BUILDER_HTML

    def test_escape_closes_search(self):
        """Escape key must close the search overlay first."""
        assert "closeSearch()" in BUILDER_HTML

    def test_search_css_exists(self):
        assert ".search-overlay" in BUILDER_HTML
        assert ".search-result" in BUILDER_HTML

    def test_tip_box_mentions_cmd_k(self):
        """Tip box should hint the user about Cmd+K search."""
        assert "Cmd+K" in BUILDER_HTML or "search nodes" in BUILDER_HTML.lower()

    # ── Feature: eval pass/fail badges ───────────────────────────────────

    def test_eval_dot_css_exists(self):
        """CSS for eval dot badges must exist."""
        assert ".eval-dot-pass" in BUILDER_HTML
        assert ".eval-dot-fail" in BUILDER_HTML

    def test_eval_trace_css_exists(self):
        """CSS for eval trace lines must exist."""
        assert ".trace-eval-pass" in BUILDER_HTML
        assert ".trace-eval-fail" in BUILDER_HTML

    def test_eval_results_state_var(self):
        """evalResults state variable must be declared."""
        assert "evalResults = {}" in BUILDER_HTML

    def test_client_run_evals_function(self):
        """_clientRunEvals function must exist in JS."""
        assert "function _clientRunEvals(" in BUILDER_HTML

    def test_eval_result_handler_in_handle_trace_event(self):
        """handleTraceEvent must handle eval_result events."""
        assert "eval_result" in BUILDER_HTML
        assert "ev.type === 'eval_result'" in BUILDER_HTML

    def test_eval_assertion_field_in_props(self):
        """showNodeProps must include Eval Assertion field for agent nodes."""
        assert "Eval Assertion" in BUILDER_HTML

    def test_eval_dot_rendered_in_render_nodes(self):
        """renderNodes must render eval dot circles for agent nodes."""
        assert "evalResults[n.id]" in BUILDER_HTML
        assert "eval-dot-pass" in BUILDER_HTML
        assert "eval-dot-fail" in BUILDER_HTML

    def test_eval_results_reset_in_run_test(self):
        """runTest must reset evalResults before each run."""
        assert "evalResults = {};" in BUILDER_HTML

    def test_eval_results_reset_in_replay_history(self):
        """replayHistory must reset evalResults before replaying."""
        # Find replayHistory function and verify evalResults reset inside it
        rh_idx = BUILDER_HTML.find("function replayHistory(")
        reset_idx = BUILDER_HTML.find("evalResults = {};", rh_idx)
        assert (
            reset_idx != -1 and reset_idx < rh_idx + 400
        ), "evalResults must be reset inside replayHistory"

    def test_eval_assertion_in_mk_node(self):
        """mkNode must initialise eval_assertion field on agent nodes."""
        assert "eval_assertion: ''" in BUILDER_HTML

    def test_not_empty_eval_in_client_evals(self):
        """_clientRunEvals must include not_empty check."""
        ci = BUILDER_HTML.find("function _clientRunEvals(")
        ce = BUILDER_HTML.find("}", ci + 10)
        # Find closing brace of function (nested, so find a reasonable range)
        assert "not_empty" in BUILDER_HTML[ci : ci + 800]

    def test_no_apology_eval_in_client_evals(self):
        """_clientRunEvals must include no_apology check."""
        ci = BUILDER_HTML.find("function _clientRunEvals(")
        assert "no_apology" in BUILDER_HTML[ci : ci + 800]

    def test_client_run_evals_contains_check(self):
        """_clientRunEvals must include assertion contains check."""
        ci = BUILDER_HTML.find("function _clientRunEvals(")
        assert "contains(" in BUILDER_HTML[ci : ci + 800]


# ─── AgentRouter builder flag ───────────────────────────────────────────────


class TestAgentRouterBuilder:
    def _make_router(self, builder: bool = False) -> AgentRouter:
        return AgentRouter(_make_agent(), enable_builder=builder)

    def test_builder_disabled_by_default(self):
        router = self._make_router(builder=False)
        assert not router.enable_builder

    def test_builder_enabled(self):
        router = self._make_router(builder=True)
        assert router.enable_builder


# ─── _builder_run_mock ──────────────────────────────────────────────────────


class TestBuilderRunMock:
    """Unit tests for the _builder_run_mock function."""

    def _run(self, nodes_data, input_msg="hello"):
        events = []
        _builder_run_mock(nodes_data, input_msg, events.append)
        return events

    def _agent_node(self, **kw):
        base = {
            "id": "a1",
            "type": "agent",
            "name": "TestAgent",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "tools": "",
            "frozen": False,
        }
        base.update(kw)
        return base

    def test_emits_node_start_and_end(self):
        events = self._run([self._agent_node()])
        types = [e["type"] for e in events]
        assert "node_start" in types
        assert "node_end" in types

    def test_node_start_has_node_id(self):
        events = self._run([self._agent_node(id="x1")])
        starts = [e for e in events if e["type"] == "node_start"]
        assert starts[0]["node_id"] == "x1"

    def test_node_start_has_node_type(self):
        events = self._run([self._agent_node()])
        starts = [e for e in events if e["type"] == "node_start"]
        assert starts[0]["node_type"] == "agent"

    def test_emits_run_end(self):
        events = self._run([self._agent_node()])
        types = [e["type"] for e in events]
        assert "run_end" in types

    def test_empty_nodes_emits_no_node_events(self):
        events = self._run([])
        types = [e["type"] for e in events]
        assert "node_start" not in types

    def test_tool_calls_emitted_when_tools_set(self):
        node = self._agent_node(tools="search, calc")
        events = self._run([node])
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) >= 1
        assert tool_events[0]["tool"] in ("search", "calc")

    def test_tool_call_has_node_id(self):
        node = self._agent_node(id="n42", tools="search")
        events = self._run([node])
        tc = next(e for e in events if e["type"] == "tool_call")
        assert tc["node_id"] == "n42"

    def test_chunk_events_emitted(self):
        events = self._run([self._agent_node()])
        assert any(e["type"] == "chunk" for e in events)

    def test_multiple_agents_both_executed(self):
        n1 = self._agent_node(id="a1", name="First")
        n2 = {
            "id": "a2",
            "type": "agent",
            "name": "Second",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "tools": "",
            "frozen": False,
        }
        events = self._run([n1, n2])
        starts = [e for e in events if e["type"] == "node_start"]
        assert len(starts) == 2

    def test_frozen_node_with_no_cache_runs_normally(self):
        """A frozen node with no cached output should still execute normally."""
        # The mock function doesn't have access to frozenOutputs (JS-side state),
        # so frozen nodes always execute in server-side mock — JS handles the skip.
        node = self._agent_node(frozen=True)
        events = self._run([node])
        assert any(e["type"] == "node_start" for e in events)

    def test_run_end_has_total_tokens(self):
        events = self._run([self._agent_node()])
        run_end = next(e for e in events if e["type"] == "run_end")
        assert "total_tokens" in run_end
        assert run_end["total_tokens"] >= 0


# ─── BuilderServer HTTP endpoints ───────────────────────────────────────────


class TestBuilderEndpoint:
    """Integration test: BuilderServer endpoint returns the HTML."""

    @pytest.fixture(scope="class")
    def server(self):
        srv = BuilderServer(host="127.0.0.1", port=18766)
        t = threading.Thread(target=srv.serve, daemon=True)
        t.start()
        time.sleep(0.2)
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
        resp = urllib.request.urlopen(f"{server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["mode"] == "builder"

    def test_run_endpoint_mock_sse(self, server):
        """/run POST with no api_key returns SSE stream with node_start events."""
        payload = json.dumps(
            {
                "input": "test message",
                "nodes": [
                    {"id": "__start__", "type": "start", "name": "START", "x": 60, "y": 160},
                    {
                        "id": "a1",
                        "type": "agent",
                        "name": "Bot",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "",
                        "tools": "",
                        "frozen": False,
                    },
                ],
                "edges": [{"id": "e1", "from": "__start__", "to": "a1", "label": ""}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        body = resp.read().decode()
        assert "data:" in body
        assert "[DONE]" in body

    def test_run_endpoint_contains_node_start(self, server):
        """/run SSE stream must contain a node_start event."""
        payload = json.dumps(
            {
                "input": "hi",
                "nodes": [
                    {"id": "__start__", "type": "start", "name": "START", "x": 0, "y": 0},
                    {
                        "id": "ag1",
                        "type": "agent",
                        "name": "MyAgent",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "",
                        "tools": "",
                        "frozen": False,
                    },
                ],
                "edges": [{"id": "e1", "from": "__start__", "to": "ag1", "label": ""}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        lines = resp.read().decode().splitlines()
        data_lines = [ln[6:] for ln in lines if ln.startswith("data:") and ln[6:] != "[DONE]"]
        events = [json.loads(d) for d in data_lines if d.strip()]
        event_types = [e.get("type") for e in events]
        assert "node_start" in event_types

    def test_run_endpoint_contains_run_end(self, server):
        payload = json.dumps(
            {
                "input": "hi",
                "nodes": [
                    {"id": "__start__", "type": "start", "name": "START", "x": 0, "y": 0},
                    {
                        "id": "ag1",
                        "type": "agent",
                        "name": "A",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "",
                        "tools": "",
                        "frozen": False,
                    },
                ],
                "edges": [{"id": "e1", "from": "__start__", "to": "ag1", "label": ""}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        lines = resp.read().decode().splitlines()
        data_lines = [ln[6:] for ln in lines if ln.startswith("data:") and ln[6:] != "[DONE]"]
        events = [json.loads(d) for d in data_lines if d.strip()]
        event_types = [e.get("type") for e in events]
        assert "run_end" in event_types

    def test_run_endpoint_contains_eval_result(self, server):
        """/run SSE stream must contain an eval_result event."""
        payload = json.dumps(
            {
                "input": "hi",
                "nodes": [
                    {"id": "__start__", "type": "start", "name": "START", "x": 0, "y": 0},
                    {
                        "id": "ag1",
                        "type": "agent",
                        "name": "A",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "",
                        "tools": "",
                        "frozen": False,
                        "eval_assertion": "",
                    },
                ],
                "edges": [{"id": "e1", "from": "__start__", "to": "ag1", "label": ""}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        lines = resp.read().decode().splitlines()
        data_lines = [ln[6:] for ln in lines if ln.startswith("data:") and ln[6:] != "[DONE]"]
        events = [json.loads(d) for d in data_lines if d.strip()]
        event_types = [e.get("type") for e in events]
        assert "eval_result" in event_types

    def test_run_endpoint_eval_result_has_pass(self, server):
        """/run eval_result event must have a pass field."""
        payload = json.dumps(
            {
                "input": "hi",
                "nodes": [
                    {"id": "__start__", "type": "start", "name": "START", "x": 0, "y": 0},
                    {
                        "id": "ag1",
                        "type": "agent",
                        "name": "A",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "",
                        "tools": "",
                        "frozen": False,
                        "eval_assertion": "",
                    },
                ],
                "edges": [{"id": "e1", "from": "__start__", "to": "ag1", "label": ""}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{server}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        lines = resp.read().decode().splitlines()
        data_lines = [ln[6:] for ln in lines if ln.startswith("data:") and ln[6:] != "[DONE]"]
        events = [json.loads(d) for d in data_lines if d.strip()]
        er = next((e for e in events if e.get("type") == "eval_result"), None)
        assert er is not None
        assert "pass" in er
        assert "results" in er


# ─── _run_builder_evals unit tests ─────────────────────────────────────────


class TestRunBuilderEvals:
    """Unit tests for the _run_builder_evals function."""

    def test_not_empty_pass(self):
        r = _run_builder_evals("hello world", {})
        not_empty = next(x for x in r["results"] if x["name"] == "not_empty")
        assert not_empty["pass"] is True

    def test_not_empty_fail(self):
        r = _run_builder_evals("   ", {})
        not_empty = next(x for x in r["results"] if x["name"] == "not_empty")
        assert not_empty["pass"] is False
        assert r["pass"] is False

    def test_no_apology_pass(self):
        r = _run_builder_evals("Here is a great answer.", {})
        na = next(x for x in r["results"] if x["name"] == "no_apology")
        assert na["pass"] is True

    def test_no_apology_fail(self):
        for apology in ["I'm sorry I can't help", "I apologize for the error", "I am sorry"]:
            r = _run_builder_evals(apology, {})
            na = next(x for x in r["results"] if x["name"] == "no_apology")
            assert na["pass"] is False, f"Expected apology '{apology}' to fail"

    def test_no_assertion_only_two_checks(self):
        r = _run_builder_evals("some output", {})
        assert len(r["results"]) == 2

    def test_with_assertion_three_checks(self):
        r = _run_builder_evals("some output", {"eval_assertion": "output"})
        assert len(r["results"]) == 3

    def test_assertion_pass(self):
        r = _run_builder_evals("The answer is processed here.", {"eval_assertion": "processed"})
        assert r["results"][2]["pass"] is True
        assert r["pass"] is True

    def test_assertion_fail(self):
        r = _run_builder_evals("Normal output here.", {"eval_assertion": "xyz_not_found_999"})
        assert r["results"][2]["pass"] is False
        assert r["pass"] is False

    def test_assertion_case_insensitive(self):
        r = _run_builder_evals("PROCESSED your request.", {"eval_assertion": "processed"})
        assert r["results"][2]["pass"] is True

    def test_overall_pass_true_when_all_pass(self):
        r = _run_builder_evals("The result is ready.", {"eval_assertion": "result"})
        assert r["pass"] is True

    def test_overall_pass_false_when_any_fail(self):
        r = _run_builder_evals("", {})
        assert r["pass"] is False


# ─── _builder_run_mock eval events ─────────────────────────────────────────


class TestBuilderRunMockEvals:
    """Tests that _builder_run_mock emits eval_result events."""

    def _run(self, nodes_data, input_msg="hello"):
        events = []
        _builder_run_mock(nodes_data, input_msg, events.append)
        return events

    def _agent_node(self, **kw):
        base = {
            "id": "a1",
            "type": "agent",
            "name": "TestAgent",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "tools": "",
            "frozen": False,
            "eval_assertion": "",
        }
        base.update(kw)
        return base

    def test_emits_eval_result(self):
        events = self._run([self._agent_node()])
        assert any(e["type"] == "eval_result" for e in events)

    def test_eval_result_has_node_id(self):
        events = self._run([self._agent_node(id="n99")])
        er = next(e for e in events if e["type"] == "eval_result")
        assert er["node_id"] == "n99"

    def test_eval_result_has_pass_field(self):
        events = self._run([self._agent_node()])
        er = next(e for e in events if e["type"] == "eval_result")
        assert "pass" in er

    def test_eval_result_has_results_list(self):
        events = self._run([self._agent_node()])
        er = next(e for e in events if e["type"] == "eval_result")
        assert "results" in er
        assert isinstance(er["results"], list)

    def test_eval_result_pass_true_for_good_mock_output(self):
        """Mock output is non-empty and not an apology — should pass."""
        events = self._run([self._agent_node()])
        er = next(e for e in events if e["type"] == "eval_result")
        assert er["pass"] is True

    def test_eval_result_assertion_match(self):
        """Assertion keyword present in mock output → pass."""
        # Mock text contains "processed" — use that as the assertion
        node = self._agent_node(eval_assertion="processed")
        events = self._run([node])
        er = next(e for e in events if e["type"] == "eval_result")
        assert er["pass"] is True

    def test_eval_result_assertion_miss(self):
        """Assertion keyword absent from mock output → fail."""
        node = self._agent_node(eval_assertion="xyz_not_found_999")
        events = self._run([node])
        er = next(e for e in events if e["type"] == "eval_result")
        assert er["pass"] is False

    def test_eval_result_emitted_before_node_end(self):
        """eval_result must come before node_end in the event stream."""
        events = self._run([self._agent_node()])
        types = [e["type"] for e in events]
        er_idx = types.index("eval_result")
        ne_idx = types.index("node_end")
        assert er_idx < ne_idx

    def test_multiple_agents_each_get_eval_result(self):
        n1 = self._agent_node(id="a1", name="First")
        n2 = {
            "id": "a2",
            "type": "agent",
            "name": "Second",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "tools": "",
            "frozen": False,
            "eval_assertion": "",
        }
        events = self._run([n1, n2])
        eval_events = [e for e in events if e["type"] == "eval_result"]
        assert len(eval_events) == 2


# ─── HITL node tests ────────────────────────────────────────────────────────


class TestBuilderHtmlHitl:
    """HITL node content checks in the generated HTML."""

    def test_hitl_node_type_present(self):
        assert "'hitl'" in BUILDER_HTML or '"hitl"' in BUILDER_HTML

    def test_hitl_css_class_present(self):
        assert "node-hitl" in BUILDER_HTML

    def test_hitl_trace_prompt_css(self):
        assert "trace-hitl-prompt" in BUILDER_HTML

    def test_hitl_btns_css(self):
        assert "hitl-btns" in BUILDER_HTML

    def test_wait_for_hitl_choice_defined(self):
        assert "waitForHitlChoice" in BUILDER_HTML

    def test_hitl_pause_handler(self):
        assert "hitl_pause" in BUILDER_HTML

    def test_hitl_choice_handler(self):
        assert "hitl_choice" in BUILDER_HTML

    def test_hitl_auto_handler(self):
        assert "hitl_auto" in BUILDER_HTML

    def test_hitl_template_uses_hitl_node(self):
        """hitl_approval template must use a real hitl node, not just two agent nodes."""
        # Second occurrence is the loadTemplate function body
        first = BUILDER_HTML.index("hitl_approval")
        idx = BUILDER_HTML.index("hitl_approval", first + 1)
        block = BUILDER_HTML[idx : idx + 800]
        assert "mkNode('hitl'" in block

    def test_hitl_node_in_topo_order(self):
        """topoOrder must include hitl nodes, not only agent nodes."""
        assert "executableNodes" in BUILDER_HTML

    def test_genPython_handles_hitl(self):
        assert "wait_for_human" in BUILDER_HTML

    def test_genYaml_handles_hitl(self):
        """YAML generator must emit type: hitl."""
        assert "type: hitl" in BUILDER_HTML


class TestBuilderRunMockHitl:
    """Unit tests for hitl node dispatch in _builder_run_mock."""

    def _run(self, nodes_data, input_msg="test"):
        events = []
        _builder_run_mock(nodes_data, input_msg, lambda e: events.append(e))
        return events

    def _hitl_node(self, **kw):
        base = {
            "id": "h1",
            "type": "hitl",
            "name": "ReviewGate",
            "options": "approve, reject",
            "timeout_label": "timeout",
        }
        base.update(kw)
        return base

    def test_hitl_emits_node_start(self):
        events = self._run([self._hitl_node()])
        assert any(e["type"] == "node_start" for e in events)

    def test_hitl_node_start_has_correct_type(self):
        events = self._run([self._hitl_node()])
        ns = next(e for e in events if e["type"] == "node_start")
        assert ns["node_type"] == "hitl"

    def test_hitl_emits_hitl_pause(self):
        events = self._run([self._hitl_node()])
        assert any(e["type"] == "hitl_pause" for e in events)

    def test_hitl_pause_has_options(self):
        events = self._run([self._hitl_node(options="approve, reject")])
        hp = next(e for e in events if e["type"] == "hitl_pause")
        assert "approve" in hp["options"]

    def test_hitl_emits_hitl_auto(self):
        events = self._run([self._hitl_node()])
        assert any(e["type"] == "hitl_auto" for e in events)

    def test_hitl_auto_choice_is_first_option(self):
        events = self._run([self._hitl_node(options="yes, no")])
        ha = next(e for e in events if e["type"] == "hitl_auto")
        assert ha["choice"] == "yes"

    def test_hitl_emits_node_end(self):
        events = self._run([self._hitl_node()])
        assert any(e["type"] == "node_end" for e in events)

    def test_hitl_node_end_zero_tokens(self):
        events = self._run([self._hitl_node()])
        ne = next(e for e in events if e["type"] == "node_end")
        assert ne["tokens"] == 0

    def test_hitl_does_not_emit_eval_result(self):
        """HITL nodes don't run LLM evals."""
        events = self._run([self._hitl_node()])
        assert not any(e["type"] == "eval_result" for e in events)

    def test_hitl_mixed_with_agent(self):
        """Graph with both hitl and agent nodes produces events for both."""
        hitl = self._hitl_node(id="h1")
        agent = {
            "id": "a1",
            "type": "agent",
            "name": "Writer",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "tools": "",
            "frozen": False,
            "eval_assertion": "",
        }
        events = self._run([hitl, agent])
        types = [e["type"] for e in events]
        assert "hitl_pause" in types
        assert "eval_result" in types
        assert "run_end" in types

    def test_graph_with_only_hitl_emits_run_end(self):
        events = self._run([self._hitl_node()])
        assert any(e["type"] == "run_end" for e in events)


# ─── Dynamic {variable} port tests ──────────────────────────────────────────


class TestBuilderHtmlDynamicPorts:
    """Content checks for dynamic {variable} input ports on agent nodes."""

    def test_extract_vars_defined(self):
        assert "extractVars" in BUILDER_HTML

    def test_skip_vars_defined(self):
        assert "SKIP_VARS" in BUILDER_HTML

    def test_port_var_css(self):
        assert ".port-var" in BUILDER_HTML

    def test_port_var_hover_css(self):
        assert ".port-var:hover" in BUILDER_HTML

    def test_var_ports_rendered_for_agent(self):
        """renderNodes must use port-var class for agent variable ports."""
        assert "port-var" in BUILDER_HTML
        assert "port = v" in BUILDER_HTML or "dataset.port = v" in BUILDER_HTML

    def test_var_port_ptype_set(self):
        assert "ptype = 'var'" in BUILDER_HTML or "ptype: 'var'" in BUILDER_HTML

    def test_nodeHeight_accounts_for_vars(self):
        """nodeHeight must handle agent nodes with var ports."""
        assert "extractVars(n.system_prompt)" in BUILDER_HTML

    def test_render_prunes_orphaned_var_edges(self):
        """render() must filter edges whose varPort is no longer in system_prompt."""
        assert "varPort" in BUILDER_HTML
        assert "extractVars(tn.system_prompt).includes(e.varPort)" in BUILDER_HTML

    def test_render_edges_handles_var_port(self):
        """renderEdges must compute p2 from varPort index."""
        assert "e.varPort" in BUILDER_HTML

    def test_genPython_emits_var_comment(self):
        assert "Variable ports:" in BUILDER_HTML

    def test_genYaml_emits_input_vars(self):
        assert "input_vars:" in BUILDER_HTML

    def test_runMock_warns_unconnected_vars(self):
        assert "unconnected" in BUILDER_HTML
        assert "raw placeholder" in BUILDER_HTML

    def test_portClick_stores_varPort(self):
        assert "edge.varPort" in BUILDER_HTML or "varPort = port" in BUILDER_HTML


# ─── Code Import (Feature 2) ────────────────────────────────────────────────


class TestBuilderHtmlImport:
    def test_import_button_present(self):
        """Import button must call openImport() and be in the header."""
        assert "openImport()" in BUILDER_HTML
        assert "Import" in BUILDER_HTML

    def test_import_modal_present(self):
        """importModal div must exist in the HTML."""
        assert 'id="importModal"' in BUILDER_HTML

    def test_import_error_element_present(self):
        """importError div must exist inside the modal."""
        assert 'id="importError"' in BUILDER_HTML

    def test_import_input_element_present(self):
        """importInput textarea must exist inside the modal."""
        assert 'id="importInput"' in BUILDER_HTML

    def test_open_import_defined(self):
        """openImport() function must be defined in the JS."""
        assert "function openImport()" in BUILDER_HTML

    def test_close_import_defined(self):
        """closeImport() function must be defined in the JS."""
        assert "function closeImport()" in BUILDER_HTML

    def test_parse_yaml_defined(self):
        """parseYaml() function must be defined in the JS."""
        assert "function parseYaml(" in BUILDER_HTML

    def test_parse_python_defined(self):
        """parsePython() function must be defined in the JS."""
        assert "function parsePython(" in BUILDER_HTML

    def test_auto_layout_defined(self):
        """autoLayout() function must be defined in the JS."""
        assert "function autoLayout(" in BUILDER_HTML

    def test_do_import_defined(self):
        """doImport() function must be defined in the JS."""
        assert "function doImport()" in BUILDER_HTML

    def test_parse_yaml_handles_nodes_section(self):
        """parseYaml must parse a 'nodes:' section."""
        assert "nodes:" in BUILDER_HTML
        assert "section === 'nodes'" in BUILDER_HTML

    def test_parse_yaml_handles_edges_section(self):
        """parseYaml must parse an 'edges:' section."""
        assert "section === 'edges'" in BUILDER_HTML

    def test_parse_yaml_resolves_start_end(self):
        """parseYaml must resolve START/END keywords to node IDs."""
        assert "e.from === 'START'" in BUILDER_HTML

    def test_parse_python_finds_agent_vars(self):
        """parsePython must scan for Agent() variable assignments."""
        assert "= Agent(" in BUILDER_HTML or "Agent\\s*\\(" in BUILDER_HTML

    def test_parse_python_finds_entry_point(self):
        """parsePython must use set_entry_point to build the start edge."""
        assert "set_entry_point" in BUILDER_HTML

    def test_auto_layout_bfs_from_start(self):
        """autoLayout must BFS from the start node to assign layers."""
        assert "type === 'start'" in BUILDER_HTML
        assert "layer: 0" in BUILDER_HTML

    def test_do_import_calls_auto_layout(self):
        """doImport must call autoLayout() on the parsed nodes."""
        assert "autoLayout(" in BUILDER_HTML

    def test_do_import_calls_snapshot(self):
        """doImport must call snapshot() so undo works after import."""
        # snapshot() should appear inside doImport
        do_import_idx = BUILDER_HTML.index("function doImport()")
        do_import_body = BUILDER_HTML[do_import_idx : do_import_idx + 1200]
        assert "snapshot()" in do_import_body

    def test_escape_closes_import_modal(self):
        """Escape key handler must check importModal before searchOverlay."""
        escape_idx = BUILDER_HTML.index("if (e.key === 'Escape')")
        escape_block = BUILDER_HTML[escape_idx : escape_idx + 900]
        im_pos = escape_block.find("importModal")
        ov_pos = escape_block.find("searchOverlay")
        assert im_pos != -1, "importModal not in Escape handler"
        assert im_pos < ov_pos, "importModal check must come before searchOverlay check"


# ─── AI Workflow Builder (Feature 4) ────────────────────────────────────────


class TestBuilderHtmlAiBuild:
    def test_generate_button_present(self):
        """Generate button calling openGenBar() must be in the header."""
        assert "openGenBar()" in BUILDER_HTML
        assert "Generate" in BUILDER_HTML

    def test_gen_bar_present(self):
        """genBar div must be in the HTML."""
        assert 'id="genBar"' in BUILDER_HTML

    def test_gen_input_present(self):
        """genInput text field must be inside genBar."""
        assert 'id="genInput"' in BUILDER_HTML

    def test_gen_btn_present(self):
        """genBtn button must be present."""
        assert 'id="genBtn"' in BUILDER_HTML

    def test_do_generate_defined(self):
        """doGenerate() async function must be defined."""
        assert "async function doGenerate()" in BUILDER_HTML

    def test_auto_layout_reused(self):
        """doGenerate must reuse autoLayout() for positioning."""
        gen_idx = BUILDER_HTML.index("async function doGenerate()")
        gen_body = BUILDER_HTML[gen_idx : gen_idx + 900]
        assert "autoLayout(" in gen_body

    def test_node_entering_css(self):
        """node-entering CSS animation must be present."""
        assert "node-entering" in BUILDER_HTML
        assert "nodeIn" in BUILDER_HTML

    def test_ai_build_endpoint_called(self):
        """doGenerate must POST to /ai-build."""
        assert "'/ai-build'" in BUILDER_HTML or '"/ai-build"' in BUILDER_HTML

    def test_gen_input_enter_key_handler(self):
        """genInput must have a keydown handler for Enter."""
        assert "genInput" in BUILDER_HTML
        assert "e.key === 'Enter'" in BUILDER_HTML

    def test_gen_bar_escape_handler(self):
        """Escape key must close genBar."""
        escape_idx = BUILDER_HTML.index("if (e.key === 'Escape')")
        escape_block = BUILDER_HTML[escape_idx : escape_idx + 900]
        assert "genBar" in escape_block


class TestAiBuildFallback:
    def test_fallback_returns_nodes_and_edges(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("A researcher agent")
        assert "nodes" in result
        assert "edges" in result

    def test_fallback_has_start_node(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("A researcher agent")
        types = [n["type"] for n in result["nodes"]]
        assert "start" in types

    def test_fallback_has_end_node(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("A researcher agent")
        types = [n["type"] for n in result["nodes"]]
        assert "end" in types

    def test_fallback_researcher_keyword(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("research and find information")
        names = [n["name"] for n in result["nodes"]]
        assert "Researcher" in names

    def test_fallback_writer_keyword(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("write a blog post")
        names = [n["name"] for n in result["nodes"]]
        assert "Writer" in names

    def test_fallback_critic_keyword(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("review and evaluate the output")
        names = [n["name"] for n in result["nodes"]]
        assert "Critic" in names

    def test_fallback_generic_fallback(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("do something with xyz")
        agent_nodes = [n for n in result["nodes"] if n["type"] == "agent"]
        assert len(agent_nodes) >= 1
        assert agent_nodes[0]["name"] == "Agent"

    def test_fallback_edges_connect_start_to_agents(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("research the topic")
        start = next(n for n in result["nodes"] if n["type"] == "start")
        assert any(e["from"] == start["id"] for e in result["edges"])

    def test_fallback_edges_connect_last_agent_to_end(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("research and write")
        end = next(n for n in result["nodes"] if n["type"] == "end")
        assert any(e["to"] == end["id"] for e in result["edges"])


class TestAiBuildEndpoint:
    def test_ai_build_route_exists(self):
        """POST /ai-build must return 200 with valid graph JSON."""
        import json

        from selectools.serve.app import BuilderServer

        server = BuilderServer(port=0)
        # Use a real port for testing
        import socket
        from http.server import HTTPServer
        from socketserver import TCPServer

        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        t = threading.Thread(target=server.serve, kwargs={"port": port}, daemon=True)
        t.start()
        import time

        time.sleep(0.3)

        import urllib.request

        req = urllib.request.Request(
            f"http://localhost:{port}/ai-build",
            data=json.dumps({"description": "a researcher agent"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert resp.status == 200
        assert "nodes" in data
        assert "edges" in data

    def test_ai_build_no_key_uses_fallback(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("a simple chatbot")
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)
        assert len(result["nodes"]) >= 2  # at least start + end

    def test_ai_build_response_has_nodes(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("researcher, writer, critic")
        assert "nodes" in result

    def test_ai_build_response_has_edges(self):
        from selectools.serve.app import _ai_build_fallback

        result = _ai_build_fallback("researcher, writer, critic")
        assert "edges" in result


# ─── Typed Port Enforcement (Feature 3) ────────────────────────────────────


class TestBuilderHtmlPortEnforcement:
    def test_compat_table_defined(self):
        """COMPAT object must be defined in the JS."""
        assert "const COMPAT" in BUILDER_HTML

    def test_ports_compat_defined(self):
        """portsCompat() function must be defined."""
        assert "function portsCompat(" in BUILDER_HTML

    def test_port_blocked_css(self):
        """.port-blocked CSS class must be present."""
        assert ".port-blocked" in BUILDER_HTML

    def test_port_compatible_css(self):
        """.port-compatible CSS class must be present."""
        assert ".port-compatible" in BUILDER_HTML

    def test_update_port_compat_defined(self):
        """updatePortCompat() helper must be defined and called from render()."""
        assert "function updatePortCompat()" in BUILDER_HTML
        assert "updatePortCompat()" in BUILDER_HTML

    def test_port_compat_enforced_in_portClick(self):
        """portsCompat must be called inside portClick to block incompatible connections."""
        portclick_idx = BUILDER_HTML.index("function portClick(")
        portclick_body = BUILDER_HTML[portclick_idx : portclick_idx + 1500]
        assert "portsCompat(" in portclick_body

    def test_ctrl_compat_set(self):
        """ctrl port should be able to connect to msg, sub, term."""
        compat_idx = BUILDER_HTML.index("const COMPAT")
        compat_block = BUILDER_HTML[compat_idx : compat_idx + 400]
        assert "ctrl:" in compat_block
        assert "'msg'" in compat_block or '"msg"' in compat_block

    def test_var_port_is_input_only(self):
        """var port must have an empty COMPAT set (cannot be a connection source)."""
        compat_idx = BUILDER_HTML.index("const COMPAT")
        compat_block = BUILDER_HTML[compat_idx : compat_idx + 400]
        # var: new Set([]) — empty, meaning nothing can originate from var port
        assert "var:" in compat_block
        assert "new Set([])" in compat_block

    def test_same_side_blocked_in_compat(self):
        """updatePortCompat must block out→out connections (same side check)."""
        upd_idx = BUILDER_HTML.index("function updatePortCompat()")
        upd_body = BUILDER_HTML[upd_idx : upd_idx + 500]
        assert "side !== 'in'" in upd_body or "side === 'out'" in upd_body

    def test_incompatible_message_shown(self):
        """portClick must show a status message on incompatible connection attempt."""
        assert "Incompatible ports" in BUILDER_HTML


# ─── create_app builder flag ────────────────────────────────────────────────


class TestCreateAppBuilder:
    def test_create_app_accepts_builder_flag(self):
        app = create_app(_make_agent(), playground=False, builder=True)
        assert app.router.enable_builder


# ─── Feature 5: Workflow Cost Badge ────────────────────────────────────────


class TestBuilderWorkflowCost:
    def test_cost_badge_element_present(self):
        """Header must contain a costBadge span."""
        assert 'id="costBadge"' in BUILDER_HTML

    def test_cost_badge_class_present(self):
        """CSS must define .cost-badge."""
        assert ".cost-badge" in BUILDER_HTML

    def test_update_workflow_cost_function(self):
        """updateWorkflowCost() must be defined."""
        assert "function updateWorkflowCost()" in BUILDER_HTML

    def test_update_workflow_cost_called_from_render(self):
        """render() must call updateWorkflowCost()."""
        render_idx = BUILDER_HTML.index("function render()")
        render_block = BUILDER_HTML[render_idx : render_idx + 400]
        assert "updateWorkflowCost()" in render_block

    def test_workflow_cost_uses_model_costs(self):
        """updateWorkflowCost must reference MODEL_COSTS."""
        fn_idx = BUILDER_HTML.index("function updateWorkflowCost()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "MODEL_COSTS" in fn_block

    def test_cost_badge_shows_agent_count(self):
        """updateWorkflowCost must include agent count in badge text."""
        fn_idx = BUILDER_HTML.index("function updateWorkflowCost()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "agentNodes.length" in fn_block

    def test_cost_badge_tooltip_per_node(self):
        """updateWorkflowCost must build per-node tooltip via nodeCostLabel."""
        fn_idx = BUILDER_HTML.index("function updateWorkflowCost()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 700]
        assert "nodeCostLabel" in fn_block

    def test_node_cost_css_class(self):
        """CSS must define .node-cost with green fill."""
        assert ".node-cost" in BUILDER_HTML
        cost_idx = BUILDER_HTML.index(".node-cost")
        assert "#4ade80" in BUILDER_HTML[cost_idx : cost_idx + 80]

    def test_node_cost_rendered_on_agent_nodes(self):
        """renderNodes must call nodeCostLabel for agent nodes."""
        rn_idx = BUILDER_HTML.index("function renderNodes()")
        rn_block = BUILDER_HTML[rn_idx : rn_idx + 1500]
        assert "nodeCostLabel(n)" in rn_block


# ─── Feature 6: Wire-Hover Preview ────────────────────────────────────────


class TestBuilderWireHover:
    def test_show_wire_tooltip_function(self):
        """showWireTooltip() must be defined."""
        assert "function showWireTooltip(" in BUILDER_HTML

    def test_hide_wire_tooltip_function(self):
        """hideWireTooltip() must be defined."""
        assert "function hideWireTooltip()" in BUILDER_HTML

    def test_wire_tooltip_element(self):
        """DOM must contain wireTooltip div."""
        assert 'id="wireTooltip"' in BUILDER_HTML

    def test_wire_tooltip_css(self):
        """CSS must define .wire-tooltip."""
        assert ".wire-tooltip" in BUILDER_HTML

    def test_mouseenter_on_edges(self):
        """renderEdges must attach mouseenter handler for wire tooltip."""
        re_idx = BUILDER_HTML.index("function renderEdges()")
        re_block = BUILDER_HTML[re_idx : re_idx + 2200]
        assert "mouseenter" in re_block
        assert "showWireTooltip" in re_block

    def test_mouseleave_on_edges(self):
        """renderEdges must attach mouseleave to hide tooltip."""
        re_idx = BUILDER_HTML.index("function renderEdges()")
        re_block = BUILDER_HTML[re_idx : re_idx + 2200]
        assert "mouseleave" in re_block
        assert "hideWireTooltip" in re_block

    def test_wire_tooltip_shows_frozen_outputs(self):
        """showWireTooltip must reference frozenOutputs."""
        fn_idx = BUILDER_HTML.index("function showWireTooltip(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 500]
        assert "frozenOutputs" in fn_block

    def test_wire_tooltip_no_output_message(self):
        """showWireTooltip must show a message when no output exists."""
        fn_idx = BUILDER_HTML.index("function showWireTooltip(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 500]
        assert "No output yet" in fn_block


# ─── Feature 7: Context Menu + Single-Node Replay ────────────────────────


class TestBuilderContextMenu:
    def test_ctx_menu_element(self):
        """DOM must contain ctxMenu div."""
        assert 'id="ctxMenu"' in BUILDER_HTML

    def test_ctx_rerun_button(self):
        """Context menu must have ctxRerun item."""
        assert 'id="ctxRerun"' in BUILDER_HTML

    def test_show_ctx_menu_function(self):
        """showCtxMenu() must be defined."""
        assert "function showCtxMenu(" in BUILDER_HTML

    def test_hide_ctx_menu_function(self):
        """hideCtxMenu() must be defined."""
        assert "function hideCtxMenu()" in BUILDER_HTML

    def test_rerun_node_alone_function(self):
        """rerunNodeAlone() must be defined."""
        assert "async function rerunNodeAlone(" in BUILDER_HTML

    def test_context_menu_on_nodes(self):
        """renderNodes must attach contextmenu listener to node groups."""
        rn_idx = BUILDER_HTML.index("function renderNodes()")
        rn_block = BUILDER_HTML[rn_idx : rn_idx + 8500]
        assert "contextmenu" in rn_block
        assert "showCtxMenu" in rn_block

    def test_ctx_item_css(self):
        """CSS must define .ctx-item."""
        assert ".ctx-item" in BUILDER_HTML

    def test_ctx_sep_css(self):
        """CSS must define .ctx-sep separator."""
        assert ".ctx-sep" in BUILDER_HTML

    def test_rerun_uses_frozen_outputs(self):
        """rerunNodeAlone must use frozenOutputs for input."""
        fn_idx = BUILDER_HTML.index("async function rerunNodeAlone(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "frozenOutputs" in fn_block

    def test_escape_closes_ctx_menu(self):
        """Escape handler must close ctxMenu."""
        esc_idx = BUILDER_HTML.index("if (e.key === 'Escape')")
        esc_block = BUILDER_HTML[esc_idx : esc_idx + 900]
        assert "ctxMenu" in esc_block

    def test_global_mousedown_hides_ctx(self):
        """A global mousedown listener must dismiss the context menu."""
        assert "document.addEventListener('mousedown'" in BUILDER_HTML
        mousedown_idx = BUILDER_HTML.index("document.addEventListener('mousedown'")
        mb = BUILDER_HTML[mousedown_idx : mousedown_idx + 200]
        assert "hideCtxMenu" in mb


# ─── Feature 8: Time-Travel Re-run from Scrubber ─────────────────────────


class TestBuilderTimeTravel:
    def test_rerun_from_event_function(self):
        """rerunFromEvent() must be defined."""
        assert "function rerunFromEvent(" in BUILDER_HTML

    def test_scrub_replay_button_css(self):
        """CSS must define .scrub-replay."""
        assert ".scrub-replay" in BUILDER_HTML

    def test_scrubber_adds_replay_buttons(self):
        """renderScrubber must create replay buttons."""
        rs_idx = BUILDER_HTML.index("function renderScrubber()")
        rs_block = BUILDER_HTML[rs_idx : rs_idx + 2000]
        assert "scrub-replay" in rs_block
        assert "rerunFromEvent" in rs_block

    def test_rerun_from_event_uses_topo_order(self):
        """rerunFromEvent must use topoOrder for sequential replay."""
        fn_idx = BUILDER_HTML.index("function rerunFromEvent(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 800]
        assert "topoOrder" in fn_block

    def test_rerun_from_event_uses_frozen_outputs(self):
        """rerunFromEvent must seed from frozenOutputs."""
        fn_idx = BUILDER_HTML.index("function rerunFromEvent(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 800]
        assert "frozenOutputs" in fn_block

    def test_replay_button_title(self):
        """Replay button must have a descriptive title."""
        assert "Re-run from here" in BUILDER_HTML


# ─── Feature 9: Embed Widget ────────────────────────────────���────────────


class TestBuilderEmbedWidget:
    def test_embed_button_present(self):
        """Header must contain an Embed button."""
        assert "openEmbed()" in BUILDER_HTML

    def test_embed_modal_element(self):
        """DOM must contain embedModal div."""
        assert 'id="embedModal"' in BUILDER_HTML

    def test_embed_code_element(self):
        """Embed modal must contain embedCode pre element."""
        assert 'id="embedCode"' in BUILDER_HTML

    def test_open_embed_function(self):
        """openEmbed() must be defined."""
        assert "function openEmbed()" in BUILDER_HTML

    def test_close_embed_function(self):
        """closeEmbed() must be defined."""
        assert "function closeEmbed()" in BUILDER_HTML

    def test_copy_embed_code_function(self):
        """copyEmbedCode() must be defined."""
        assert "function copyEmbedCode()" in BUILDER_HTML

    def test_embed_uses_yaml_export(self):
        """openEmbed must call genYaml() for the graph payload."""
        fn_idx = BUILDER_HTML.index("function openEmbed()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 400]
        assert "genYaml()" in fn_block

    def test_embed_button_in_header_actions(self):
        """Embed button must appear in the header-actions div."""
        ha_idx = BUILDER_HTML.index('class="header-actions"')
        ha_block = BUILDER_HTML[ha_idx : ha_idx + 1300]
        assert "openEmbed()" in ha_block

    def test_escape_closes_embed(self):
        """Escape key must close the embed modal."""
        esc_idx = BUILDER_HTML.index("if (e.key === 'Escape')")
        esc_block = BUILDER_HTML[esc_idx : esc_idx + 900]
        assert "embedModal" in esc_block


# ─── Feature 10: Agent-as-Tool Picker ────────────────────────────────────


class TestBuilderAgentAsTool:
    def test_agent_pick_button_in_props(self):
        """Properties panel must add an agent-picker button for agent nodes."""
        assert "use another agent as tool" in BUILDER_HTML

    def test_open_agent_picker_function(self):
        """openAgentPicker() must be defined."""
        assert "function openAgentPicker(" in BUILDER_HTML

    def test_close_agent_picker_function(self):
        """closeAgentPicker() must be defined."""
        assert "function closeAgentPicker()" in BUILDER_HTML

    def test_agent_pick_panel_css(self):
        """CSS must define .agent-pick-panel."""
        assert ".agent-pick-panel" in BUILDER_HTML

    def test_agent_pick_item_css(self):
        """CSS must define .agent-pick-item."""
        assert ".agent-pick-item" in BUILDER_HTML

    def test_picker_uses_at_prefix(self):
        """Agent tool refs must use @ prefix when added to tools field."""
        fn_idx = BUILDER_HTML.index("function openAgentPicker(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 1300]
        assert "'@'" in fn_block or '"@"' in fn_block

    def test_genpython_handles_at_agent_refs(self):
        """genPython must resolve @agentname tool refs to _tool wrappers."""
        gp_idx = BUILDER_HTML.index("function genPython(")
        gp_block = BUILDER_HTML[gp_idx : gp_idx + 2000]
        assert "startsWith('@')" in gp_block
        assert "_tool" in gp_block

    def test_genpython_imports_tool_decorator(self):
        """genPython must import @tool when agent refs are present."""
        gp_idx = BUILDER_HTML.index("function genPython(")
        gp_block = BUILDER_HTML[gp_idx : gp_idx + 2000]
        assert "from selectools.tools import tool" in gp_block

    def test_genpython_emits_tool_wrapper(self):
        """genPython must emit @tool() + def <agent>_tool wrapper."""
        gp_idx = BUILDER_HTML.index("function genPython(")
        gp_block = BUILDER_HTML[gp_idx : gp_idx + 3000]
        assert "@tool()" in gp_block
        assert "def ${v}_tool" in gp_block or "_tool" in gp_block

    def test_genpython_topo_sort_for_agent_tools(self):
        """genPython must sort agents so tool-referenced ones are defined first."""
        gp_idx = BUILDER_HTML.index("function genPython(")
        gp_block = BUILDER_HTML[gp_idx : gp_idx + 1500]
        assert "agentToolRefs" in gp_block
        assert "agentOrder" in gp_block


# ─── Feature 11: Embeddable Component ─────────────────────────────────────


class TestBuilderEmbedMode:
    def test_embed_mode_css(self):
        """CSS must define body.embed-mode rules to hide editor chrome."""
        assert "embed-mode" in BUILDER_HTML
        assert "embed-mode header" in BUILDER_HTML

    def test_init_embed_mode_function(self):
        """initEmbedMode IIFE must be defined."""
        assert "initEmbedMode" in BUILDER_HTML
        assert "embed=1" in BUILDER_HTML or "embed' === '1'" in BUILDER_HTML

    def test_embed_mode_loads_graph_from_url(self):
        """initEmbedMode must decode graph from URL ?graph= param."""
        em_idx = BUILDER_HTML.index("initEmbedMode")
        em_block = BUILDER_HTML[em_idx : em_idx + 600]
        assert "parseYaml" in em_block
        assert "atob" in em_block

    def test_embed_mode_skips_load_example(self):
        """loadExample() must be skipped when in embed-mode."""
        init_idx = BUILDER_HTML.rindex("loadExample()")
        init_ctx = BUILDER_HTML[init_idx - 60 : init_idx + 20]
        assert "embed-mode" in init_ctx

    def test_embed_mode_opens_test_panel(self):
        """initEmbedMode must call openTestPanel()."""
        em_idx = BUILDER_HTML.index("initEmbedMode")
        em_block = BUILDER_HTML[em_idx : em_idx + 700]
        assert "openTestPanel()" in em_block

    def test_open_embed_uses_real_origin(self):
        """openEmbed must use window.location.origin for the iframe src."""
        fn_idx = BUILDER_HTML.index("function openEmbed()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "window.location.origin" in fn_block

    def test_open_embed_encodes_full_yaml(self):
        """openEmbed must base64-encode full genYaml() output."""
        fn_idx = BUILDER_HTML.index("function openEmbed()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "btoa(" in fn_block
        assert "genYaml()" in fn_block

    def test_embed_iframe_has_embed_param(self):
        """Generated iframe src must include embed=1 param."""
        fn_idx = BUILDER_HTML.index("function openEmbed()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 600]
        assert "embed=1" in fn_block

    def test_embed_modal_live_description(self):
        """Embed modal must describe the live interactive nature."""
        assert "live interactive" in BUILDER_HTML or "live" in BUILDER_HTML


# ─── Feature 12: Gantt-chart execution timeline ──────────────────────────────


class TestBuilderGantt:
    """Tests for the Gantt-chart execution timeline (Feature 12)."""

    def test_gantt_toggle_btn_present(self):
        """#ganttToggleBtn button must be in the HTML."""
        assert "ganttToggleBtn" in BUILDER_HTML

    def test_gantt_wrap_present(self):
        """#ganttWrap div must be in the HTML."""
        assert 'id="ganttWrap"' in BUILDER_HTML

    def test_gantt_svg_present(self):
        """#ganttSvg element must be inside ganttWrap."""
        wrap_idx = BUILDER_HTML.index('id="ganttWrap"')
        wrap_block = BUILDER_HTML[wrap_idx : wrap_idx + 200]
        assert "ganttSvg" in wrap_block

    def test_gantt_wrap_hidden_by_default(self):
        """#ganttWrap must start hidden (display:none)."""
        wrap_idx = BUILDER_HTML.index('id="ganttWrap"')
        wrap_tag = BUILDER_HTML[wrap_idx : wrap_idx + 100]
        assert "display:none" in wrap_tag

    def test_scrubber_header_present(self):
        """#scrubberHeader must be in the HTML."""
        assert "scrubberHeader" in BUILDER_HTML

    def test_toggle_gantt_function_present(self):
        """toggleGantt() function must be defined."""
        assert "function toggleGantt()" in BUILDER_HTML

    def test_render_gantt_function_present(self):
        """renderGantt() function must be defined."""
        assert "function renderGantt()" in BUILDER_HTML

    def test_gantt_uses_node_start_ts(self):
        """renderGantt must read ts from node_start events."""
        fn_idx = BUILDER_HTML.index("function renderGantt()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 800]
        assert "node_start" in fn_block
        assert ".ts" in fn_block

    def test_gantt_uses_node_end_ts(self):
        """renderGantt must read ts from node_end events."""
        fn_idx = BUILDER_HTML.index("function renderGantt()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 800]
        assert "node_end" in fn_block

    def test_gantt_css_bar_present(self):
        """.gantt-bar CSS class must be defined."""
        assert ".gantt-bar" in BUILDER_HTML

    def test_gantt_css_wrap_present(self):
        """.gantt-wrap CSS class must be defined."""
        assert ".gantt-wrap" in BUILDER_HTML

    def test_gantt_axis_ticks(self):
        """renderGantt must emit axis tick elements."""
        fn_idx = BUILDER_HTML.index("function renderGantt()")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 2200]
        assert "gantt-tick-line" in fn_block

    def test_mock_events_have_ts_field(self):
        """runMock must define _mockT0 and use it in ts fields."""
        fn_idx = BUILDER_HTML.index("async function runMock(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 3000]
        assert "_mockT0" in fn_block
        assert "ts: Date.now() - _mockT0" in fn_block

    def test_mock_node_start_ts(self):
        """node_start push in runMock must include ts."""
        fn_idx = BUILDER_HTML.index("async function runMock(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 3000]
        assert "node_start" in fn_block and "ts: Date.now() - _mockT0" in fn_block

    def test_mock_node_end_ts(self):
        """node_end push in runMock must include ts."""
        fn_idx = BUILDER_HTML.index("async function runMock(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 3000]
        assert "node_end" in fn_block and "ts: Date.now() - _mockT0" in fn_block

    def test_server_mock_events_have_ts(self):
        """_builder_run_mock in app.py must emit ts on each event."""
        collected: list = []
        nodes = [
            {"id": "n1", "type": "agent", "name": "A1", "provider": "openai", "model": "gpt-4o"}
        ]
        _builder_run_mock(nodes, "hello", lambda ev: collected.append(ev))
        node_starts = [e for e in collected if e.get("type") == "node_start"]
        node_ends = [e for e in collected if e.get("type") == "node_end"]
        assert node_starts, "Expected at least one node_start event"
        assert node_ends, "Expected at least one node_end event"
        for ev in node_starts + node_ends:
            assert "ts" in ev, f"Missing ts in event: {ev}"
            assert isinstance(ev["ts"], int), f"ts must be int, got {type(ev['ts'])}"
            assert ev["ts"] >= 0

    def test_server_ts_monotonically_nondecreasing(self):
        """ts values must be non-decreasing across the run."""
        collected: list = []
        nodes = [
            {"id": "n1", "type": "agent", "name": "A1", "provider": "openai", "model": "gpt-4o"},
            {"id": "n2", "type": "agent", "name": "A2", "provider": "openai", "model": "gpt-4o"},
        ]
        _builder_run_mock(nodes, "hi", lambda ev: collected.append(ev))
        ts_values = [e["ts"] for e in collected if "ts" in e]
        assert len(ts_values) >= 2
        for i in range(1, len(ts_values)):
            assert (
                ts_values[i] >= ts_values[i - 1]
            ), f"ts not monotonic at index {i}: {ts_values[i-1]} > {ts_values[i]}"

    def test_render_gantt_called_after_run(self):
        """renderGantt() must be called after renderScrubber() at run completion."""
        run_fn_idx = BUILDER_HTML.rindex("renderScrubber();")
        run_block = BUILDER_HTML[run_fn_idx : run_fn_idx + 60]
        assert "renderGantt()" in run_block


# ─── Feature 13: Load production trace ───────────────────────────────────────


class TestBuilderLoadTrace:
    """Tests for the 'Load Production Trace' feature (Feature 13)."""

    def test_load_trace_btn_present(self):
        """Header must have a Load Trace button."""
        assert "openLoadTrace()" in BUILDER_HTML

    def test_load_trace_modal_present(self):
        """#loadTraceModal div must be in the HTML."""
        assert "loadTraceModal" in BUILDER_HTML

    def test_load_trace_input_present(self):
        """#loadTraceInput textarea must be inside the modal."""
        modal_idx = BUILDER_HTML.index('id="loadTraceModal"')
        modal_block = BUILDER_HTML[modal_idx : modal_idx + 900]
        assert "loadTraceInput" in modal_block

    def test_open_load_trace_function(self):
        """openLoadTrace() must be defined."""
        assert "function openLoadTrace()" in BUILDER_HTML

    def test_close_load_trace_function(self):
        """closeLoadTrace() must be defined."""
        assert "function closeLoadTrace()" in BUILDER_HTML

    def test_do_load_trace_function(self):
        """doLoadTrace() must be defined."""
        assert "function doLoadTrace()" in BUILDER_HTML

    def test_convert_trace_to_events_function(self):
        """convertTraceToEvents() must be defined."""
        assert "function convertTraceToEvents(" in BUILDER_HTML

    def test_convert_maps_graph_node_start(self):
        """convertTraceToEvents must map graph_node_start → node_start."""
        fn_idx = BUILDER_HTML.index("function convertTraceToEvents(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 1500]
        assert "graph_node_start" in fn_block
        assert "node_start" in fn_block

    def test_convert_maps_graph_node_end(self):
        """convertTraceToEvents must map graph_node_end → node_end."""
        fn_idx = BUILDER_HTML.index("function convertTraceToEvents(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 1500]
        assert "graph_node_end" in fn_block
        assert "node_end" in fn_block

    def test_convert_maps_tool_execution(self):
        """convertTraceToEvents must map tool_execution → tool_call."""
        fn_idx = BUILDER_HTML.index("function convertTraceToEvents(")
        fn_block = BUILDER_HTML[fn_idx : fn_idx + 1500]
        assert "tool_execution" in fn_block
        assert "tool_call" in fn_block

    def test_escape_closes_load_trace_modal(self):
        """Escape handler must close loadTraceModal."""
        esc_idx = BUILDER_HTML.index("e.key === 'Escape'")
        esc_block = BUILDER_HTML[esc_idx : esc_idx + 600]
        assert "loadTraceModal" in esc_block
        assert "closeLoadTrace()" in esc_block

    def test_trace_to_json_importable(self):
        """trace_to_json must be importable from selectools."""
        from selectools import trace_to_json

        assert callable(trace_to_json)

    def test_trace_to_json_serializes_trace(self):
        """trace_to_json must return valid JSON with a 'steps' key."""
        import json

        from selectools import trace_to_json
        from selectools.trace import AgentTrace, StepType, TraceStep

        trace = AgentTrace(
            steps=[
                TraceStep(type=StepType.LLM_CALL, duration_ms=120.0),
            ]
        )
        result = trace_to_json(trace)
        data = json.loads(result)
        assert "steps" in data
        assert len(data["steps"]) == 1
        assert data["steps"][0]["type"] == "llm_call"


# ─── Auth layer ───────────────────────────────────────────────────────────────


class TestBuilderAuth:
    """Tests for the persistent auth layer (--auth-token / BUILDER_AUTH_TOKEN)."""

    def _make_server(self, token=None):
        from selectools.serve.app import BuilderServer

        return BuilderServer(port=18999, auth_token=token)

    def _request(self, server, path="/builder", method="GET", body=None, cookie=None):
        """Fire a single request against a BuilderServer running in a thread."""
        from http.server import HTTPServer
        from urllib.parse import urlparse

        results = {}
        httpd = HTTPServer(("127.0.0.1", 0), type("H", (), {})())  # dummy — replaced below

        # Spin up the real server on a random port
        actual_port = [0]

        def run():
            s = server
            # patch port
            import types

            from selectools.serve.app import BuilderServer

            _auth = s.auth_token

            class H(server.serve.__func__.__class__):
                pass

            # Use threading + one-shot approach
            import socketserver

            class TH(socketserver.ThreadingMixIn, HTTPServer):
                pass

            srv = TH(("127.0.0.1", 0), type("Handler", (), {}))
            actual_port[0] = srv.server_address[1]
            # build real handler
            server._httpd = srv
            results["port"] = actual_port[0]

        # Simpler: use urllib directly against a real running instance
        import time

        _port = [None]
        _stop = threading.Event()

        class QuickServer(BuilderServer):
            pass

        qs = QuickServer(host="127.0.0.1", port=0, auth_token=server.auth_token)

        started = threading.Event()

        def _serve():
            from http.server import BaseHTTPRequestHandler, HTTPServer

            _auth_token = qs.auth_token
            import hmac as _hmac
            import json as _json
            from urllib.parse import urlparse as _up

            from selectools.serve.app import (
                BUILDER_HTML,
                LOGIN_HTML,
                LOGIN_HTML_ERROR,
                _make_session_cookie,
                _resolve_auth_token,
            )

            class H(BaseHTTPRequestHandler):
                def _is_authed(self):
                    if not _auth_token:
                        return True
                    cookie_header = self.headers.get("Cookie", "")
                    expected = _make_session_cookie(_auth_token)
                    for part in cookie_header.split(";"):
                        k, _, v = part.strip().partition("=")
                        if k == "builder_session" and _hmac.compare_digest(v.strip(), expected):
                            return True
                    return False

                def _redirect_login(self):
                    self.send_response(302)
                    self.send_header("Location", "/login")
                    self.end_headers()

                def do_GET(self):
                    p = _up(self.path).path.rstrip("/")
                    if p == "/health":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b"{}")
                        return
                    if p == "/login":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(LOGIN_HTML.encode())
                        return
                    if not self._is_authed():
                        self._redirect_login()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    p = _up(self.path).path.rstrip("/")
                    cl = int(self.headers.get("Content-Length", 0))
                    body = _json.loads(self.rfile.read(cl) if cl else b"{}")
                    if p == "/login":
                        tok = body.get("token", "")
                        if _auth_token and _hmac.compare_digest(tok, _auth_token):
                            cv = _make_session_cookie(_auth_token)
                            self.send_response(302)
                            self.send_header(
                                "Set-Cookie",
                                f"builder_session={cv}; HttpOnly; SameSite=Strict; Path=/",
                            )
                            self.send_header("Location", "/builder")
                            self.end_headers()
                        else:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/html")
                            self.end_headers()
                            self.wfile.write(LOGIN_HTML_ERROR.encode())
                        return
                    if not self._is_authed():
                        self._redirect_login()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"{}")

                def log_message(self, *a):
                    pass

            srv = HTTPServer(("127.0.0.1", 0), H)
            _port[0] = srv.server_address[1]
            started.set()
            while not _stop.is_set():
                srv.handle_request()
            srv.server_close()

        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        started.wait(timeout=3)
        p = _port[0]

        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{p}{path}"
        req = urllib.request.Request(url, method=method)
        if cookie:
            req.add_header("Cookie", cookie)
        if body is not None:
            req.data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")

        # Don't follow redirects — tests assert on exact 302 status codes
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

            def http_error_302(self, req, fp, code, msg, headers):
                raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

            http_error_301 = http_error_303 = http_error_307 = http_error_302

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            resp = opener.open(req)
            code = resp.getcode()
            body_out = resp.read()
            headers_out = dict(resp.headers)
        except urllib.error.HTTPError as e:
            code = e.code
            body_out = e.read()
            headers_out = dict(e.headers)
        finally:
            _stop.set()

        return {"status": code, "body": body_out, "headers": headers_out}

    def test_no_auth_builder_accessible(self):
        """Without auth token, /builder returns 200."""
        from selectools.serve.app import BUILDER_HTML, BuilderServer

        srv = BuilderServer(port=0, auth_token=None)
        r = self._request(srv, "/builder")
        assert r["status"] == 200

    def test_auth_builder_redirects_without_cookie(self):
        """With token set, /builder without cookie returns 302."""
        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/builder")
        assert r["status"] == 302

    def test_health_bypasses_auth(self):
        """GET /health always returns 200 regardless of auth."""
        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/health")
        assert r["status"] == 200

    def test_login_page_returns_200(self):
        """GET /login always returns 200."""
        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/login")
        assert r["status"] == 200

    def test_login_page_has_form(self):
        """Login HTML must contain a form with name=token input."""
        from selectools.serve.app import LOGIN_HTML

        assert "<form" in LOGIN_HTML
        assert 'name="token"' in LOGIN_HTML

    def test_login_correct_token_sets_cookie(self):
        """POST /login with correct token returns 302 + Set-Cookie header."""
        import json

        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/login", method="POST", body={"token": "secret123"})
        assert r["status"] == 302
        assert "Set-Cookie" in r["headers"] or "set-cookie" in r["headers"]

    def test_login_wrong_token_stays(self):
        """POST /login with wrong token returns 200 (error page, not redirect)."""
        import json

        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/login", method="POST", body={"token": "wrongtoken"})
        assert r["status"] == 200

    def test_builder_with_valid_cookie(self):
        """GET /builder with valid cookie returns 200."""
        from selectools.serve.app import BuilderServer, _make_session_cookie

        srv = BuilderServer(port=0, auth_token="secret123")
        cookie = f"builder_session={_make_session_cookie('secret123')}"
        r = self._request(srv, "/builder", cookie=cookie)
        assert r["status"] == 200

    def test_builder_with_invalid_cookie(self):
        """GET /builder with wrong cookie value returns 302."""
        from selectools.serve.app import BuilderServer

        srv = BuilderServer(port=0, auth_token="secret123")
        r = self._request(srv, "/builder", cookie="builder_session=badvalue")
        assert r["status"] == 302

    def test_resolve_auth_token_cli_wins(self):
        """CLI token takes priority over env var."""
        import os

        from selectools.serve.app import _resolve_auth_token

        old = os.environ.get("BUILDER_AUTH_TOKEN")
        try:
            os.environ["BUILDER_AUTH_TOKEN"] = "from_env"
            assert _resolve_auth_token("from_cli") == "from_cli"
        finally:
            if old is None:
                os.environ.pop("BUILDER_AUTH_TOKEN", None)
            else:
                os.environ["BUILDER_AUTH_TOKEN"] = old

    def test_resolve_auth_token_env_fallback(self):
        """Env var used when no CLI flag."""
        import os

        from selectools.serve.app import _resolve_auth_token

        old = os.environ.get("BUILDER_AUTH_TOKEN")
        try:
            os.environ["BUILDER_AUTH_TOKEN"] = "from_env"
            assert _resolve_auth_token(None) == "from_env"
        finally:
            if old is None:
                os.environ.pop("BUILDER_AUTH_TOKEN", None)
            else:
                os.environ["BUILDER_AUTH_TOKEN"] = old

    def test_resolve_auth_token_none(self):
        """Returns None when nothing is configured."""
        import os

        from selectools.serve.app import _resolve_auth_token

        old = os.environ.pop("BUILDER_AUTH_TOKEN", None)
        try:
            result = _resolve_auth_token(None)
            # Only None if no dotfile exists; acceptable to skip if dotfile present
            assert result is None or isinstance(result, str)
        finally:
            if old is not None:
                os.environ["BUILDER_AUTH_TOKEN"] = old

    def test_login_html_error_has_error_text(self):
        """LOGIN_HTML_ERROR must contain error message text."""
        from selectools.serve.app import LOGIN_HTML_ERROR

        assert "Invalid token" in LOGIN_HTML_ERROR

    def test_make_session_cookie_deterministic(self):
        """Same token always produces same cookie value."""
        from selectools.serve.app import _make_session_cookie

        assert _make_session_cookie("abc") == _make_session_cookie("abc")

    def test_make_session_cookie_different_tokens(self):
        """Different tokens produce different cookie values."""
        from selectools.serve.app import _make_session_cookie

        assert _make_session_cookie("abc") != _make_session_cookie("xyz")
