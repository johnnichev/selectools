"""Live eval dashboard — real-time browser UI for eval runs."""

from __future__ import annotations

import html
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

from .types import CaseVerdict


class _DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the live dashboard."""

    # Note: dashboard_state is a class variable shared across handler instances.
    # Concurrent serve_eval() calls would overwrite each other's state.
    dashboard_state: Dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self._serve_dashboard()
        elif self.path == "/api/state":
            self._serve_state()
        else:
            self.send_error(404)

    def _serve_dashboard(self) -> None:
        content = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_state(self) -> None:
        data = json.dumps(self.dashboard_state).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress request logging


def serve_eval(
    suite: Any,
    *,
    port: int = 8888,
    open_browser: bool = True,
) -> Any:
    """Run an eval suite with a live browser dashboard.

    Starts a local HTTP server, opens the dashboard in a browser,
    runs the eval suite, and updates the dashboard in real-time.

    Args:
        suite: An EvalSuite instance.
        port: Port for the local server (default: 8888).
        open_browser: Whether to open the browser automatically.

    Returns:
        The final EvalReport.
    """
    state: Dict[str, Any] = {
        "status": "starting",
        "suite_name": suite.name,
        "total_cases": len(suite.cases),
        "completed": 0,
        "cases": [],
        "accuracy": 0.0,
        "pass_count": 0,
        "fail_count": 0,
        "error_count": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
    }
    _DashboardHandler.dashboard_state = state

    server = HTTPServer(("127.0.0.1", port), _DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"
    print(f"Live dashboard: {url}")

    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # nosec B110
            pass  # Browser open is best-effort

    # Patch on_progress to update state
    original_progress = suite.on_progress

    def _live_progress(done: int, total: int) -> None:
        state["completed"] = done
        state["status"] = "running"
        if original_progress:
            original_progress(done, total)

    suite.on_progress = _live_progress
    state["status"] = "running"

    try:
        report = suite.run()
    finally:
        suite.on_progress = original_progress

    # Final state update
    state["status"] = "complete"
    state["completed"] = len(report.case_results)
    state["accuracy"] = report.accuracy
    state["pass_count"] = report.pass_count
    state["fail_count"] = report.fail_count
    state["error_count"] = report.error_count
    state["total_cost"] = report.total_cost
    state["total_tokens"] = report.total_tokens
    state["latency_p50"] = report.latency_p50
    state["latency_p95"] = report.latency_p95
    state["duration_ms"] = report.metadata.duration_ms
    state["cases"] = [
        {
            "name": cr.case.name or cr.case.input[:50],
            "verdict": cr.verdict.value,
            "latency_ms": cr.latency_ms,
            "cost_usd": cr.cost_usd,
            "failures": len(cr.failures),
        }
        for cr in report.case_results
    ]

    print(f"\nDashboard still running at {url} — press Ctrl+C to stop")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        server.shutdown()

    return report


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Selectools Eval — Live Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;max-width:900px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:0.5rem}
.status{font-size:0.9rem;color:#94a3b8;margin-bottom:1.5rem}
.status .running{color:#fbbf24}.status .complete{color:#4ade80}
.progress{background:#1e293b;border-radius:0.5rem;height:24px;overflow:hidden;margin-bottom:1.5rem;border:1px solid #334155}
.progress-bar{height:100%;background:linear-gradient(90deg,#3b82f6,#06b6d4);transition:width 0.3s ease;border-radius:0.5rem}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.5rem}
.stat{background:#1e293b;border-radius:0.5rem;padding:0.75rem;border:1px solid #334155;text-align:center}
.stat-label{font-size:0.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em}
.stat-value{font-size:1.3rem;font-weight:700;margin-top:0.15rem}
.good{color:#4ade80}.warn{color:#fbbf24}.bad{color:#f87171}
.cases{background:#1e293b;border-radius:0.5rem;border:1px solid #334155;overflow:hidden}
.case-row{display:flex;padding:0.5rem 0.75rem;border-top:1px solid #334155;font-size:0.8rem;align-items:center;gap:0.75rem}
.case-row:first-child{border-top:none}
.case-name{flex:1}
.badge{padding:0.1rem 0.4rem;border-radius:0.2rem;font-size:0.7rem;font-weight:600;text-transform:uppercase}
.badge.pass{background:rgba(74,222,128,0.15);color:#4ade80}
.badge.fail{background:rgba(248,113,113,0.15);color:#f87171}
.badge.error{background:rgba(251,191,36,0.15);color:#fbbf24}
.latency{color:#94a3b8;width:70px;text-align:right}
footer{margin-top:1.5rem;font-size:0.7rem;color:#475569;text-align:center}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.pulsing{animation:pulse 1.5s infinite}
</style>
</head>
<body>
<h1>Selectools Eval — Live Dashboard</h1>
<div class="status" id="status">Connecting...</div>
<div class="progress"><div class="progress-bar" id="progress" style="width:0%"></div></div>
<div class="grid">
  <div class="stat"><div class="stat-label">Accuracy</div><div class="stat-value" id="accuracy">—</div></div>
  <div class="stat"><div class="stat-label">Pass / Fail</div><div class="stat-value" id="counts">—</div></div>
  <div class="stat"><div class="stat-label">Cost</div><div class="stat-value" id="cost">—</div></div>
  <div class="stat"><div class="stat-label">Latency p50</div><div class="stat-value" id="latency">—</div></div>
</div>
<div class="cases" id="cases"></div>
<footer>Selectools Eval — an open-source project from NichevLabs</footer>
<script>
async function poll(){
  try{
    const r=await fetch('/api/state');
    const s=await r.json();
    const pct=s.total_cases?Math.round(s.completed/s.total_cases*100):0;
    document.getElementById('progress').style.width=pct+'%';
    const statusEl=document.getElementById('status');
    if(s.status==='running'){
      statusEl.innerHTML='<span class="running pulsing">Running...</span> '+s.completed+'/'+s.total_cases+' cases';
    }else if(s.status==='complete'){
      statusEl.innerHTML='<span class="complete">Complete</span> — '+s.total_cases+' cases in '+(s.duration_ms||0).toFixed(0)+'ms';
    }else{
      statusEl.textContent='Starting...';
    }
    if(s.status==='running'||s.status==='complete'){
      const accEl=document.getElementById('accuracy');
      accEl.textContent=(s.accuracy*100).toFixed(1)+'%';
      accEl.className='stat-value '+(s.accuracy>=0.9?'good':s.accuracy>=0.7?'warn':'bad');
      document.getElementById('counts').textContent=s.pass_count+' / '+s.fail_count;
      document.getElementById('cost').textContent='$'+s.total_cost.toFixed(4);
      document.getElementById('latency').textContent=(s.latency_p50||0).toFixed(0)+'ms';
    }
    const casesEl=document.getElementById('cases');
    casesEl.innerHTML='';s.cases.forEach(c=>{const row=document.createElement('div');row.className='case-row';const badge=document.createElement('span');badge.className='badge '+c.verdict;badge.textContent=c.verdict;const nm=document.createElement('span');nm.className='case-name';nm.textContent=c.name;const lat=document.createElement('span');lat.className='latency';lat.textContent=c.latency_ms.toFixed(0)+'ms';row.append(badge,nm,lat);casesEl.appendChild(row)});
  }catch(e){}
  if(document.getElementById('status').textContent.indexOf('Complete')===-1){
    setTimeout(poll,500);
  }else{
    setTimeout(poll,2000);
  }
}
poll();
</script>
</body>
</html>"""
