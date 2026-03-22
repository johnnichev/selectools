"""Self-contained HTML report renderer."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Union

from .types import CaseVerdict


def render_html_report(report: Any, filepath: Union[str, Path]) -> None:
    """Render an EvalReport as a self-contained HTML file."""
    rows = []
    for i, cr in enumerate(report.case_results):
        name = html.escape(cr.case.name or cr.case.input[:60])
        verdict_class = {
            CaseVerdict.PASS: "pass",
            CaseVerdict.FAIL: "fail",
            CaseVerdict.ERROR: "error",
            CaseVerdict.SKIP: "skip",
        }.get(cr.verdict, "")

        failure_html = ""
        if cr.failures:
            items = "".join(
                f"<li><strong>{html.escape(f.evaluator_name)}:</strong> "
                f"{html.escape(f.message)}</li>"
                for f in cr.failures
            )
            failure_html = f'<ul class="failures">{items}</ul>'
        elif cr.error:
            failure_html = f'<div class="error-msg">{html.escape(cr.error)}</div>'

        tools = ", ".join(cr.tool_calls) if cr.tool_calls else "-"

        rows.append(
            f"<tr class='{verdict_class}'>"
            f"<td>{i + 1}</td>"
            f"<td>{name}</td>"
            f"<td><span class='badge {verdict_class}'>{cr.verdict.value}</span></td>"
            f"<td>{cr.latency_ms:.0f}ms</td>"
            f"<td>${cr.cost_usd:.6f}</td>"
            f"<td>{html.escape(tools)}</td>"
            f"<td>{failure_html}</td>"
            f"</tr>"
        )

    table_rows = "\n".join(rows)

    failures_by_eval = report.failures_by_evaluator()
    eval_breakdown = ""
    if failures_by_eval:
        items = "".join(
            f"<li><strong>{html.escape(k)}:</strong> {v}</li>"
            for k, v in sorted(failures_by_eval.items(), key=lambda x: -x[1])
        )
        eval_breakdown = f"<h3>Failures by Evaluator</h3><ul>{items}</ul>"

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Report: {html.escape(report.metadata.suite_name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: Inter, system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
  padding: 2rem; max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
h2 {{ font-size: 1.3rem; color: #94a3b8; margin-bottom: 1.5rem; font-weight: 400; }}
h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem; margin-bottom: 2rem; }}
.stat {{ background: #1e293b; border-radius: 0.5rem; padding: 1rem; border: 1px solid #334155; }}
.stat-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
.stat-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }}
.stat-value.good {{ color: #4ade80; }}
.stat-value.warn {{ color: #fbbf24; }}
.stat-value.bad {{ color: #f87171; }}
table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 0.5rem;
  overflow: hidden; margin-bottom: 2rem; }}
th {{ text-align: left; padding: 0.75rem 1rem; background: #334155; font-size: 0.8rem;
  text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }}
td {{ padding: 0.75rem 1rem; border-top: 1px solid #334155; font-size: 0.875rem;
  vertical-align: top; }}
.badge {{ padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;
  font-weight: 600; text-transform: uppercase; }}
.badge.pass {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; }}
.badge.fail {{ background: rgba(248, 113, 113, 0.15); color: #f87171; }}
.badge.error {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; }}
.badge.skip {{ background: rgba(148, 163, 184, 0.15); color: #94a3b8; }}
.failures {{ list-style: none; font-size: 0.8rem; color: #f87171; }}
.failures li {{ margin-top: 0.25rem; }}
.error-msg {{ font-size: 0.8rem; color: #fbbf24; }}
ul {{ list-style: none; }}
ul li {{ padding: 0.25rem 0; font-size: 0.875rem; }}
footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155;
  font-size: 0.75rem; color: #64748b; }}
</style>
</head>
<body>
<h1>Eval Report: {html.escape(report.metadata.suite_name)}</h1>
<h2>{report.metadata.model or 'unknown model'} &middot;
  {report.metadata.provider or 'unknown provider'} &middot;
  {report.metadata.total_cases} cases &middot;
  {report.metadata.duration_ms:.0f}ms</h2>

<div class="summary">
  <div class="stat">
    <div class="stat-label">Accuracy</div>
    <div class="stat-value {'good' if report.accuracy >= 0.9 else 'warn' if report.accuracy >= 0.7 else 'bad'}">{report.accuracy:.1%}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Pass / Fail / Error</div>
    <div class="stat-value">{report.pass_count} / {report.fail_count} / {report.error_count}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Latency p50</div>
    <div class="stat-value">{report.latency_p50:.0f}ms</div>
  </div>
  <div class="stat">
    <div class="stat-label">Latency p95</div>
    <div class="stat-value">{report.latency_p95:.0f}ms</div>
  </div>
  <div class="stat">
    <div class="stat-label">Total Cost</div>
    <div class="stat-value">${report.total_cost:.6f}</div>
  </div>
  <div class="stat">
    <div class="stat-label">Total Tokens</div>
    <div class="stat-value">{report.total_tokens}</div>
  </div>
</div>

{eval_breakdown}

<table>
<thead>
<tr><th>#</th><th>Test Case</th><th>Verdict</th><th>Latency</th><th>Cost</th><th>Tools</th><th>Details</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<footer>
  Generated by Selectools v{html.escape(report.metadata.selectools_version)} &middot;
  Run ID: {html.escape(report.metadata.run_id)} &middot;
  An open-source project from NichevLabs
</footer>
</body>
</html>"""

    Path(filepath).write_text(content)
