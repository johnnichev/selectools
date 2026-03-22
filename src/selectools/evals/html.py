"""Self-contained interactive HTML report renderer."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, List, Optional, Union

from .types import CaseVerdict


def _donut_svg(pass_n: int, fail_n: int, error_n: int, skip_n: int) -> str:
    """Generate an SVG donut chart for pass/fail/error/skip distribution."""
    total = pass_n + fail_n + error_n + skip_n
    if total == 0:
        return ""
    segments = [
        (pass_n, "#4ade80"),
        (fail_n, "#f87171"),
        (error_n, "#fbbf24"),
        (skip_n, "#64748b"),
    ]
    cx, cy, r = 60, 60, 50
    inner_r = 35
    paths: List[str] = []
    start_angle = -90.0
    for count, color in segments:
        if count == 0:
            continue
        sweep = (count / total) * 360
        end_angle = start_angle + sweep
        large = 1 if sweep > 180 else 0
        sa = math.radians(start_angle)
        ea = math.radians(end_angle)
        x1_o, y1_o = cx + r * math.cos(sa), cy + r * math.sin(sa)
        x2_o, y2_o = cx + r * math.cos(ea), cy + r * math.sin(ea)
        x1_i, y1_i = cx + inner_r * math.cos(ea), cy + inner_r * math.sin(ea)
        x2_i, y2_i = cx + inner_r * math.cos(sa), cy + inner_r * math.sin(sa)
        d = (
            f"M {x1_o:.1f} {y1_o:.1f} "
            f"A {r} {r} 0 {large} 1 {x2_o:.1f} {y2_o:.1f} "
            f"L {x1_i:.1f} {y1_i:.1f} "
            f"A {inner_r} {inner_r} 0 {large} 0 {x2_i:.1f} {y2_i:.1f} Z"
        )
        paths.append(f'<path d="{d}" fill="{color}" opacity="0.9"/>')
        start_angle = end_angle
    return f'<svg viewBox="0 0 120 120" width="120" height="120">' f'{"".join(paths)}</svg>'


def _histogram_svg(latencies: List[float]) -> str:
    """Generate an SVG histogram of latency distribution."""
    if not latencies:
        return ""
    min_v = min(latencies)
    max_v = max(latencies)
    if max_v == min_v:
        max_v = min_v + 1
    n_bins = min(12, len(latencies))
    bin_width = (max_v - min_v) / n_bins
    bins = [0] * n_bins
    for v in latencies:
        idx = min(int((v - min_v) / bin_width), n_bins - 1)
        bins[idx] += 1
    max_count = max(bins) or 1
    w, h = 300, 100
    bar_w = w / n_bins - 2
    bars: List[str] = []
    for i, count in enumerate(bins):
        bar_h = (count / max_count) * (h - 20)
        x = i * (w / n_bins) + 1
        y = h - 15 - bar_h
        bars.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" '
            f'height="{bar_h:.0f}" fill="#3b82f6" rx="2" opacity="0.8"/>'
        )
        label_v = min_v + (i + 0.5) * bin_width
        if i % max(1, n_bins // 4) == 0:
            bars.append(
                f'<text x="{x + bar_w / 2:.0f}" y="{h - 2}" fill="#64748b" '
                f'font-size="8" text-anchor="middle">{label_v:.0f}</text>'
            )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="margin-top:0.5rem">{" ".join(bars)}'
        f'<text x="{w // 2}" y="10" fill="#94a3b8" font-size="9" '
        f'text-anchor="middle">Latency Distribution (ms)</text></svg>'
    )


def _trend_svg(accuracies: List[float]) -> str:
    """Generate an SVG sparkline for accuracy trend over time."""
    if len(accuracies) < 2:
        return ""
    w, h = 200, 60
    n = len(accuracies)
    max_v = max(accuracies) if max(accuracies) > 0 else 1.0
    min_v = min(accuracies)
    v_range = max_v - min_v if max_v != min_v else 0.1

    points: List[str] = []
    for i, v in enumerate(accuracies):
        x = i * (w - 20) / (n - 1) + 10
        y = h - 10 - ((v - min_v) / v_range) * (h - 25)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    # Color based on trend
    color = "#4ade80" if accuracies[-1] >= accuracies[0] else "#f87171"

    dots = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="2.5" fill="{color}"/>'
        for p in points
    )

    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round"/>'
        f"{dots}"
        f'<text x="{w // 2}" y="10" fill="#94a3b8" font-size="9" '
        f'text-anchor="middle">Accuracy Trend</text></svg>'
    )


def render_html_report(  # noqa: C901
    report: Any,
    filepath: Union[str, Path],
    history: Optional[Any] = None,
) -> None:
    """Render an EvalReport as a self-contained interactive HTML file.

    Args:
        report: EvalReport instance.
        filepath: Path to write the HTML file.
        history: Optional HistoryTrend instance for trend chart.
    """
    # Build table rows with expandable details
    rows = []
    for i, cr in enumerate(report.case_results):
        name = html.escape(cr.case.name or cr.case.input[:60])
        input_text = html.escape(cr.case.input[:300])
        verdict_class = {
            CaseVerdict.PASS: "pass",
            CaseVerdict.FAIL: "fail",
            CaseVerdict.ERROR: "error",
            CaseVerdict.SKIP: "skip",
        }.get(cr.verdict, "")

        # Expandable detail content
        detail_parts = [f"<strong>Input:</strong> {input_text}"]
        if cr.agent_result:
            output = html.escape((cr.agent_result.content or "")[:500])
            detail_parts.append(f"<strong>Output:</strong> {output}")
            if cr.agent_result.reasoning:
                reasoning = html.escape(str(cr.agent_result.reasoning)[:300])
                detail_parts.append(f"<strong>Reasoning:</strong> {reasoning}")
        if cr.tool_calls:
            detail_parts.append(f"<strong>Tools:</strong> {html.escape(', '.join(cr.tool_calls))}")
        if cr.failures:
            items = "".join(
                f"<li><span class='fail-label'>{html.escape(f.evaluator_name)}:</span> "
                f"{html.escape(f.message)}</li>"
                for f in cr.failures
            )
            detail_parts.append(f"<strong>Failures:</strong><ul class='failures'>{items}</ul>")
        if cr.error:
            detail_parts.append(
                f"<strong>Error:</strong> <span class='error-msg'>"
                f"{html.escape(cr.error)}</span>"
            )

        detail_html = "<br>".join(detail_parts)
        tags_data = html.escape(" ".join(cr.case.tags)) if cr.case.tags else ""
        fail_count = len(cr.failures) if cr.failures else (1 if cr.error else 0)

        # Build tag pills outside f-string to avoid backslash issue
        tag_pills = ""
        if cr.case.tags:
            pill_items = "".join(
                '<span class="tag">' + html.escape(t) + "</span>" for t in cr.case.tags
            )
            tag_pills = '<span class="tag-pills">' + pill_items + "</span>"

        rows.append(
            f"<tr class='case-row {verdict_class}' data-verdict='{cr.verdict.value}' "
            f"data-tags='{tags_data}' onclick='toggleDetail({i})'>"
            f"<td>{i + 1}</td>"
            f"<td><span class='case-name'>{name}</span>{tag_pills}</td>"
            f"<td><span class='badge {verdict_class}'>{cr.verdict.value}</span></td>"
            f"<td>{cr.latency_ms:.0f}ms</td>"
            f"<td>${cr.cost_usd:.6f}</td>"
            f"<td>{fail_count}</td>"
            f"</tr>"
            f"<tr class='detail-row' id='detail-{i}' style='display:none'>"
            f"<td colspan='6'><div class='detail-content'>{detail_html}</div></td>"
            f"</tr>"
        )

    table_rows = "\n".join(rows)

    # Charts
    donut = _donut_svg(report.pass_count, report.fail_count, report.error_count, report.skip_count)
    latencies = [cr.latency_ms for cr in report.case_results if cr.verdict != CaseVerdict.SKIP]
    histogram = _histogram_svg(latencies)
    trend_chart = ""
    if history and hasattr(history, "accuracy_trend") and len(history.accuracy_trend) >= 2:
        trend_chart = _trend_svg(history.accuracy_trend)

    # Failure breakdown
    failures_by_eval = report.failures_by_evaluator()
    eval_bars = ""
    if failures_by_eval:
        max_f = max(failures_by_eval.values())
        bars = "".join(
            f"<div class='eval-bar-row'>"
            f"<span class='eval-bar-label'>{html.escape(k)}</span>"
            f"<div class='eval-bar-track'><div class='eval-bar-fill' "
            f"style='width:{v / max_f * 100:.0f}%'></div></div>"
            f"<span class='eval-bar-count'>{v}</span></div>"
            for k, v in sorted(failures_by_eval.items(), key=lambda x: -x[1])
        )
        eval_bars = f"<div class='eval-bars'><h3>Failures by Evaluator</h3>{bars}</div>"

    # Collect unique tags for filter buttons
    all_tags = sorted({t for cr in report.case_results for t in cr.case.tags})
    tag_buttons = "".join(
        f"<button class='filter-btn' onclick='filterByTag(\"{html.escape(t)}\")'>"
        f"{html.escape(t)}</button>"
        for t in all_tags
    )
    filter_bar = ""
    if all_tags:
        filter_bar = (
            f"<div class='filter-bar'>"
            f"<button class='filter-btn active' onclick='filterByTag(\"\")'>All</button>"
            f"<button class='filter-btn' onclick='filterByVerdict(\"fail\")'>Failures</button>"
            f"<button class='filter-btn' onclick='filterByVerdict(\"error\")'>Errors</button>"
            f"{tag_buttons}</div>"
        )
    else:
        filter_bar = (
            "<div class='filter-bar'>"
            "<button class='filter-btn active' onclick='filterByTag(\"\")'>All</button>"
            "<button class='filter-btn' onclick='filterByVerdict(\"fail\")'>Failures</button>"
            "<button class='filter-btn' onclick='filterByVerdict(\"error\")'>Errors</button>"
            "</div>"
        )

    acc_class = "good" if report.accuracy >= 0.9 else "warn" if report.accuracy >= 0.7 else "bad"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Report: {html.escape(report.metadata.suite_name)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Inter,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;max-width:1280px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:0.25rem}}
.subtitle{{font-size:1rem;color:#94a3b8;margin-bottom:1.5rem}}
h3{{font-size:1rem;margin:1rem 0 0.5rem;color:#cbd5e1}}
.top-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:2rem}}
@media(max-width:768px){{.top-grid{{grid-template-columns:1fr}}}}
.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem}}
.stat{{background:#1e293b;border-radius:0.5rem;padding:0.75rem 1rem;border:1px solid #334155}}
.stat-label{{font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em}}
.stat-value{{font-size:1.4rem;font-weight:700;margin-top:0.15rem}}
.good{{color:#4ade80}}.warn{{color:#fbbf24}}.bad{{color:#f87171}}
.charts{{background:#1e293b;border-radius:0.5rem;padding:1rem;border:1px solid #334155;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.5rem}}
.charts-row{{display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;justify-content:center}}
.legend{{display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.5rem}}
.legend-item{{display:flex;align-items:center;gap:0.3rem;font-size:0.75rem;color:#94a3b8}}
.legend-dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
.eval-bars{{background:#1e293b;border-radius:0.5rem;padding:1rem;border:1px solid #334155;margin-bottom:1.5rem}}
.eval-bar-row{{display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0}}
.eval-bar-label{{width:140px;font-size:0.75rem;color:#94a3b8;text-align:right;flex-shrink:0}}
.eval-bar-track{{flex:1;height:16px;background:#334155;border-radius:3px;overflow:hidden}}
.eval-bar-fill{{height:100%;background:#f87171;border-radius:3px;transition:width 0.5s ease}}
.eval-bar-count{{width:30px;font-size:0.75rem;color:#94a3b8}}
.filter-bar{{display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap}}
.filter-btn{{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:0.3rem 0.75rem;border-radius:0.25rem;font-size:0.75rem;cursor:pointer;transition:all 0.2s}}
.filter-btn:hover,.filter-btn.active{{border-color:#3b82f6;color:#e2e8f0}}
table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:0.5rem;overflow:hidden;margin-bottom:1.5rem}}
th{{text-align:left;padding:0.6rem 0.75rem;background:#334155;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:#94a3b8}}
td{{padding:0.6rem 0.75rem;border-top:1px solid #1e293b;font-size:0.8rem;vertical-align:top}}
.case-row{{cursor:pointer;transition:background 0.15s}}.case-row:hover{{background:#334155}}
.case-name{{font-weight:500}}
.tag-pills{{margin-left:0.5rem}}.tag{{background:#334155;color:#94a3b8;font-size:0.65rem;padding:0.1rem 0.4rem;border-radius:0.2rem;margin-left:0.25rem}}
.badge{{padding:0.15rem 0.5rem;border-radius:0.25rem;font-size:0.7rem;font-weight:600;text-transform:uppercase}}
.badge.pass{{background:rgba(74,222,128,0.15);color:#4ade80}}
.badge.fail{{background:rgba(248,113,113,0.15);color:#f87171}}
.badge.error{{background:rgba(251,191,36,0.15);color:#fbbf24}}
.badge.skip{{background:rgba(148,163,184,0.15);color:#94a3b8}}
.detail-row td{{padding:0;background:#0f172a}}
.detail-content{{padding:1rem 1.5rem;font-size:0.8rem;line-height:1.6;color:#94a3b8;border-left:3px solid #3b82f6;margin:0.25rem 0 0.25rem 1rem}}
.detail-content strong{{color:#e2e8f0}}
.failures{{list-style:none;margin-top:0.25rem}}.failures li{{margin:0.15rem 0;color:#f87171;font-size:0.8rem}}
.fail-label{{color:#f87171;font-weight:600}}
.error-msg{{color:#fbbf24}}
footer{{margin-top:1.5rem;padding-top:1rem;border-top:1px solid #334155;font-size:0.7rem;color:#475569;display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem}}
</style>
</head>
<body>

<h1>Eval Report: {html.escape(report.metadata.suite_name)}</h1>
<div class="subtitle">{report.metadata.model or 'unknown model'} &middot; {report.metadata.provider or 'unknown provider'} &middot; {report.metadata.total_cases} cases &middot; {report.metadata.duration_ms:.0f}ms</div>

<div class="top-grid">
  <div>
    <div class="summary">
      <div class="stat"><div class="stat-label">Accuracy</div><div class="stat-value {acc_class}">{report.accuracy:.1%}</div></div>
      <div class="stat"><div class="stat-label">Pass</div><div class="stat-value good">{report.pass_count}</div></div>
      <div class="stat"><div class="stat-label">Fail</div><div class="stat-value {'bad' if report.fail_count else ''}">{report.fail_count}</div></div>
      <div class="stat"><div class="stat-label">Latency p50</div><div class="stat-value">{report.latency_p50:.0f}ms</div></div>
      <div class="stat"><div class="stat-label">Latency p95</div><div class="stat-value">{report.latency_p95:.0f}ms</div></div>
      <div class="stat"><div class="stat-label">Total Cost</div><div class="stat-value">${report.total_cost:.4f}</div></div>
      <div class="stat"><div class="stat-label">Cost/Case</div><div class="stat-value">${report.cost_per_case:.6f}</div></div>
      <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">{report.total_tokens:,}</div></div>
      <div class="stat"><div class="stat-label">Errors</div><div class="stat-value {'warn' if report.error_count else ''}">{report.error_count}</div></div>
    </div>
  </div>
  <div class="charts">
    <div class="charts-row">
      {donut}
      <div>{histogram}</div>
      {f'<div>{trend_chart}</div>' if trend_chart else ''}
    </div>
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#4ade80"></span>Pass ({report.pass_count})</span>
      <span class="legend-item"><span class="legend-dot" style="background:#f87171"></span>Fail ({report.fail_count})</span>
      <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Error ({report.error_count})</span>
      <span class="legend-item"><span class="legend-dot" style="background:#64748b"></span>Skip ({report.skip_count})</span>
    </div>
  </div>
</div>

{eval_bars}

{filter_bar}

<table id="results-table">
<thead>
<tr><th>#</th><th>Test Case</th><th>Verdict</th><th>Latency</th><th>Cost</th><th>Issues</th></tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<footer>
  <span>Generated by Selectools v{html.escape(report.metadata.selectools_version)} &middot; Run ID: {html.escape(report.metadata.run_id)}</span>
  <span>An open-source project from <a href="https://nichevlabs.com" style="color:#06b6d4;text-decoration:none">NichevLabs</a></span>
</footer>

<script>
function toggleDetail(i){{
  const row=document.getElementById('detail-'+i);
  row.style.display=row.style.display==='none'?'table-row':'none';
}}
function filterByTag(tag){{
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.case-row').forEach(r=>{{
    const tags=r.dataset.tags||'';
    r.style.display=(!tag||tags.includes(tag))?'':'none';
    const id=r.nextElementSibling?.id;
    if(id)document.getElementById(id).style.display='none';
  }});
}}
function filterByVerdict(v){{
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.case-row').forEach(r=>{{
    r.style.display=r.dataset.verdict===v?'':'none';
    const id=r.nextElementSibling?.id;
    if(id)document.getElementById(id).style.display='none';
  }});
}}
</script>
</body>
</html>"""

    Path(filepath).write_text(page)
