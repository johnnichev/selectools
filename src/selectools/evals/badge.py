"""Generate SVG eval badges for README and CI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union
from xml.sax.saxutils import escape as _xml_escape  # nosec B406


def _badge_color(accuracy: float) -> str:
    """Return badge color based on accuracy."""
    if accuracy >= 0.95:
        return "#4ade80"  # green
    if accuracy >= 0.9:
        return "#22d3ee"  # cyan
    if accuracy >= 0.8:
        return "#3b82f6"  # blue
    if accuracy >= 0.7:
        return "#fbbf24"  # yellow
    if accuracy >= 0.5:
        return "#f97316"  # orange
    return "#f87171"  # red


def generate_badge(
    report: Any,
    filepath: Union[str, Path],
    *,
    label: str = "eval",
) -> None:
    """Generate an SVG badge showing eval accuracy.

    Creates a shields.io-style badge like: [eval | 95%]

    Args:
        report: An EvalReport instance.
        filepath: Path to write the SVG file.
        label: Left-side label text.
    """
    value = f"{report.accuracy:.0%}"
    color = _badge_color(report.accuracy)
    safe_label = _xml_escape(label)
    safe_value = _xml_escape(value)

    # Calculate text widths (approximate: 6.5px per char)
    label_width = len(label) * 6.5 + 12
    value_width = len(value) * 6.5 + 12
    total_width = label_width + value_width

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.0f}" height="20">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total_width:.0f}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width:.0f}" height="20" fill="#555"/>
    <rect x="{label_width:.0f}" width="{value_width:.0f}" height="20" fill="{color}"/>
    <rect width="{total_width:.0f}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{safe_label}</text>
    <text x="{label_width / 2:.0f}" y="14">{safe_label}</text>
    <text x="{label_width + value_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{safe_value}</text>
    <text x="{label_width + value_width / 2:.0f}" y="14">{safe_value}</text>
  </g>
</svg>"""

    dest = Path(filepath)
    tmp = dest.with_suffix(".svg.tmp")
    tmp.write_text(svg)
    tmp.replace(dest)


def generate_detailed_badge(
    report: Any,
    filepath: Union[str, Path],
) -> None:
    """Generate a wider badge with accuracy + pass/fail counts.

    Example: [eval | 95% · 19/20 pass]
    """
    value = f"{report.accuracy:.0%} \u00b7 {report.pass_count}/{report.metadata.total_cases} pass"
    color = _badge_color(report.accuracy)
    label = "eval"
    safe_label = _xml_escape(label)
    safe_value = _xml_escape(value)

    label_width = len(label) * 6.5 + 12
    value_width = len(value) * 6.5 + 12
    total_width = label_width + value_width

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.0f}" height="20">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total_width:.0f}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width:.0f}" height="20" fill="#555"/>
    <rect x="{label_width:.0f}" width="{value_width:.0f}" height="20" fill="{color}"/>
    <rect width="{total_width:.0f}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{safe_label}</text>
    <text x="{label_width / 2:.0f}" y="14">{safe_label}</text>
    <text x="{label_width + value_width / 2:.0f}" y="15" fill="#010101" fill-opacity=".3">{safe_value}</text>
    <text x="{label_width + value_width / 2:.0f}" y="14">{safe_value}</text>
  </g>
</svg>"""

    dest = Path(filepath)
    tmp = dest.with_suffix(".svg.tmp")
    tmp.write_text(svg)
    tmp.replace(dest)
