"""Visual Agent Builder — assembles self-contained HTML from _static/ source files."""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).parent / "_static"


def _build_html() -> str:
    """Assemble the single-file builder HTML by inlining CSS and JS."""
    css = (_STATIC / "builder.css").read_text(encoding="utf-8")
    js = (_STATIC / "builder.js").read_text(encoding="utf-8")
    template = (_STATIC / "builder.html").read_text(encoding="utf-8")
    return template.replace("{{CSS}}", css).replace("{{JS}}", js)


BUILDER_HTML = _build_html()
