#!/usr/bin/env python3
"""Keep marketing-surface version strings in sync with pyproject.toml.

The docs site, README banner, landing page, and OG social cards each carry
the current version in prose/markup. Three releases in one day (1.1 -> 1.3)
showed these drift silently: the served OG card sat at v0.20.1 for months
because nothing tied it to the package version.

This script is the single source of truth binding. Run modes:

    python scripts/sync_marketing_version.py           # rewrite drifted files
    python scripts/sync_marketing_version.py --check    # exit 1 on any drift

``--check`` runs in CI so a stale marketing version fails the build instead
of shipping. Write mode is called from the release flow.

NOTE: the OG *PNG* (landing/assets/og-image.png) is a raster derived from
landing/assets/og-image.svg. This script updates the SVG text and reminds
you to re-render the PNG; it does not rasterize (to avoid font-fallback
drift from an ad-hoc CI renderer). Re-render with the pinned tool:

    scripts/render_og_image.py   # headless Chromium, 1200x630
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_version() -> str:
    """The one source of truth: pyproject.toml ``version``."""
    content = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise SystemExit("could not find version in pyproject.toml")
    return match.group(1)


def _minor(version: str) -> str:
    """``1.3.0`` -> ``1.3`` for the badges/banner that use the minor form."""
    parts = version.split(".")
    return ".".join(parts[:2])


@dataclass
class Rule:
    """A single file + a regex whose one match holds the version token."""

    path: str
    pattern: "re.Pattern[str]"
    # (version) -> the correct full matched string
    replacement: Callable[[str], str]
    required: bool = True


# Each rule pins ONE occurrence. A drift or a count change (someone adds a
# second version mention) is surfaced rather than silently half-fixed.
RULES: List[Rule] = [
    # README top banner: "Latest: [v1.3](#whats-new-in-v13) ..."
    Rule(
        path="README.md",
        pattern=re.compile(r"Latest: \[v(\d+\.\d+)\]\(#whats-new-in-v\d+\)"),
        replacement=lambda v: (
            f"Latest: [v{_minor(v)}](#whats-new-in-v{_minor(v).replace('.', '')})"
        ),
    ),
    # Landing status bar: <span class="status-val">v1.3.0</span>
    Rule(
        path="landing/index.html",
        pattern=re.compile(r'<span class="status-val">v(\d+\.\d+\.\d+)</span>'),
        replacement=lambda v: f'<span class="status-val">v{v}</span>',
    ),
    # Landing footer terminal comment: "# selectools v1.3.0 &middot;"
    Rule(
        path="landing/index.html",
        pattern=re.compile(r"# selectools v(\d+\.\d+\.\d+) &middot;"),
        replacement=lambda v: f"# selectools v{v} &middot;",
    ),
    # Landing OG SVG footer: "v1.3.0 · Apache-2.0"
    Rule(
        path="landing/assets/og-image.svg",
        pattern=re.compile(r"v(\d+\.\d+\.\d+) · Apache-2\.0"),
        replacement=lambda v: f"v{v} · Apache-2.0",
    ),
    # Docs OG card badge: "v1.3 · STABLE"
    Rule(
        path="docs/assets/og-card.html",
        pattern=re.compile(r"v(\d+\.\d+) · STABLE"),
        replacement=lambda v: f"v{_minor(v)} · STABLE",
    ),
    # Docs OG SVG badge: "v1.3 &#183; STABLE"
    Rule(
        path="docs/assets/og-image.svg",
        pattern=re.compile(r"v(\d+\.\d+) &#183; STABLE"),
        replacement=lambda v: f"v{_minor(v)} &#183; STABLE",
    ),
]


def apply_rule(rule: Rule, version: str, *, check: bool) -> List[str]:
    """Return a list of human-readable drift messages (empty when in sync)."""
    path = REPO_ROOT / rule.path
    if not path.exists():
        return [] if not rule.required else [f"{rule.path}: missing"]
    text = path.read_text()
    pat = rule.pattern
    matches = pat.findall(text)
    if not matches:
        return [f"{rule.path}: no version token matched {pat.pattern!r}"]
    if len(matches) > 1:
        return [f"{rule.path}: {len(matches)} version tokens matched (expected 1)"]
    correct = rule.replacement(version)
    if pat.search(text).group(0) == correct:
        return []
    if check:
        return [f"{rule.path}: found v{matches[0]}, expected {version}"]
    path.write_text(pat.sub(correct, text, count=1))
    print(f"  updated {rule.path} -> {version}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit 1 without writing (for CI)",
    )
    args = parser.parse_args()

    version = read_version()
    drift: List[str] = []
    for rule in RULES:
        drift.extend(apply_rule(rule, version, check=args.check))

    if drift:
        header = "Marketing version drift" if args.check else "Could not sync"
        print(f"{header} (pyproject = {version}):", file=sys.stderr)
        for msg in drift:
            print(f"  - {msg}", file=sys.stderr)
        if args.check:
            print(
                "\nRun `python scripts/sync_marketing_version.py` to fix, then "
                "re-render the OG PNG with `python scripts/render_og_image.py`.",
                file=sys.stderr,
            )
        return 1

    if not args.check:
        print(f"Marketing surfaces in sync at v{version}.")
        print("Reminder: re-render the OG PNG with `python scripts/render_og_image.py`.")
    else:
        print(f"Marketing surfaces in sync at v{version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
