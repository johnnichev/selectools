#!/usr/bin/env python3
"""Regenerate ``docs/llms-full.txt`` from the mkdocs nav.

``llms-full.txt`` is a single-file concatenation of every documentation page,
for AI-agent consumption. It was previously hand-maintained and drifted (stale
counts, ~15 missing module pages, an embedded version several releases old).
This script rebuilds it deterministically from ``mkdocs.yml`` so it can never
drift again — run it as part of the release doc sweep.

Usage::

    python scripts/build_llms_full.py          # writes docs/llms-full.txt
    python scripts/build_llms_full.py --check   # exit 1 if out of date

Pages are emitted in nav order. ``CHANGELOG.md`` is excluded (it is large and
already shipped verbatim as its own file); everything else referenced in the
nav is included.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT = DOCS_DIR / "llms-full.txt"

# Pages referenced in the nav that we deliberately do NOT inline.
EXCLUDE = {"CHANGELOG.md"}

_BANNER = "=" * 60
_MD_TOKEN = re.compile(r"([A-Za-z0-9_./-]+\.md)")


def _nav_pages() -> List[str]:
    """Extract .md paths from the mkdocs ``nav:`` block, in order, deduped.

    A plain line scan rather than a YAML parse: mkdocs.yml carries
    Python-specific tags elsewhere (pymdownx emoji generators) that defeat a
    safe YAML load, and the nav is a simple indented list. We read only the
    ``nav:`` block (from the ``nav:`` line until the next column-0 key) so no
    stray .md references from other sections are picked up.
    """
    lines = MKDOCS.read_text(encoding="utf-8").splitlines()
    pages: List[str] = []
    in_nav = False
    for line in lines:
        if not in_nav:
            if line.rstrip() == "nav:":
                in_nav = True
            continue
        # End of the nav block: the next top-level (column-0) key.
        if line and not line[0].isspace():
            break
        match = _MD_TOKEN.search(line)
        if match:
            rel = match.group(1)
            if rel not in pages and rel not in EXCLUDE:
                pages.append(rel)
    return pages


def build() -> str:
    pages = _nav_pages()
    included: List[str] = []
    blocks: List[str] = []
    for rel in pages:
        path = DOCS_DIR / rel
        if not path.is_file():
            print(f"warning: nav references missing doc, skipping: docs/{rel}", file=sys.stderr)
            continue
        content = path.read_text(encoding="utf-8").rstrip()
        blocks.append(f"\n\n{_BANNER}\n\n## FILE: docs/{rel}\n\n{_BANNER}\n\n\n{content}\n")
        included.append(rel)

    header = (
        "# selectools — Full Documentation\n\n"
        "> This file concatenates all selectools documentation pages for AI agent consumption.\n\n"
        f"> {len(included)} pages included. "
        "Generated from docs/ source files in mkdocs nav order "
        "(run `python scripts/build_llms_full.py`).\n"
    )
    # Exactly one trailing newline, to match the repo's end-of-file-fixer hook
    # (so --check stays stable after a commit).
    return (header + "".join(blocks)).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if llms-full.txt is stale")
    args = parser.parse_args()

    rendered = build()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print("docs/llms-full.txt is out of date — run: python scripts/build_llms_full.py")
            return 1
        print("docs/llms-full.txt is up to date.")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    n = rendered.count("## FILE:")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({n} pages, {len(rendered):,} chars).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
