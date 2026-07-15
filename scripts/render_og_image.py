#!/usr/bin/env python3
"""Rasterize the landing OG social card SVG to PNG at 1200x630.

The served social card (``landing/assets/og-image.png``) is a raster of
``landing/assets/og-image.svg``. Run this after the SVG changes (e.g. after
``sync_marketing_version.py`` bumps the version footer) so the PNG link
preview matches.

Headless Chromium is used deliberately (not librsvg/cairosvg) so the font
fallback matches the design; ad-hoc rasterizers substitute fonts and shift
the layout. Requires the ``playwright`` package + its chromium browser:

    uv run --with playwright python scripts/render_og_image.py
    # first time: uv run --with playwright playwright install chromium
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SVG = REPO_ROOT / "landing/assets/og-image.svg"
PNG = REPO_ROOT / "landing/assets/og-image.png"
WIDTH, HEIGHT = 1200, 630


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright not installed. Run:\n"
            "  uv run --with playwright python scripts/render_og_image.py\n"
            "  (first time also: ... playwright install chromium)"
        )
        return 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.goto(SVG.as_uri())
        page.wait_for_timeout(500)
        page.screenshot(path=str(PNG))
        browser.close()
    print(f"Rendered {PNG.relative_to(REPO_ROOT)} ({WIDTH}x{HEIGHT}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
