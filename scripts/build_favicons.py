#!/usr/bin/env python3
"""Render the selectools favicon to PNG (apple-touch-icon) and ICO (legacy fallback).

The source-of-truth design lives in landing/favicon.svg as a 32x32 viewBox:
  - Rounded square background, fill #0f172a, corner radius 6
  - Cyan (#22d3ee) brackets [ ] flanking a center dot
  - Dot at (16, 16) with r=3

Rather than rendering the SVG (which depends on JetBrains Mono being installed),
we redraw the design from primitives so it's reproducible in any environment with
just Pillow. Output goes back into landing/ where the deploy workflow picks it up.

Run: python3 scripts/build_favicons.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
LANDING = REPO_ROOT / "landing"

BG = (15, 23, 42, 255)  # #0f172a
FG = (34, 211, 238, 255)  # #22d3ee


def draw_master(size: int) -> Image.Image:
    """Draw the favicon at an arbitrary square size, scaled from the 32x32 design."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Scale factor from the 32-unit SVG viewBox to the target pixel size.
    s = size / 32.0

    # Background: rounded square. SVG has rx=6 on a 32x32 — scale proportionally.
    radius = round(6 * s)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=BG)

    # Bracket geometry (in 32-unit design space):
    #   bracket "stem" is 1.6 wide, 14 tall, centered vertically (y=9..23)
    #   bracket "arms" stick out 3 units to the right (left bracket) or left (right)
    stem_w = 1.6
    arm_h = 1.6
    arm_w = 3.0
    bracket_top = 9
    bracket_bottom = 23

    def bracket(x_stem: float, arm_dir: int) -> None:
        # arm_dir: +1 = arms extend right (left bracket), -1 = arms extend left (right bracket)
        # Vertical stem
        draw.rectangle(
            (
                round(x_stem * s),
                round(bracket_top * s),
                round((x_stem + stem_w) * s),
                round(bracket_bottom * s),
            ),
            fill=FG,
        )
        # Top arm
        arm_x0 = x_stem if arm_dir > 0 else x_stem + stem_w - arm_w
        arm_x1 = arm_x0 + arm_w
        draw.rectangle(
            (
                round(arm_x0 * s),
                round(bracket_top * s),
                round(arm_x1 * s),
                round((bracket_top + arm_h) * s),
            ),
            fill=FG,
        )
        # Bottom arm
        draw.rectangle(
            (
                round(arm_x0 * s),
                round((bracket_bottom - arm_h) * s),
                round(arm_x1 * s),
                round(bracket_bottom * s),
            ),
            fill=FG,
        )

    # Left bracket (arms extend right)
    bracket(x_stem=7.0, arm_dir=+1)
    # Right bracket (arms extend left)
    bracket(x_stem=23.4, arm_dir=-1)

    # Center dot — design has r=3 in 32-unit space
    cx, cy, r = 16, 16, 3
    draw.ellipse(
        (
            round((cx - r) * s),
            round((cy - r) * s),
            round((cx + r) * s),
            round((cy + r) * s),
        ),
        fill=FG,
    )

    return img


def main() -> None:
    # Draw at 512 once, then downsample for each target size — antialiasing wins.
    master = draw_master(512)

    # apple-touch-icon: iOS Home Screen, Safari pinned tab. 180x180 PNG is the
    # canonical size; everything else is a downscale from there in practice.
    apple = master.resize((180, 180), Image.Resampling.LANCZOS)
    apple_path = LANDING / "apple-touch-icon.png"
    apple.save(apple_path, "PNG", optimize=True)
    print(f"wrote {apple_path.relative_to(REPO_ROOT)} ({apple_path.stat().st_size} bytes)")

    # favicon.ico: legacy fallback for very old browsers + Windows tile cache.
    # Multi-size container: 16, 32, 48 — Pillow generates each from the master.
    ico_path = LANDING / "favicon.ico"
    master.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)],
    )
    print(f"wrote {ico_path.relative_to(REPO_ROOT)} ({ico_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
