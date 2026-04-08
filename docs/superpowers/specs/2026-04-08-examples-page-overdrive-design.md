# Examples Page Overdrive — Design Spec

**Date**: 2026-04-08
**Target file**: `landing/examples/index.html`
**Driver**: Bring the `/examples/` surface into the same visual language as the redesigned landing page so clicking "Examples" in the site nav does not break the "this page is still running" feeling established by `2026-04-07-landing-overdrive-fused-design.md`.

## Why this exists

The landing page now ships a full **execution-pointer system** — shared design tokens (`--exec-color`, `--exec-glow`, `--exec-pulse-dur`, `--exec-ease-soft`, etc.) and a family of atoms (`.exec-dot`, `.exec-caret`, `.exec-scan`, `@keyframes exec-pulse / exec-blink / exec-scan-sweep / exec-stamp`) defined at `landing/index.html:278-311` and `landing/index.html:2545-2624`. Six landing sections compose those atoms into their own hero moments. The examples page shares none of them. It is a different visual house, not a different room of the same house.

The rhetorical goal of `/examples/` is different from the landing page. Landing wants to *convince*; examples wants to *equip* (help a developer find a file and copy it). Any motion we add here must serve the equipping goal. The chosen direction — **Terminal Gallery (B) plus a category rail stolen from Runtime Catalog (C)** — does exactly that: a developer reads `ls -la` output faster than pill stacks, and a proportional-width category bar shows the shape of the catalog at a glance.

## Scope and files touched

**In scope**
- `landing/examples/index.html` — single-file redesign. Inline `<style>`, markup, and `<script>` blocks all change. The inlined `SRC` object of 88 example source files is unchanged.

**Out of scope**
- `landing/index.html` — already redesigned in `2026-04-07-landing-overdrive-fused-design.md`.
- `landing/builder/index.html`, `landing/simulations/index.html`, `landing/eval-report-preview.html`, `landing/trace-preview.html` — each deserves its own overdrive pass; not this PR.
- `landing/examples.json` — the underlying catalog data format is unchanged. Only the rendered HTML changes.
- `docs/` MkDocs site — untouched.
- `examples/*.py` source files — untouched. The 88 Python files are not modified in any way.
- No new images, fonts, or external assets.

**PR shape.** Single PR, atomic commits per design section so `git log -p` stays readable.

## Shared-atoms approach — duplicate inline, do not extract

There are two ways to share atoms between `landing/index.html` and `landing/examples/index.html`:

1. **Duplicate the atoms** into the examples file's inline `<style>` block.
2. **Extract to `landing/overdrive.css`** and `<link>` it from both pages.

**This spec uses option 1**: duplicate the atoms inline. Rationale:

- Matches the architectural choice the landing page already made (single self-contained HTML file).
- Adds no network request, preserves LCP.
- Drift risk is acceptable because the atoms are genuinely atomic — color tokens, durations, easings, `@keyframes` — no business logic.
- Revisit extraction later if and when `/builder/`, `/simulations/`, and other surfaces get the same treatment.

**What to duplicate** (verbatim from `landing/index.html`):

Tokens (from lines 298-311):
```
--exec-color, --exec-glow, --exec-glow-soft,
--exec-pulse-dur, --exec-step-dur,
--exec-ease-step, --exec-ease-soft,
--exec-blink-dur
```

Atoms (from lines 2553-2624):
```
.exec-dot, .exec-dot--lg, .exec-dot--sm
.exec-caret, .exec-caret--thin
.exec-scan, .exec-scan.in-view::after
@keyframes exec-pulse
@keyframes exec-blink
@keyframes exec-scan-sweep
@keyframes exec-stamp
```

The examples page keeps its existing short-class-prefix convention (`.ec`, `.eh`, `.eb`, `.ep`, etc.) for the gallery-specific machinery. New classes introduced in this redesign use the `ex-` prefix (`ex-term`, `ex-prompt`, `ex-rail`, `ex-row`) to avoid collisions.

## Section specifications

Each section below describes **what it replaces**, the **new treatment**, its **hero moment** (if any), **mobile fallback**, and **reduced-motion fallback**.

### §1 — Terminal-session header

**Replaces**: the `.ph` block at `landing/examples/index.html:58` containing `<h1>88 Example Scripts</h1>` and the descriptive paragraph. The existing nav strip (`.nl`, `.nr`) is kept but adopts the landing page's blurred-backdrop treatment (`background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(12px)`) — which it already has, so this is a no-op in practice.

**New markup** (conceptual):

```html
<header class="ex-term">
  <div class="ex-term__bar">
    <span class="ex-term__dot ex-term__dot--r"></span>
    <span class="ex-term__dot ex-term__dot--y"></span>
    <span class="ex-term__dot ex-term__dot--g"></span>
    <span class="ex-term__name">~/selectools/examples</span>
    <span class="ex-term__shell">zsh</span>
  </div>
  <div class="ex-term__body">
    <div class="ex-prompt" aria-hidden="true">
      <span class="ex-prompt__user">selectools</span>
      <span class="ex-prompt__at">@</span>
      <span class="ex-prompt__host">examples.dev</span>
      <span class="ex-prompt__colon">:</span>
      <span class="ex-prompt__path">~/selectools/examples</span>
      <span class="ex-prompt__glyph">$</span>
      <span class="ex-prompt__cmd">ls examples/<span class="ex-prompt__flags" id="ex-flags"></span><span class="ex-prompt__grep" id="ex-grep"></span></span>
      <span class="exec-caret"></span>
    </div>
    <h1 class="sr-only">Selectools examples — 88 runnable Python scripts</h1>
    <p class="ex-subtitle">88 runnable scripts covering agents, RAG, multi-agent graphs, evals, streaming, and guardrails. 34 run without an API key.</p>
  </div>
</header>
```

**Visual style**:
- `.ex-term` — matches the landing page's `.what-term`: `background: #0b1220; border: 1px solid var(--bd); border-radius: 14px; box-shadow: 0 20px 60px -28px rgba(0,0,0,0.55), 0 0 0 1px rgba(34,211,238,0.05);`
- `.ex-term__bar` — 44px tall, red/yellow/green dots, path label in JetBrains Mono 12px dim, trailing `zsh` shell label.
- `.ex-term__body` — 22px padding, font-family JetBrains Mono 13px for the prompt, Plus Jakarta Sans 14px for the subtitle.
- `.ex-prompt__user` cyan, `.ex-prompt__at` dim, `.ex-prompt__host` cyan, `.ex-prompt__path` amber, `.ex-prompt__glyph` green, `.ex-prompt__cmd` bright.

**Live behavior**:
- `#ex-grep` mirrors the search input (`<input class="si">`). Typing `rag` rewrites the prompt suffix to ` | grep -i rag`. Empty search → the grep pipe is not rendered at all (no dangling `|`).
- `#ex-flags` mirrors the active category. Clicking a category rail segment for `rag` rewrites the prompt suffix to ` --tags rag`. The "all" state shows no flag.
- Both are updated by a single function `syncPrompt()` called from the search input's `oninput` handler and the category rail's click handler.
- The `.exec-caret` blinks continuously via the shared `@keyframes exec-blink`.

**Hero moment**. On **page load** (not viewport entry — the header is above the fold), a `typeLine()` routine types `ls examples/` one character at a time into `.ex-prompt__cmd` at ~35ms/char. Total type-on duration ~420ms. After the base command is typed, the caret settles, and `syncPrompt()` becomes the only thing that modifies the prompt from then on.

**Accessibility**. The entire `.ex-prompt` block is `aria-hidden="true"` because it is decorative and its content is a reflection of the search state. A visually-hidden `<h1 class="sr-only">` adjacent to it carries the real page title for screen readers. The `.ex-subtitle` is a normal `<p>` readable by everybody.

**Mobile fallback** (max-width: 640px): `.ex-prompt__user`, `.ex-prompt__at`, `.ex-prompt__host`, `.ex-prompt__colon`, `.ex-prompt__path` are all hidden. Only the green `$` glyph + the command (`ls examples/` + live flags/grep) + caret are shown. This fits a 360px viewport without wrapping on the first line; if flags/grep push past the edge, they wrap to a second line inside the terminal body.

**Reduced-motion fallback**: type-on animation is skipped. The command renders fully typed on load. The caret is rendered but `.exec-caret { animation: none; opacity: 1; }` inside the reduced-motion media query, so it is a static block, not blinking.

### §2 — Category rail

**Replaces**: the existing chip row `.cr` at `landing/examples/index.html:61-78` (18 `<button class="cb">` buttons). The chips are removed entirely and replaced with a single proportional-width bar.

**New markup**:

```html
<div class="ex-rail" id="ex-rail" role="tablist" aria-label="Filter examples by category">
  <button class="ex-rail__seg ex-rail__seg--all on" data-cat="all" role="tab" aria-selected="true">
    <span class="ex-rail__name">all</span>
    <span class="ex-rail__count">88</span>
  </button>
  <button class="ex-rail__seg" data-cat="agent" role="tab" style="--seg-weight: 21">
    <span class="ex-rail__name">agent</span>
    <span class="ex-rail__count">21</span>
  </button>
  <!-- ... 16 more segments ... -->
</div>
```

**Layout**. CSS flex with each segment's count as `flex-grow`: `.ex-rail { display: flex; gap: 2px; }` and each segment has `flex: var(--seg-weight) 1 0;` where `--seg-weight` is its count. The `all` segment is fixed-width at 72px (`flex: 0 0 72px`) as an anchor on the left.

**Visual style**:
- `.ex-rail` — 40px tall, `border-radius: 8px`, `overflow: hidden`, `border: 1px solid var(--bd)`, `background: rgba(30, 41, 59, 0.4)`.
- `.ex-rail__seg` — full-height button, no individual borders, monospace font, lowercase category name, cyan count trailing. Inactive state: `color: var(--dm); background: transparent;`. Hover: `background: rgba(34, 211, 238, 0.08);` and a 2px cyan underline grows from left via `::after` transform.
- Active state (`.on`): `background: rgba(34, 211, 238, 0.12); color: var(--cy); box-shadow: inset 0 -2px 0 var(--exec-color);`.
- `.ex-rail__name` — primary text, 12px. `.ex-rail__count` — trailing 11px cyan number: `agent 21`.
- `min-width: 56px` on each segment to keep count-1 categories legible.

**Hero moment**. On **viewport entry** (first intersection of `.ex-rail` with the viewport), an `IntersectionObserver` adds class `.in-view` to the rail, triggering a left-to-right stamp sweep: each segment runs `@keyframes exec-stamp` with `animation-delay: calc(var(--seg-index) * 80ms)`. Each segment is generated with `style="--seg-weight: N; --seg-index: I"` where `N` is its category count and `I` is its zero-based position along the rail. Total sweep duration ~1.4s for 18 segments. The observer disconnects after the first trigger (one-shot).

**Interaction**. Clicking a segment:
1. Removes `.on` from the previous active segment, adds it to the clicked one.
2. Triggers a one-shot `exec-stamp` on the clicked segment (re-run via `animation: none;` then `animation: exec-stamp 0.6s var(--exec-ease-soft);` on next frame).
3. Calls the existing `flt()` function (at `landing/examples/index.html:174`) which filters the `.ec` cards.
4. Calls `syncPrompt()` to update the prompt flags in §1.

The existing `.cb` chip row event listener loop at `landing/examples/index.html:175` is removed. The new rail provides the same `data-cat` dispatch.

**Mobile fallback** (max-width: 640px): rail becomes horizontally scrollable — `overflow-x: auto; -webkit-overflow-scrolling: touch; scroll-snap-type: x mandatory;`. Each segment gets `scroll-snap-align: start; min-width: 80px; flex: 0 0 auto;`. Segments no longer size proportionally on mobile — they become uniform-width cards that swipe past the viewport edge. Hero sweep still runs but off-screen segments animate off-screen too; acceptable because the visible segments still animate.

**Reduced-motion fallback**: no stamp sweep on viewport entry. Rail renders fully in its rest state. Hover/click effects work but with `transition: background-color 0.01s;`.

### §3 — Filter row and search input

**Replaces**: the current `.ct` container with its search box, chip row (removed in §2), and result counter. The sticky behavior is preserved.

**New structure**:

```html
<div class="ex-ct">
  <div class="ex-search">
    <span class="ex-search__glyph">⌕</span>
    <input class="ex-search__input si" type="text"
           placeholder="search by name or keyword…"
           oninput="flt(); syncPrompt();" id="si" autocomplete="off" />
    <kbd class="ex-search__kbd">/</kbd>
  </div>
  <!-- ex-rail from §2 sits here -->
  <div class="ex-count rc" id="rc"># 88 files match</div>
</div>
```

**Changes**:
- Search input gains a leading `⌕` glyph and a trailing `kbd` hint (`/`) indicating the keyboard shortcut. A top-level `keydown` listener focuses the search input when `/` is pressed (unless a form element is already focused).
- Result counter text format changes from `88 examples` → `# 88 files match`. The `#` prefix ties it visually to the monospace comment aesthetic in the terminal header. The counter uses JetBrains Mono 11px dim color.
- The category rail (§2) sits between the search input and the count.
- The container is still sticky (`position: sticky; top: 52px; z-index: 40;`) so it tracks below the main nav.

**Mobile fallback**: unchanged from current behavior. Sticky works; count sits beneath the rail.

**Reduced-motion fallback**: no animations in this section, nothing to fall back.

### §4 — Card list as `ls -la` output

**Replaces**: the `.eh` header row inside each `.ec` card (the currently-rendered accordion row that shows `01`, title, description, key badge, line count, chevron). The card container (`.ec`) and body (`.eb`) stay. Only the header row changes.

**New row structure** — a CSS Grid row with 7 columns:

```html
<div class="ex-row eh" onclick="toggle(this)">
  <span class="ex-row__num">01</span>
  <span class="ex-row__perm">-rw-r--r--</span>
  <span class="ex-row__size">46L</span>
  <span class="ex-row__key ex-row__key--free">no-key</span>
  <span class="ex-row__file">hello_world.py</span>
  <span class="ex-row__desc">Your first selectools agent</span>
  <span class="ex-row__chev ev">▾</span>
</div>
```

**Grid definition**:
```css
.ex-row {
  display: grid;
  grid-template-columns: 32px 112px 54px 72px minmax(180px, 1.5fr) minmax(0, 3fr) 20px;
  align-items: center;
  gap: 16px;
  padding: 12px 18px;
  font-family: var(--mono);
  font-size: 12px;
  cursor: pointer;
  user-select: none;
}
```

**Column semantics**:

| Col | Class | Width | Content | Color |
|-----|-------|-------|---------|-------|
| 1 | `ex-row__num` | 32px | `01`–`88` zero-padded | cyan (`var(--cy)`) |
| 2 | `ex-row__perm` | 112px | literal `-rw-r--r--` | dim (`var(--ft)`) |
| 3 | `ex-row__size` | 54px right-align | `46L`, `354L`, etc. | dim |
| 4 | `ex-row__key` | 72px | `no-key` or `api-key` | green or amber |
| 5 | `ex-row__file` | flex 1.5 | filename *without* the `NN_` prefix (e.g. `hello_world.py`, not `01_hello_world.py`) | bright cyan |
| 6 | `ex-row__desc` | flex 3 | existing description, single-line ellipsis | bright text |
| 7 | `ex-row__chev` | 20px | `▾` rotates 180° when open | dim |

**Why strip the `NN_` prefix from the filename column.** Column 1 already carries the number. Showing `01_hello_world.py` in column 5 duplicates information. Stripping to `hello_world.py` lets column 5 stay readable at narrower viewports. The GitHub link (`<a href="...examples/01_hello_world.py">`) inside the expanded body still uses the full filename — the strip is visual-only.

**Hover**: `background: rgba(34, 211, 238, 0.04); border-left: 2px solid var(--cy);` (the border-left pushes the row rightward by 2px, which is intentional — it makes the hovered row feel "picked up").

**Open state** (`.ec.op .ex-row`): `background: rgba(34, 211, 238, 0.06);` and the chevron rotates via `transform: rotate(180deg)` (same as current `.ec.op .ev`).

**Per-row entry animation** — a subtle one, pure CSS, no JS observer. Rows with index 0–29 (the ~30 rows visible on first paint at a 1440×900 viewport) are generated with an extra class `.ex-row--enter` and `style="--row-index: N"`. The `.ex-row--enter` class carries `animation: ex-row-in 0.35s var(--exec-ease-soft) both; animation-delay: calc(var(--row-index) * 14ms)`. Rows with index ≥ 30 are generated without the `--enter` class and without `--row-index` — they render directly in their final visual state (opacity 1, no transform) and never animate, even when scrolled into view. This prevents a cascade-of-88 effect that would feel slow.

```css
@keyframes ex-row-in {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: none; }
}
```

**Mobile fallback** (max-width: 640px): grid collapses to 2 visual lines using `grid-template-columns: 32px 1fr 20px; grid-template-rows: auto auto;` with explicit cell placement:

```css
@media (max-width: 640px) {
  .ex-row { grid-template-columns: 32px 1fr 20px; gap: 8px 12px; }
  .ex-row__num  { grid-column: 1; grid-row: 1 / 3; align-self: start; }
  .ex-row__perm { display: none; }
  .ex-row__file { grid-column: 2; grid-row: 1; }
  .ex-row__chev { grid-column: 3; grid-row: 1 / 3; align-self: start; }
  .ex-row__size { grid-column: 2; grid-row: 2; display: inline; margin-right: 8px; color: var(--ft); }
  .ex-row__key  { grid-column: 2; grid-row: 2; display: inline; margin-right: 8px; }
  .ex-row__desc { grid-column: 2; grid-row: 2; display: inline; color: var(--dm); }
}
```

Line 2 reads as: `46L no-key Your first selectools agent` — compact but all critical info is there. Permissions column is hidden (pure decoration, not worth the space).

**Reduced-motion fallback**: `.ex-row` animation is disabled. All rows render in their final state immediately.

### §5 — Card expansion as `$ cat` output

**Replaces**: the `.eb` body interior. The body container still toggles `display: none` → `display: ''` via the existing `toggle()` function at `landing/examples/index.html:177`.

**New first child of `.eb`** — a one-line terminal prefix inserted ABOVE the existing action buttons (`.ea`) and code pane (`.ep`):

```html
<div class="eb" style="display:none">
  <div class="ex-cat-prefix"><span class="ex-cat-prefix__glyph">$</span> cat examples/01_hello_world.py</div>
  <div class="eg">...tag links...</div>
  <div class="ea">...Copy / GitHub / Docs buttons...</div>
  <pre class="ep"></pre>
</div>
```

**Style**:
```css
.ex-cat-prefix {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--ft);
  padding: 0 0 10px;
  user-select: text;
}
.ex-cat-prefix__glyph { color: var(--gn); margin-right: 6px; }
```

**Interaction**. The prefix is generated at render time from `data-file` on the `.ec` element — no runtime DOM manipulation on toggle. It exists in the DOM whether the card is open or closed; it is hidden along with the rest of `.eb` via the existing `display: none` toggle.

**Card open/close transition**. The current implementation uses an instant display toggle. This design keeps that (no `max-height` tween) because:
1. Tweening `max-height` on 88 variable-height cards is fussy.
2. An instant reveal + smooth chevron rotation feels snappy, not janky.
3. The existing code at line 177 is preserved — minimal risk.

The chevron rotation gets one upgrade: `transition: transform 0.22s var(--exec-ease-soft);` instead of the current `.2s` unspecified easing.

**Mobile fallback**: unchanged. Prefix, buttons, and code pane all render as today, just with the prefix on top.

**Reduced-motion fallback**: chevron rotation transition shortened to `0.01s`. Everything else unchanged.

### §6 — Nav bar (minimal change)

The existing `<nav>` at `landing/examples/index.html:54-57` is kept with one edit: the left brand gains a permanent pulsing dot to match the landing page's wordmark variant 1:

```html
<a href="../" class="nl"><span class="exec-dot"></span>&nbsp;selectools <span>examples</span></a>
```

The inline-block `.exec-dot` sits before the wordmark and pulses continuously via `@keyframes exec-pulse`. This is the single most visible cross-page coherence signal — a user clicking between `/` and `/examples/` sees the same dot pulsing in the same place.

**Mobile fallback**: unchanged. `.exec-dot` is 8px — fits on mobile.
**Reduced-motion fallback**: `.exec-dot` gets `animation: none; box-shadow: 0 0 6px var(--exec-glow);` — it stays glowing but does not pulse.

## Implementation order

Single PR, one commit per design section so the diff reads cleanly in `git log -p`. Commits in this order:

1. **Setup commit** — duplicate the execution-pointer atoms (tokens + `.exec-dot`, `.exec-caret`, `.exec-scan`, `@keyframes exec-pulse / exec-blink / exec-scan-sweep / exec-stamp`) into `landing/examples/index.html` inline `<style>`. Add the `sr-only` utility class. No visual change yet.
2. **§6 nav dot** — add `.exec-dot` to the nav brand. First visible proof of coherence, smallest diff. Verifies the atoms landed correctly.
3. **§1 terminal header** — replace `.ph` with `.ex-term` block. Implement `typeLine()` for page-load type-on and `syncPrompt()` stub (does nothing yet until §2 and §3 wire it up).
4. **§2 category rail** — replace `.cr` chip row with `.ex-rail` proportional bar. Implement `IntersectionObserver` for hero stamp sweep. Wire clicks to existing `flt()` and to `syncPrompt()`.
5. **§3 search row** — rewrap search input with `.ex-search` glyph/kbd chrome. Update counter format to `# N files match`. Hook up `/` keyboard shortcut.
6. **§4 card rows** — replace `.eh` internals across all 88 `.ec` cards with the 7-column `.ex-row` grid. Add mobile collapse media query. Add `ex-row-in` stagger for first ~30 rows.
7. **§5 card expansion** — add `.ex-cat-prefix` inside each `.eb` body (generated at render time from `data-file`). Update chevron transition.
8. **Final cleanup commit** — reduced-motion audit (grep for every `@keyframes` and confirm each has a `prefers-reduced-motion` fallback), mobile audit via Playwright viewport emulation, cross-browser spot-check.

Each commit gets a Playwright visual check before moving on: load the page at desktop (1440×900) and mobile (375×812), confirm the section under test renders and animates correctly.

## Performance budget

- **Added CSS**: under 6KB unminified (atoms + §1–§6 new classes).
- **Added JS**: under 1.5KB unminified (one `IntersectionObserver`, one `typeLine()` routine, one `syncPrompt()` helper, one `/` keydown listener). No new dependencies. Vanilla.
- **No new fonts**: Plus Jakarta Sans + JetBrains Mono already loaded.
- **No new image assets**: everything CSS, SVG, or text.
- **Initial paint**: must not regress LCP. The 600KB payload from the inlined `SRC` object is unchanged.
- **Animation FPS target**: 60fps on a mid-range Android baseline (Galaxy S23).
- **IntersectionObserver cleanup**: the category rail observer disconnects after the first trigger. The card-row entry animation uses pure CSS `animation-delay` on first render, so no observer is needed and no cleanup required.

## Accessibility requirements

- **WCAG 2.1 AA** maintained. Every new color pairing checked against the `#0f172a` background for contrast.
- **`prefers-reduced-motion: reduce`**: every animation listed in this spec has a static fallback specified. Not optional.
- **Keyboard navigation**: search input focusable with `/`, category rail segments are `<button>` elements keyboard-reachable with visible focus ring. Card expand/collapse is keyboard-activatable (`.eh` gets `tabindex="0"` + `role="button"` + `aria-expanded` + `keydown` handler for Enter/Space).
- **Screen readers**: the `.ex-prompt` block is `aria-hidden="true"` (decorative). A visually-hidden `<h1 class="sr-only">` carries the semantic page title. The category rail has `role="tablist"` with `role="tab"` on each segment and `aria-selected` on the active one. Card rows use `aria-expanded` on the toggle button.
- **Touch targets**: 44×44px minimum on `pointer: coarse`. Category rail segments on mobile are at least 44px tall after the scroll-snap collapse.
- **Visible focus rings**: all interactive elements get a cyan 2px focus outline (`:focus-visible { outline: 2px solid var(--cy); outline-offset: 2px; }`).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Proportional-width category rail produces tiny segments for categories with count 1 | Minimum segment width 56px via `min-width` on `.ex-rail__seg`, slightly breaks strict proportionality but guarantees legibility |
| Terminal-session prompt wraps ugly on narrow viewports | Mobile media query hides `user@host:path` portion, showing only `$ ls examples/<flags>` |
| Search/filter mirroring into prompt causes layout thrash on each keystroke | `syncPrompt()` writes to `textContent` on two spans only, no layout-triggering CSS changes; each update is O(1) |
| Card-row entry animation causes 88 simultaneous paints | Only the first ~30 rows animate on entry; rows 31+ render immediately |
| Keyboard `/` shortcut conflicts with browser in-page search | Only fire when no input/textarea is focused; user can still use Cmd+F for browser search |
| `IntersectionObserver` leaks if page is kept open long-term | Rail observer is one-shot (disconnect after first trigger); no long-lived listeners |
| Card-row grid columns misalign if a description contains a very long unbroken token | `ex-row__desc` gets `overflow: hidden; text-overflow: ellipsis; white-space: nowrap;` |
| Removing the visible category chip row breaks muscle memory for users who were already using them | Category rail is visually louder and positioned in the same place — the affordance is not hidden, just restyled |
| `.exec-dot` in nav pulsing forever triggers distraction complaints | Respects `prefers-reduced-motion` (becomes static glow); the landing page already does this and is shipped, so the precedent is set |
| Examples page is statically generated from `landing/examples.json` — hand-edits get clobbered on next regen | Confirm generator location before implementation; if one exists, the redesign must land in the generator source, not the generated HTML |

## Out of scope

- No changes to the inlined `SRC` object of 88 example source files. The highlighter function `hl()` is preserved as-is.
- No changes to the existing `flt()` filter function beyond its call-site updates.
- No changes to the category taxonomy — the 18 existing categories stay.
- No new Python files, no changes to `examples/*.py`.
- No changes to `landing/examples.json` format.
- No new images, fonts, or external assets.
- No copy rewrites — only visual/structural treatment changes. Existing titles and descriptions are preserved.
- No changes to the site nav, the `/builder/` page, the `/simulations/` page, or the MkDocs site.

## Definition of done

- All six sections (§1–§6) rebuilt per spec.
- Nav `.exec-dot` visible and pulsing on both `/` and `/examples/`.
- Terminal header types `ls examples/` on page load; prompt live-reflects search + category.
- Category rail sweeps on viewport entry with staggered stamps; clicking a segment filters the list and rewrites the prompt flag.
- Card rows render as `ls -la` columns on desktop, collapse to 2-line layout on mobile.
- Card expansion shows `$ cat examples/NN_name.py` prefix above the source.
- `prefers-reduced-motion: reduce` audit confirms every `@keyframes` has a static fallback.
- Mobile audit at 360px confirms header, rail, and card rows all render without horizontal overflow.
- Keyboard navigation: `/` focuses search; Tab cycles through rail segments and card rows; Enter/Space on a row toggles expansion.
- Performance budget respected (CSS ≤ 6KB, JS ≤ 1.5KB unminified added).
- No regression: all 88 cards still load their source on expand, Copy button still copies, GitHub/Docs links still open correctly.
- PR opened with section-by-section commits readable in `git log -p`.
