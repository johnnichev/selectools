# Selectools Design System

The concrete tokens, components, animation language, and rules that make up the selectools visual language.

This is the **what**. For the **why** — brand personality, design principles, voice and tone — see `.impeccable.md` at the project root.

**Source of truth**: `landing/index.html` (inline `<style>` block + JS observers). This document mirrors that source. When the source changes, this document must be updated to match.

---

## 1. Design tokens

All tokens live on `:root` in `landing/index.html` and are referenced via `var(--name)`. Never hardcode hex values in component CSS.

### 1.1 Colors

#### Surface palette

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#0f172a` | Page background. The slate-950 anchor everything else sits on. |
| `--surface` | `#1e293b` | Cards, code frames, default panel background. |
| `--surface-2` | `#273548` | Slightly elevated surface (rarely used). |
| `--border` | `#334155` | Default card/panel border. 1px solid. |

#### Text palette

| Token | Hex | Use |
|---|---|---|
| `--text` | `#e2e8f0` | Primary body text. Headings on dark surface. |
| `--text-dim` | `#94a3b8` | Secondary text. Section descriptions. Card body copy. |
| `--text-faint` | `#64748b` | Tertiary text. Labels, captions, comment color in code. |

#### Accent palette

| Token | Hex | Use |
|---|---|---|
| `--cyan` | `#22d3ee` | **Primary accent.** The signature brand color. Use for: links, active states, focus rings, the execution pointer, code keywords (`from`, `import`), interactive feedback. |
| `--blue` | `#3b82f6` | CTAs (secondary buttons). LLM-judge evaluator tags. |
| `--green` | `#22c55e` | Success states, "passed" badges, costs (when low is good), version-stack lit rungs, stability "@stable". |
| `--amber` | `#f59e0b` | Warning, "running" states, latency bars, stability "@beta". |
| `--red` | `#ef4444` | Errors, mac terminal close dot, broken-state vignettes, stability "@deprecated". |
| `--purple` | `#a855f7` | Code keywords (`from`, `class`, `def` in code blocks). |

**Rule**: One accent per component. Never mix two accents (cyan + green) on the same element. The exception is the eval report, which uses cyan / amber / green to differentiate metric types — that's intentional and earned.

#### Glow tokens (used by the execution-pointer system)

| Token | Value |
|---|---|
| `--exec-color` | `#22d3ee` (alias of `--cyan`, named for intent) |
| `--exec-glow` | `rgba(34, 211, 238, 0.55)` |
| `--exec-glow-soft` | `rgba(34, 211, 238, 0.18)` |

### 1.2 Typography

| Token | Stack |
|---|---|
| `--font-ui` | `'Plus Jakarta Sans', system-ui, -apple-system, sans-serif` |
| `--font-mono` | `'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace` |

**Loaded weights**: Plus Jakarta Sans 400 / 500 / 600 / 700 / 800; JetBrains Mono 400 / 500.

#### Type scale (de facto, extracted from existing components)

| Use | Size | Weight | Family |
|---|---|---|---|
| Hero h1 | 64-80px | 800 | UI |
| Section h2 | 32-44px (clamp) | 700-800 | UI |
| Card h3 | 18-26px | 700 | UI |
| Body | 15px | 400 | UI |
| Card body | 13.5px | 400 | UI |
| Caption | 12.5px | 400 | UI |
| Section label | 11px (uppercase, letter-spacing 0.12em) | 500 | Mono |
| Code body | 13px | 400 | Mono |
| Mono labels | 9-11px (uppercase) | 400-500 | Mono |

**Headings**: Tight tracking (`letter-spacing: -0.02em` to `-0.03em`). **Body**: Generous line-height (`1.7`). **Mono labels**: Wide tracking (`0.12em` to `0.18em`), uppercase.

### 1.3 Spacing

There is no formal spacing scale — values are chosen per-component for visual rhythm. Common values that recur:

| px | Use |
|---|---|
| 4 | Tight inline gaps. |
| 8 | Compact card padding edges. |
| 10-14 | Default internal gaps. |
| 16 | Mobile section padding. |
| 18-22 | Default panel padding. |
| 24 | Code body padding. |
| 28 | Default desktop card padding. |
| 32-44 | Section heading bottom margin. |
| 64-100 | Section vertical padding. |

### 1.4 Radii

| px | Use |
|---|---|
| 6 | Inline code, small chips. |
| 8-9 | Buttons, version stack rungs. |
| 10-12 | Default card radius. |
| 14 | Big panels (terminals, bento cells, eval report). |
| 999 | Pills (status badges, capability chips, eval tags). |

### 1.5 Easing & duration

| Token | Value | Use |
|---|---|---|
| `--ease` | `cubic-bezier(0.16, 1, 0.3, 1)` | Default. Almost-elastic ease-out. Use for hover, reveal, layout transitions. |
| `--exec-ease-soft` | `cubic-bezier(0.16, 1, 0.3, 1)` | Alias of `--ease` for the execution-pointer system. |
| `--exec-ease-step` | `cubic-bezier(0.4, 0, 0.2, 1)` | Snappier. Use for stamps, scan lines, node-to-node travel. |
| `--exec-pulse-dur` | `1.6s` | Default pulse loop. |
| `--exec-step-dur` | `0.55s` | Node-to-node travel. |
| `--exec-blink-dur` | `1.05s` | Terminal caret blink cadence. |

**Rule**: Never use `cubic-bezier` literals in component CSS. Reference the tokens.

### 1.6 Layout

| Token | Value |
|---|---|
| `--max-w` | `1120px` | Max content width. Wrap everything in `.wrap { max-width: var(--max-w) }`. |
| `--grain` | inline SVG turbulence | Subtle texture overlay for body background (opacity 0.018). |

### 1.7 Breakpoints

| Width | Devices | What changes |
|---|---|---|
| `≥1280px` | Desktop | Default. 6-col bento, side-by-side stages, full nav. |
| `≤1100px` | Small desktop | Bento drops to 4-col. |
| `≤1024px` | Tablet | Nav links hide, hamburger shows, multi-col layouts collapse to 1-col. |
| `≤880px` | Large mobile | Section stages collapse to single column. |
| `≤768px` | Tablet portrait | Builder iframe fallback shows. |
| `≤700px` | Mobile | Bento goes single-column with `!important` overrides. Drag-to-compare stacks vertically. |
| `≤640px` | Small mobile | Tighter padding, smaller code body font. |
| `≤480px` | Tiny mobile | Hero scale adjusts. |

**Rule**: Use content-driven breakpoints. The 700/880/1100 values exist because that's where specific sections break, not because they're standard widths.

### 1.8 Touch targets

`@media (pointer: coarse)`: any touch device, regardless of viewport width. Used to defeat Chrome's "Desktop Site" mode which lies about the viewport. All interactive elements must be ≥44×44px on coarse pointer.

---

## 2. Component inventory

All components live in `landing/index.html`. Search for the class name to find the styles.

### 2.1 Layout shells

| Class | Purpose |
|---|---|
| `.wrap` | Max-width container. Use as the first child of every `<section>`. |
| `.section` | Vertical-padded section wrapper. |
| `.section-label` | Mono uppercase label that sits above an h2 (e.g. "WHAT IS SELECTOOLS?"). |
| `.section-desc` | Body paragraph below the h2. |

### 2.2 Code frame (mac-style terminal)

The reusable code window with traffic-light dots and a filename label.

```html
<div class="code-frame code-frame-accent">
  <div class="code-bar">
    <div class="code-dot" style="background:rgba(239,68,68,0.8)"></div>
    <div class="code-dot" style="background:rgba(250,204,21,0.8)"></div>
    <div class="code-dot" style="background:rgba(34,197,94,0.8)"></div>
    <span class="code-bar-label">filename.py</span>
  </div>
  <div class="code-body"><span class="kw">from</span> selectools <span class="kw">import</span> Agent</div>
</div>
```

| Class | Purpose |
|---|---|
| `.code-frame` | The outer frame with border, radius, surface bg. |
| `.code-frame-accent` | Cyan-bordered variant. Use for the "good" code in comparisons. |
| `.code-bar` | Top bar with dots + filename. |
| `.code-dot` | Mac traffic-light dot (style with inline `background`). |
| `.code-bar-label` | The filename label. Mono, 11px, faint color. |
| `.code-body` | The code container. Has `white-space: pre` and `mask-image` right-edge fade. |
| `.code-dim` | Modifier for "the bad code" (opacity 0.45). |

**Syntax token classes** (apply to `<span>` inside `.code-body`):
| Class | Color | For |
|---|---|---|
| `.kw` | `#c084fc` (purple) | `from`, `import`, `def`, `class`, `return`, `@`. |
| `.cls` | `#7dd3fc` (cyan-300) | Class names, type annotations. |
| `.fn` | `#fbbf24` (amber) | Function names, callable invocations. |
| `.str` | `#86efac` (green) | String literals. |
| `.cmt` | `var(--text-faint)` | Comments. |
| `.num` / `.bool` | `#fb923c` (orange) | Numeric literals, booleans. |

### 2.3 Buttons

| Class | Purpose |
|---|---|
| `.btn` | Base button. |
| `.btn-primary` | Cyan-bg primary CTA. |
| `.btn-ghost` | Outlined secondary. |
| `.btn-link` | Inline cyan text link. |

All buttons: `transform: scale(0.97)` on `:active`, `transition` via `--ease`. Touch hover guards via `@media (hover: none)`.

### 2.4 Cards & bento

| Class | Purpose |
|---|---|
| `.card` | Default rounded panel with border + grain overlay. |
| `.bento` | 6-col CSS Grid container. Auto-flow dense. |
| `.bento__cell` | Default cell (`grid-column: span 2`). |
| `.bento__cell--hero` | 3-col × 2-row hero cell. |
| `.bento__cell--wide` | 3-col × 1-row wide cell. |
| `.bento__cell--full` | Full-width footer cell. |
| `.bento__label`, `.bento__title`, `.bento__desc`, `.bento__metric` | Cell content slots. |

**Bento mobile rule**: at `≤700px`, all cells force `grid-column: 1 / -1 !important` to override any inline `style="grid-row:N;grid-column:M / N"` placements that pin cells in the 6-col grid.

### 2.5 The execution-pointer atoms (the unifying device)

These are the **shared visual vocabulary** that every section composes. The "execution pointer" is a conceptual entity (a cyan pulse representing live execution), not a single DOM node. Each section has its own animation that reuses these atoms.

| Class / keyframe | Purpose |
|---|---|
| `.exec-dot` | Pulsing cyan dot. 8×8 default. Modifiers: `--lg` (10×10), `--sm` (6×6). |
| `.exec-caret` | Blinking block cursor. Used in terminals and the wordmark. |
| `.exec-scan` | One-shot cyan beam sweep across an element. Apply to a positioned parent; the `::after` is the beam. |
| `@keyframes exec-pulse` | Box-shadow ripple, 1.6s loop. |
| `@keyframes exec-blink` | Caret on/off, 1.05s, `steps(2, jump-none)`. |
| `@keyframes exec-scan-sweep` | Beam translateX, 1.4s, one-shot. |
| `@keyframes exec-stamp` | One-shot scale + cyan ring "stamp" effect. Use on viewport entry. |

**Rule**: When you build a new animated section, compose these atoms instead of inventing new ones. Cohesion comes from sharing color, easing, and timing — not from a runtime framework.

### 2.6 Section-specific components

These were designed once for the landing page sections. Reuse if a similar use case appears.

| Section | Class prefix | What it is |
|---|---|---|
| `#what` | `.what-stage`, `.what-term`, `.what-rail`, `.what-chip` | Terminal panel + capability rail with chips that light up via `--chip-delay` CSS var. |
| `#see-difference` | `.diff-stage`, `.diff-pane`, `.diff-pane__col`, `.diff-anno` | CSS Grid 2-column drag-to-compare slider driven by `--split` (unitless 0..100). |
| `#production` | `.prod-stage`, `.prod-trace`, `.prod-node`, `.prod-panel` | Clickable trace explorer with auto-play through nodes. |
| `#enterprise` | `.ent-shelf`, `.ent-exhibit`, `.ent-ring`, `.ent-counter`, `.ent-versions`, `.ent-sbom`, `.ent-markers` | 5-column compliance shelf. Each exhibit is its own micro-component. |
| `#eval` | `.eval-stage`, `.eval-report`, `.eval-metric`, `.eval-marquee`, `.eval-tag` | Hardcoded mock test report + 2-row marquee. |
| `#faq` | `.repl`, `.repl__rail`, `.repl__cat`, `.repl__searchbar`, `.repl__q` | Search-driven docs REPL. |
| Footer | `.footer-term`, `.footer-term__tree` | Terminal sitemap (`$ tree selectools.dev/`). |
| Wordmark | `.wm`, `.wm--1`, `.wm--2`, `.wm--3` | 3 switchable variants gated on `[data-logo="N"]`. |

### 2.7 Bento legacy components (from PR #46)

| Class prefix | What it is |
|---|---|
| `.why-vignette` | "Three problems" before/after diff blocks with sticky cyan numerals. |
| `.persona` | Tilted persona postcards with display-font pull quotes. |
| `.path` | Live preview cards for "Get started" section. |
| `.path-term`, `.path-marquee`, `.path-graph` | Path-card content variants. |

---

## 3. Animation language

### 3.1 The reveal pattern

Every animated section uses the same pattern:

1. **CSS** sets the initial state (opacity 0, transform Y, etc.) and defines the animated state under a `.in-view` modifier class.
2. **JS** uses an `IntersectionObserver` (one IIFE per section) to add `.in-view` when the element enters the viewport.
3. The observer **disconnects after the first trigger** (one-shot).

```javascript
(function() {
  var stage = document.querySelector('.my-stage');
  if (!stage) return;
  var io = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) {
      if (!e.isIntersecting) return;
      stage.classList.add('in-view');
      io.disconnect();
    });
  }, { threshold: 0.25, rootMargin: '0px 0px -8% 0px' });
  io.observe(stage);
})();
```

**Standard threshold**: `0.25` to `0.35` depending on how committed the user should be. **Standard rootMargin**: `-8%` to `-10%` bottom inset so the animation fires after the section is comfortably in view, not the moment its top edge crosses.

**No global helper.** Each section's IIFE is self-contained. This makes the diff per-section reviewable and prevents one section's animation from breaking another.

### 3.2 Stagger recipes

Use **CSS variables on individual elements** to drive per-element delays:

```html
<div class="thing" style="--delay:0.45s">A</div>
<div class="thing" style="--delay:0.65s">B</div>
```

```css
.parent.in-view .thing {
  transition-delay: var(--delay, 0s);
}
```

For uniform staggers (e.g. compliance shelf exhibits), use `nth-child` rules with hard-coded delays. For non-uniform staggers (e.g. capability chips that fire when their corresponding code line is typed), use inline `--delay` vars.

### 3.3 Reduced motion

**Every animated component** must have a `prefers-reduced-motion: reduce` override that snaps to the final state. The fallback must be a complete, intentional static composition — not just "animation: none".

All overrides live in the single `@media (prefers-reduced-motion: reduce)` block at the bottom of the `<style>` section. Find the existing block and add to it; do not create a second.

---

## 4. Voice & tone rules

### 4.1 Banned punctuation

**Em-dashes (`—`, `&mdash;`) in user-facing copy.** When every section uses one, it becomes the load-bearing punctuation of the page and reads as AI-generated. Use a period, comma, colon, or rewrite. Source-code comments (CSS/HTML/JS) may keep their em-dashes — they're developer prose, not marketing copy.

### 4.2 Banned words in body copy

| Avoid | Use instead |
|---|---|
| leverage | use |
| seamless / seamlessly | (drop the word) |
| robust | specific quality (e.g. "tested across 5 Python versions") |
| comprehensive | complete, full, every |
| navigate (figurative) | move through, find, search |
| delve | look at, read, study |
| elevate | improve, make better |
| unlock (figurative) | enable, give you |
| harness (figurative) | use, take advantage of |
| transformative / game-changer / cutting-edge | (drop and rewrite around what it actually does) |
| boasts (a feature) | has, ships with, includes |
| in the realm of | for, in |
| It's worth noting that | (drop, just say the thing) |
| Moreover / Furthermore | (start a new sentence; the connection is implicit) |

### 4.3 Sentence structure rules

- **Short sentences over long ones.** A period is better than a comma in most marketing contexts.
- **No triple-adjective stacks** ("fast, secure, and scalable"). Pick one and prove it.
- **Numbers over adjectives.** "4612 tests" beats "well-tested". "$0.0012 per call" beats "affordable".
- **Active voice over passive.** "We tested across Python 3.9-3.13" beats "It has been tested across..."
- **No redundant phrases.** Don't say "in order to" when "to" works. Don't say "the fact that" when "that" works.
- **Don't restate the heading in the description.** The h2 says it. The desc adds something new.

### 4.4 Tone

- **Senior to senior.** The reader is a senior engineer evaluating a framework. Don't explain Python concepts. Don't use marketing-school formulas (Problem → Agitation → Solution).
- **Earned confidence, not arrogance.** Every claim is followed by a number, a code snippet, or a link to proof. "Production-ready" alone is empty. "Production-ready: 4612 tests, 95% coverage, CycloneDX SBOM, audited" is earned.
- **Specific over generic.** "PII redaction in the constructor" beats "enterprise security". "50 evaluators, free, local" beats "comprehensive testing".

---

## 5. Anti-patterns (what NOT to do)

These are real mistakes the project has made and corrected. Don't repeat them.

| Don't | Why |
|---|---|
| Use 6+ rectangular icon-cards in a grid | Templated. Reads as "AI slop". Use distinct visual genres per section instead. |
| Pin grid cells with inline `style="grid-row:N;grid-column:M / N"` without a mobile reset | Inline styles beat media queries via specificity. Cells overflow on mobile. |
| Rely on `clip-path: inset(0 calc(100% - var(--x)) 0 0)` for animated split layers | Browsers freeze the parsed clip-path value and don't re-eval `calc()` when the var changes. Use CSS Grid or set clip-path inline via JS. |
| Use `<nav class="my-rail">` for section sub-navigation | The global `nav { position: fixed }` selector will hoist your element to the top of the page. Use `<div role="navigation">`. |
| Define `.hide-mobile { display: none }` before `.nav-links { display: flex }` in source | Equal specificity, source order wins, the hide loses. Use `.nav-links.hide-mobile` (two-class specificity). |
| Add new keyframes for a new section's animation | Compose the existing exec-* atoms. New keyframes mean new visual vocabulary, which breaks cohesion. |
| Use generic icon SVGs as section accents | Use either real terminal output, real code, or composed CSS shapes (block-drawing, rings, counters). |
| Skip the `prefers-reduced-motion` fallback because "it's just a small animation" | Accessibility requirement. Every animation has a static fallback. |
| Hardcode hex colors in component CSS | Reference `var(--cyan)` etc. so palette changes propagate. |
| Use `cubic-bezier()` literals in component CSS | Reference `var(--ease)` or `var(--exec-ease-step)`. |

---

## 6. File structure

| Path | Purpose |
|---|---|
| `.impeccable.md` | Design **context** — brand, voice, principles, references. The why. |
| `docs/design-system.md` | Design **system** — tokens, components, animation, anti-patterns. The what. (this file) |
| `docs/superpowers/specs/` | Per-feature design specs. Each one references both files above. |
| `landing/index.html` | The **source of truth** for tokens and components. Contains the inline `<style>` block (~3500 lines) and JS observers. |
| `landing/favicon.svg` | Default favicon (Wordmark Variant 1). |
| `landing/favicon-cursor.svg` | Alternate favicon (Wordmark Variants 2 & 3). |

---

## 7. When to update this file

Update `docs/design-system.md` whenever you:

1. Add a new design token to `:root` (color, spacing, easing, duration)
2. Create a new reusable component class (anything with `.foo__*` BEM-style children)
3. Establish a new animation pattern (new IntersectionObserver flow, new keyframe family)
4. Catch a new anti-pattern that someone might repeat
5. Add a banned word or rewrite rule to the voice section

Keep this file in sync with `landing/index.html`. The file is the documentation; the HTML is the implementation. If they diverge, the implementation wins and this file gets corrected.
