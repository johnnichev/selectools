# Landing Overdrive — Fused Design Spec

**Date**: 2026-04-07
**Branch**: `feat/landing-overdrive-fused`
**Driver**: Replace 6 templated sections + footer + wordmark on `landing/index.html` with a unified, technically ambitious treatment that lives up to the .impeccable.md design principles.

## Why this exists

The user identified 6 sections of `landing/index.html` that read as "AI slop" — templated rectangular content blocks with the same code-on-one-side / cards-on-the-other geometry. Even when individual content is good, the uniform composition makes the page blur together. The /bolder + /animate + /delight + /overdrive skill stack was invoked to give each section its own visual genre and tie them together with a unified animating principle.

The solution is **A+C fused** (Direction A "Living Documentation" + Direction C "Agent in Motion") with C reframed to strip the cuteness: instead of an agent character, the page is unified by an **execution pointer** — an abstract cyan pulse that represents where the program is executing right now, like a debugger step indicator.

## The animating principle: the execution pointer

There is one persistent visual concept across the page: a small cyan pulse called the **execution pointer**. It is **not a single DOM node moving across the page** — it is a conceptual entity, implemented as section-local animations that share the same color, easing curves, and timing language.

**Why section-local instead of global**: a page-spanning element creates performance hazards (constant scroll listeners, layout thrash) and accessibility hazards (one element responsible for too much motion). Section-local animations are individually controlled by `IntersectionObserver`, paused when off-screen, and degrade independently.

**Visual specification**:
- Color: brand cyan `#22d3ee` (same as existing accent)
- Default form: `8px` cyan dot with `0 0 16px rgba(34, 211, 238, 0.6)` glow
- Default motion: pulses on a 1.6s loop with `cubic-bezier(0.4, 0, 0.6, 1)`
- Section variants: scan line (sweeps horizontally), step indicator (jumps between graph nodes), caret (blinks at end of text), stamp (pulses on element entry)

**The pointer is the cohesion**. Every section's hero animation uses one of these forms. Users will not consciously notice the connection — they will feel the page hangs together.

## Section specifications

Each section below has: **what it replaces**, **the new treatment**, **the hero moment**, **mobile fallback**, **reduced-motion fallback**.

### Section 1 — `#what-is` "What is Selectools?"

**Replaces**: Bullet stack on the left + generic Mac terminal code window on the right.

**New treatment**: A single full-width terminal panel. The pointer types out a `pip install` → `python hello.py` session in real time. As the script runs, the capability tags (Multi-agent graphs, Tool calling, Hybrid RAG, 50 evaluators, Guardrails, Audit logging, Visual builder) **light up in sync with the lines that import them**. The tags are not a list — they are a runtime trace.

Layout: terminal panel takes ~60% of the column, capability tags become a rail down the right side that "lights up" line by line. Section heading sits above.

**Hero moment**: When the section enters the viewport, the pointer types `>>> from selectools import Agent` and the "Multi-agent graphs" tag pulses cyan. Then `>>> agent = Agent(tools=[search])` and "Tool calling" pulses. The sequence is choreographed so each line earns its tag.

**Mobile fallback** (`pointer: coarse`): same animation, plays on viewport entry as a one-shot. Tags wrap below the terminal instead of right rail.

**Reduced-motion fallback**: terminal renders fully complete, all tags pre-lit. Static. Still tells the same story.

### Section 2 — `#see-the-difference` "See the difference"

**Replaces**: Side-by-side LangGraph (25 lines) vs selectools (1 line) code blocks + green checkmark list.

**New treatment**: A horizontal **drag-to-compare slider** between the two code blocks. Default position: 50/50. Drag the cyan handle right to wipe LangGraph away and reveal more selectools; drag left to do the opposite. As you drag past key positions, **inline annotations appear** (the four old checkmarks: "Plain Python", "HITL resumes at yield point", "Zero extra deps", "Deploy with `selectools serve`"). The annotations are positioned at the line numbers where the corresponding pain point lives in the LangGraph code, so they earn their place by tying to specific ceremony.

**Hero moment**: On viewport entry, the slider auto-demonstrates by sweeping from 50/50 → 100% selectools → 50/50 once, then sits idle awaiting user drag. The pointer is the slider handle.

**Mobile fallback**: drag-to-compare doesn't work cleanly on touch. Replaced with a tap-to-flip card: tap to show LangGraph, tap to show selectools. Annotations still appear inline.

**Reduced-motion fallback**: no auto-sweep. Slider sits at 50/50, fully usable as static comparison.

### Section 3 — `#production-ready` "Production-ready"

**Replaces**: `agent.py` code window on the left + 4 stacked `result.*` cards on the right.

**New treatment**: Code window stays on the left. Right side becomes an **interactive trace explorer**. The agent loop is rendered as a horizontal sequence of clickable nodes: `llm_call → tool_selection → tool_execution → llm_call`. Below the nodes, there's a single result panel that swaps content based on which node is active. Default: `llm_call` is active showing `result.content`. Click any node to swap the panel.

The 4 old cards (`result.content` / `result.reasoning` / `result.trace` / `result.usage`) become the 4 panel states: each step of the agent loop reveals one of them.

**Hero moment**: On viewport entry, the pointer travels through the trace nodes one by one (`llm_call → tool_selection → tool_execution → llm_call`), and the result panel below auto-swaps to match. After the auto-play finishes, the user can click any node to inspect.

**Mobile fallback**: nodes stack vertically. Auto-play still runs. Tap to swap.

**Reduced-motion fallback**: all 4 nodes pre-rendered as a static row. Result panel defaults to `result.trace` (the most informative). User can still click to swap.

### Section 4 — `#enterprise-ready` "Enterprise-ready"

**Replaces**: A single full-width pill row card with `SBOM (CycloneDX) · Security audit published · @stable / @beta / @deprecated · Python 3.9 to 3.13 · 95% test coverage · 4612 tests`.

**New treatment**: A **compliance shelf** with 5 distinct artifacts laid out as exhibits in a grid. Each exhibit has its own micro-component:

1. **Coverage ring** — animated SVG circular gauge climbing from 0% to 95% on viewport entry. JetBrains Mono number in the center.
2. **Test counter** — large number that ticks from 0 to 4612 with `requestAnimationFrame` easing. "tests passing" caption beneath.
3. **Version stack** — a vertical strip showing `3.9 / 3.10 / 3.11 / 3.12 / 3.13` like a compatibility ladder, each rung lighting up as the pointer climbs it.
4. **CycloneDX SBOM card** — a paper-document-style card with a folded corner, "CycloneDX 1.5" label, and a tiny barcode pattern. Static visual exhibit.
5. **Audit doc** — a stamped-document visual with `@stable / @beta / @deprecated` rendered as colored stamps.

Layout: 5-column grid on desktop, 2-column on tablet, 1-column on mobile. Section header sits above.

**Hero moment**: Pointer becomes a stamp. As each exhibit enters the viewport, the pointer "stamps" it — the exhibit pulses cyan, then settles into its rest state. Sequence is staggered by ~150ms per exhibit.

**Mobile fallback**: same animations, played sequentially on viewport entry, single column.

**Reduced-motion fallback**: all exhibits in their final state. Coverage ring at 95%, counter at 4612, version stack fully lit.

### Section 5 — `#evals` "Built-in evaluation"

**Replaces**: `test_agent.py` code window + 3 cards (Deterministic chips, LLM-as-Judge chips, Infrastructure checkmarks).

**New treatment**: Code window stays on the left. Right side becomes a **live test report skeleton** that fills in as the script "runs". The report is a **hardcoded mock**, not a real test runner — it hardcodes the same metrics that the existing static cards display (accuracy 1.0, latency p50 142ms, cost $0.002). The "running" effect is purely visual choreography. The report has:
- A header bar mimicking real test runner output (`Running 1 test... ✓ 1 passed in 142ms`)
- Three metric rows: accuracy bar (climbs to 100%), latency p50 (ticks to 142ms), total cost (ticks to $0.002)
- A status badge ("PASSED" with green dot)

Below the report (and below the code), spanning full width, sits a **2-row marquee of all 50 evaluator names** scrolling in opposite directions: deterministic evaluators on top, LLM-judge evaluators on bottom. Speed: ~32s and ~38s loops. Too many to count, which is the point.

**Hero moment**: Pointer enters the code window, the cursor steps through the `EvalSuite` definition, then "runs" — the report on the right fills in metric by metric. Once metrics are filled, the marquee starts scrolling.

**Mobile fallback**: report and code stack vertically. Marquee still runs, narrower viewport.

**Reduced-motion fallback**: report fully filled in. Marquee replaced with a static 2-column grid of all 50 evaluator names.

### Section 6 — `#faq` "Frequently asked questions"

**Replaces**: 10 stacked accordion buttons.

**New treatment**: A **`selectools docs` REPL**. Layout:
- Search bar at top, prompt-styled (`docs > _`)
- Left rail: category tabs (`# getting-started`, `# concepts`, `# providers`, `# advanced`)
- Main panel: questions render as `?` prompts in JetBrains Mono. When clicked (or auto-selected), the answer streams in below in monospace, with **inline syntax-highlighted code blocks** where relevant.

Search filters questions in real time as the user types. Categories filter by tag. The answer panel feels like reading docs in a terminal.

**Hero moment**: On viewport entry, the pointer types the first question into the search bar (`How is Selectools different from LangChain?`) and the answer streams in character-by-character. Then idles, awaiting user input.

**Mobile fallback**: left rail collapses into a horizontal scroll of category chips above the panel. Search bar stays. Animations same.

**Reduced-motion fallback**: search bar empty. First question selected by default. Answer rendered in full immediately.

### Footer

**Replaces**: Whatever is currently there (3-column links + legal).

**New treatment**: A **terminal sitemap**. Renders as `$ tree selectools.dev/` output:

```
$ tree selectools.dev/
.
├── docs/
│   ├── quickstart.md
│   ├── concepts.md
│   ├── providers.md
│   └── api-reference/
├── examples/  # 88 numbered scripts
├── builder/   # visual flow editor
├── github/    # source + issues
└── pypi/      # pip install selectools

# selectools v0.20.1 · Apache-2.0 · NichevLabs
# made for developers who ship
$ _
```

Each path is a hover-highlighted link. The trailing `$ _` has a blinking caret that never stops — the page is "still running". Social/legal/version inline as `# comments` in the dev-grey color.

**Mobile fallback**: same layout, JetBrains Mono shrinks to fit, paths wrap on long lines.

**Reduced-motion fallback**: caret static, no blink.

## Wordmark — three switchable variants

All three are built and switchable via `?logo=1|2|3` URL param. Default is variant `1`. The user picks the winner visually after seeing all three live.

### Variant 1 — `[•] selectools`

A pulsing cyan dot inside square brackets, followed by the wordmark in Plus Jakarta Sans semibold. The dot pulses on a 1.6s loop. Hover: brightens. Page load: one strong pulse.

**Favicon**: the bracket-dot reduces cleanly to 16×16 (`[•]` in cyan on slate). Written to `landing/favicon.svg` (the active favicon when Variant 1 is the default).

**State variants** for reuse elsewhere: `[• ]` (idle), `[•]` (running), `[✓]` (done), `[✗]` (error).

### Variant 2 — `s▌electools`

A block cursor `▌` lives inside the wordmark. On page load, types out: `s▌` → `se▌` → `sel▌` → ... → `selectools▌`. After type-on, the cursor settles between `s` and `e` and blinks there forever. Hover: cursor jumps to the position you hover over.

**Favicon**: paired with a standalone `▌` block-cursor mark, written to `landing/favicon.svg` only when Variant 2 is the active default. Since the default is Variant 1, this favicon ships as `landing/favicon-cursor.svg` for reference and only swaps in if the user picks Variant 2 as the winner.

### Variant 3 — Terminal banner art

A 6-row box-drawing block-glyph rendering of `SELECTOOLS` set in JetBrains Mono. Cyan-on-slate. On page load, the execution pointer runs a **scan line across the top edge of the glyphs** like a dot-matrix printer printing the banner. After scan completes, the whole banner gets one strong glow pulse and settles.

On scroll, in the sticky header, the banner **collapses** into a single-row `> selectools` prompt to save vertical space. View Transitions API (same-document mode, supported in all evergreen browsers as of 2025 — no Firefox fallback needed) morphs between the two states cleanly. If `document.startViewTransition` is missing, the swap is instant with a brief opacity crossfade fallback.

**Mobile**: full 6-row banner doesn't fit at 360px. Replaced with a 3-row compact box-drawing version that fits:

```
┌─┐┌─┐┬  ┌─┐┌─┐┌┬┐┌─┐┌─┐┬  ┌─┐
└─┐├┤ │  ├┤ │   │ │ ││ ││  └─┐
└─┘└─┘┴─┘└─┘└─┘ ┴ └─┘└─┘┴─┘└─┘
```

**Favicon**: standalone `▌` block-cursor mark (same as Variant 2 — they share the cursor glyph). Same `landing/favicon-cursor.svg` reference file. Becomes the active `landing/favicon.svg` only if the user picks Variant 3 as the winner.

**Accessibility**: full banner gets `aria-hidden="true"`. An adjacent visually-hidden `<h1>selectools</h1>` carries the semantic content for screen readers.

## Implementation order

Single PR, atomic commits per section so review stays clean in `git log -p`. Build order is chosen to minimize cross-section CSS conflicts:

1. **Setup commit**: branch from `main`, create CSS architecture for execution-pointer system (shared variables for color, easing, timing). Add `IntersectionObserver` helpers.
2. **Section 1** (What is Selectools?) — establishes the terminal-typing pattern other sections will reuse.
3. **Section 5** (Built-in evaluation) — reuses Section 1's terminal pattern, adds report skeleton.
4. **Section 3** (Production-ready) — introduces the trace-explorer pattern.
5. **Section 2** (See the difference) — introduces the drag-to-compare pattern (more isolated, lower reuse).
6. **Section 4** (Enterprise-ready) — compliance shelf, no shared patterns with other sections.
7. **Section 6** (FAQ) — REPL pattern, reuses terminal styling from Section 1.
8. **Footer** — terminal sitemap, reuses terminal styling.
9. **Wordmark variants** — all three built, default is Variant 1, query param to switch.
10. **Final cleanup commit**: cross-section testing, reduced-motion audit, mobile audit, copy polish.

Each commit gets a Playwright visual check before moving to the next.

## Performance budget

`landing/index.html` is a single self-contained file with inline `<style>` and `<script>` blocks. The budget below is for **bytes added to those inline blocks**, not separate files.

- **Added JS**: must stay under 8KB unminified (the file is uncompressed; gzip on the wire handles minification). No frameworks. Vanilla.
- **Added CSS**: must stay under 16KB unminified.
- **No new fonts**: reuses Plus Jakarta Sans + JetBrains Mono already loaded.
- **No new image assets**: everything is CSS, SVG, or text. Favicons are SVG.
- **Animation FPS target**: 60fps on mid-range Android (Galaxy S23 baseline from prior PRs).
- **Initial paint**: must not regress LCP. New sections lazy-init their animations via IntersectionObserver and pause when off-screen.

## Accessibility requirements

- **WCAG 2.1 AA** compliance maintained. Color contrast on all new content checked against `#0f172a` background.
- **`prefers-reduced-motion: reduce`**: every animated element has a static fallback specified in this doc. Not optional.
- **Keyboard navigation**: all interactive elements (slider handle, trace nodes, FAQ search, FAQ category tabs) are keyboard reachable with visible focus rings.
- **Screen readers**: animations are `aria-hidden`. Semantic content lives in adjacent visually-hidden elements where needed.
- **Touch targets**: 44×44px minimum on `pointer: coarse`.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Drag-to-compare slider unusable on touch | Replaced with tap-to-flip card on `pointer: coarse` |
| Marquee causes layout shift on slow connections | Marquee container has fixed height; content lazy-mounts |
| Trace explorer click handlers conflict with anchor links | Uses `<button>` elements, not `<a>` |
| Banner ASCII art breaks selection (users dragging to copy) | `user-select: none` on the banner; visually-hidden `<h1>selectools</h1>` provides selectable text |
| Section-local IntersectionObservers leak memory | Each section's observer disconnects after first trigger; one-shot animations |
| Reduced-motion fallback looks "broken" | Each fallback is designed to be a complete, intentional static composition, not a stripped animation |
| Wordmark variants hard to switch during testing | `?logo=1|2|3` query param; default is Variant 1; persisted to `localStorage` after first toggle |

## Out of scope

- No changes to other landing sections (hero, integrations, comparison table at top, CTAs)
- No changes to MkDocs theme or `docs/` site
- No changes to `pyproject.toml`, `README.md`, or any code outside `landing/`
- No copy rewrites — only the structural/visual treatment changes. Existing copy is preserved unless a section's new structure makes a line redundant.
- No new images, logos, or external assets. Everything is hand-built in CSS/SVG/text.

## Definition of done

- All 6 sections rebuilt per spec, each with its hero animation working
- Footer rebuilt as terminal sitemap
- All 3 wordmark variants live and switchable via `?logo=`
- Favicon updated (`landing/favicon.svg` for Variants 1 / 2-3)
- Mobile testing on a `pointer: coarse` viewport (Playwright emulation) confirms all sections render correctly
- `prefers-reduced-motion: reduce` audit confirms all animations have static fallbacks
- No regression in existing sections (hero, comparison table, integrations)
- Performance budget respected (CSS/JS sizes within limits)
- PR opened against `main` with section-by-section commits in `git log -p`
