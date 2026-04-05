# docs/ — Documentation Conventions

## Engine

- **MkDocs Material**, deployed to GitHub Pages
- Build: `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build`
- Preview: `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs serve`

## File Organization

- Module docs: `docs/modules/UPPER_CASE.md` (e.g., `MEMORY.md`, `AGENT.md`)
- Guides: `docs/lower_case.md` (e.g., `QUICKSTART.md` is the exception — kept uppercase)
- ADRs: `docs/decisions/NNN-title.md`
- Nav structure defined in `mkdocs.yml`

## Linking Rules

- Files **inside** `docs/` use relative paths: `[Memory](modules/MEMORY.md)`
- Files **outside** `docs/` (CHANGELOG.md, ROADMAP.md, examples/) MUST use absolute GitHub URLs
- Never use relative `../` paths to escape the docs directory

## Pre-commit

- `check-yaml` hook needs `args: ["--unsafe"]` for mkdocs.yml (Python tags for emoji extensions)

## Feature Documentation Checklist

Every feature MUST update all of these:
1. **Module doc** in `docs/modules/*.md` — add sections, examples, "Since: vX.Y.Z"
2. **`docs/index.md`** — feature table and counts
3. **`docs/QUICKSTART.md`** — if user-facing
4. **`docs/llms.txt`** — module descriptions, counts, links
5. **`landing/index.html`** — stats bar (tests, examples, models, tools)
6. **`mkdocs.yml`** — nav labels if counts changed
7. **Example script**: `examples/NN_descriptive_name.py` (zero-padded number)
8. **Notebook**: step in `notebooks/getting_started.ipynb`

## Gotchas

- CHANGELOG.md lives at repo root — must be copied into `docs/` before build
- Gallery JSON (`landing/examples.json`) must be regenerated when examples are added
- Hardcoded counts (model count, test count, example count) appear across many files — audit all
- `docs/ARCHITECTURE.md` must be updated when adding new components
