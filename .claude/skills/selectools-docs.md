# Selectools Documentation

Skill for writing documentation that follows selectools conventions.

## Trigger

Use when creating or updating documentation for selectools features, modules, or guides.

## Context

- **Docs engine**: MkDocs Material
- **Docs location**: `docs/`
- **Config**: `mkdocs.yml`
- **Build**: `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build`
- **Serve locally**: `cp CHANGELOG.md docs/CHANGELOG.md && mkdocs serve`

## Documentation Checklist (ALL required for every feature)

### 1. Module Documentation

Create `docs/modules/<FEATURE>.md` with:

```markdown
# Feature Name

Brief description of what this feature does and why it matters.

## Quick Start

\`\`\`python
# Minimal working example
from selectools import FeatureClass
result = FeatureClass(...)
\`\`\`

## API Reference

### ClassName

\`\`\`python
class ClassName:
    """One-line description."""

    def method(self, param: type) -> return_type:
        """What it does."""
\`\`\`

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param` | `str` | required | What it controls |

## Examples

### Basic Usage
...

### Advanced Usage
...

## Integration with Agent

\`\`\`python
agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(feature_option=True),
)
\`\`\`
```

### 2. Navigation Entry

Add to `mkdocs.yml` nav under the appropriate tab section.

### 3. Landing Page Update

Update `docs/index.md`:
- Add to the feature table/grid
- Update model/tool/test counts if changed

### 4. Documentation Index

Update `docs/README.md` with the new module page.

### 5. Quickstart Update

If user-facing, add a step to `docs/QUICKSTART.md` "next steps" table.

### 6. Architecture Update

If it adds a new system component, update `docs/ARCHITECTURE.md`.

### 7. Notebook Section

Add a section to `notebooks/getting_started.ipynb` with interactive examples.

### 8. Example Script

Create `examples/NN_feature_name.py` (next available number, zero-padded).

## Link Rules

- **Within docs/**: Use relative paths (`modules/AGENT.md`, `../ARCHITECTURE.md`)
- **Outside docs/** (ROADMAP.md, examples/, notebooks/): Use absolute GitHub URLs
  - Example: `https://github.com/johnnichev/selectools/blob/main/examples/01_hello_world.py`
- **Anchor format**: `#heading-text` (lowercase, hyphens, no special chars)
  - `## Tool Policy & Human-in-the-Loop` becomes `#tool-policy-human-in-the-loop`

## Hardcoded Counts (update ALL together)

These appear in multiple files:
- **Model count** (146): `index.md`, `README.md`, `MODELS.md`, `QUICKSTART.md`, `ARCHITECTURE.md`
- **Test count** (1183+): `README.md`, `index.md`, `CHANGELOG.md`
- **Example count** (32): `README.md`, `index.md`
- **Tool count** (24): `index.md`, `README.md`, `TOOLBOX.md`

## Style Conventions

- Use admonitions for callouts: `!!! tip`, `!!! warning`, `!!! example`
- Use tabbed content for install/usage variants: `=== "Tab Name"`
- Use Material icons: `:material-icon-name:`
- Code examples should be complete and runnable
- No comments explaining obvious code — only non-obvious intent

## Verify Build

Always run after doc changes:
```bash
cp CHANGELOG.md docs/CHANGELOG.md && mkdocs build
```

Check for broken links and warnings in the output.
