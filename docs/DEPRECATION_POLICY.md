# Deprecation Policy

**Added in:** v0.19.2

This document describes how selectools handles deprecations of public APIs.

---

## Stability Levels

Every public API in selectools carries one of three stability markers:

| Marker | Meaning |
|--------|---------|
| `stable` | API is frozen. Breaking changes require a new major version. |
| `beta` | API may change in a minor release. No deprecation cycle guaranteed. |
| `deprecated` | API will be removed. A `DeprecationWarning` is emitted on every use. |

You can import the markers directly:

```python
from selectools.stability import stable, beta, deprecated
```

---

## Deprecation Window

Any API deprecated in version `0.X` will **not be removed before version `0.X+2`**.

| Deprecated in | Earliest removal |
|---------------|-----------------|
| v0.19 | v0.21 |
| v0.20 | v0.22 |
| v0.21 | v0.23 |

This gives you at least two minor releases to migrate.

---

## How to Detect Deprecations

### At runtime

Deprecated APIs emit a `DeprecationWarning` on every call or instantiation:

```
DeprecationWarning: hooks is deprecated since v0.16. Use AgentObserver instead.
```

Python silences `DeprecationWarning` by default in application code. To surface them during development:

```bash
python -W error::DeprecationWarning your_script.py
```

### In tests

```python
# pytest — turn deprecation warnings into errors
pytest -W error::DeprecationWarning

# or in pytest.ini / pyproject.toml:
[tool.pytest.ini_options]
filterwarnings = ["error::DeprecationWarning"]
```

### Programmatically

Every deprecated object exposes these attributes:

```python
from selectools.stability import deprecated

@deprecated(since="0.19", replacement="NewProvider")
def old_provider(): ...

old_provider.__stability__              # "deprecated"
old_provider.__deprecated_since__       # "0.19"
old_provider.__deprecated_replacement__ # "NewProvider"
```

---

## Migration Path

When you see a `DeprecationWarning`:

1. Read the message — it always names the replacement: `Use <X> instead.`
2. Check the [Changelog](CHANGELOG.md) for the version where `<X>` was introduced.
3. Update your code before the removal version.

---

## Scope

This policy covers the **public API** — everything exported from `selectools.__all__` and documented in this site.

It does **not** cover:

- Private symbols (prefixed with `_`)
- Modules marked `beta`
- Dev/test utilities in `tests/` or `scripts/`
- The `selectools serve` CLI (beta until v1.0.0)

---

## v1.0.0 and Beyond

At v1.0.0, all `stable` modules will be covered by semantic versioning. Breaking changes will require `v2.0.0`. Modules still marked `beta` at v1.0.0 are excluded from this guarantee.
