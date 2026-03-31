# Stability Markers

**Added in:** v0.19.2
**File:** `src/selectools/stability.py`
**Exports:** `stable`, `beta`, `deprecated`

## Overview

Stability markers are decorators that signal the maturity of any public class or function. They let users know which APIs are safe to depend on, which may change, and which are being phased out.

```python
from selectools.stability import stable, beta, deprecated
# or
from selectools import stable, beta, deprecated
```

| Marker | Meaning |
|--------|---------|
| `@stable` | API is frozen. Breaking changes require a major version bump. |
| `@beta` | API may change in a minor release. No deprecation cycle guaranteed. |
| `@deprecated` | API will be removed. Emits `DeprecationWarning` on every use. |

## Quick Start

```python
from selectools.stability import stable, beta, deprecated

@stable
class MyAgent:
    """This API is frozen."""
    ...

@beta
class MyExperimentalFeature:
    """May change in the next minor release."""
    ...

@deprecated(since="0.19", replacement="MyAgent")
class OldAgent:
    """Emits DeprecationWarning on every instantiation."""
    ...
```

## API Reference

### `stable(obj)`

Marks a function or class as stable. Sets `obj.__stability__ = "stable"`. Zero runtime overhead — the original object is returned unchanged.

```python
@stable
def my_function(x: int) -> int:
    return x * 2
```

Works on both functions and classes.

### `beta(obj)`

Marks a function or class as beta. Sets `obj.__stability__ = "beta"`. Zero runtime overhead.

```python
@beta
class ExperimentalProvider:
    ...
```

### `deprecated(since, replacement=None)`

Marks a function or class as deprecated.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `since` | `str` | Version string when deprecation was introduced, e.g. `"0.19"` |
| `replacement` | `str \| None` | Name of the replacement API (omit if no replacement exists) |

**On functions** — wraps the function with `functools.wraps`; emits `DeprecationWarning(stacklevel=2)` on every call.

**On classes** — patches `__init__`; emits `DeprecationWarning` on every instantiation.

```python
@deprecated(since="0.19", replacement="NewProvider")
def old_factory() -> Agent:
    return Agent(...)

# Warning: old_factory is deprecated since v0.19. Use NewProvider instead.
agent = old_factory()
```

### Introspection

Every decorated object exposes these attributes:

```python
fn.__stability__              # "stable" | "beta" | "deprecated"
fn.__deprecated_since__       # set only on @deprecated objects
fn.__deprecated_replacement__ # set only on @deprecated objects (may be None)
```

## Surfacing Warnings in Tests

Python silences `DeprecationWarning` by default. To turn them into errors during development:

```bash
python -W error::DeprecationWarning your_script.py
pytest -W error::DeprecationWarning
```

Or in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::DeprecationWarning"]
```

## Deprecation Window

Any API deprecated in `v0.X` will not be removed before `v0.X+2`.

See [Deprecation Policy](../DEPRECATION_POLICY.md) for the full policy.

## See Also

- [Deprecation Policy](../DEPRECATION_POLICY.md)
- [Security Audit](../SECURITY.md)
- [Changelog](../CHANGELOG.md)
