"""
Stability markers for the selectools public API.

Use these decorators to signal the stability of any public function or class:

- ``@stable``  — API is frozen; breaking changes require a major version bump.
- ``@beta``    — API may change in a minor release; no deprecation cycle guaranteed.
- ``@deprecated(since, replacement)`` — Emits ``DeprecationWarning`` on use and will
  be removed after the minimum deprecation window (2 minor releases).

Examples::

    from selectools.stability import stable, beta, deprecated

    @stable
    def my_function(): ...

    @beta
    class MyExperimentalClass: ...

    @deprecated(since="0.19", replacement="NewThing")
    def old_function(): ...

Some public symbols cannot carry a ``__stability__`` attribute:

- module-level constants (``str``/``float``/``list``/``dict``/``tuple`` instances)
- typing aliases (``Union[...]``, ``Callable[...]``, and friends)
- ``@runtime_checkable`` Protocol classes — on Python 3.9-3.11 a
  ``__stability__`` class attribute becomes a structural member of the
  protocol and silently breaks ``isinstance()`` for conforming
  implementations that do not define it.

Those symbols are registered in :data:`STABILITY_REGISTRY` via
:func:`register_stability` at their definition site instead.

Module-level stability follows a plain-attribute convention: every public
submodule sets ``__stability__ = "stable"`` or ``"beta"`` at module scope to
declare the default guarantee for its surface (enforced by
``tests/test_architecture.py``).
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

_F = TypeVar("_F", bound=Callable[..., Any])
_C = TypeVar("_C", bound=type)


@overload
def stable(obj: _C) -> _C: ...


@overload
def stable(obj: _F) -> _F: ...


def stable(obj: Any) -> Any:
    """Set stability marker to 'stable' (API is frozen)."""
    obj.__stability__ = "stable"
    return obj


stable.__stability__ = "stable"  # type: ignore[attr-defined]


@overload
def beta(obj: _C) -> _C: ...


@overload
def beta(obj: _F) -> _F: ...


def beta(obj: Any) -> Any:
    """Set stability marker to 'beta' (API may change in minor releases)."""
    obj.__stability__ = "beta"
    return obj


beta.__stability__ = "stable"  # type: ignore[attr-defined]


def deprecated(
    since: str,
    replacement: Optional[str] = None,
) -> Callable[[Union[_F, _C]], Union[_F, _C]]:
    """Mark a function or class as deprecated.

    Emits a ``DeprecationWarning`` on every call or instantiation.

    Args:
        since: The version in which the deprecation was introduced (e.g. ``"0.19"``).
        replacement: Optional name of the replacement API to suggest in the warning.

    Example::

        @deprecated(since="0.19", replacement="NewProvider")
        class OldProvider:
            ...
    """

    def _make_message(name: str) -> str:
        msg = f"{name} is deprecated since v{since}."
        if replacement:
            msg += f" Use {replacement} instead."
        return msg

    @overload
    def decorator(obj: _C) -> _C: ...

    @overload
    def decorator(obj: _F) -> _F: ...

    def decorator(obj: Union[_F, _C]) -> Union[_F, _C]:
        if inspect.isclass(obj):
            # Wrap __init__ so the warning fires on instantiation.
            original_init = obj.__init__  # type: ignore[misc]
            message = _make_message(obj.__qualname__)

            @functools.wraps(original_init)
            def _new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = _new_init  # type: ignore[method-assign]
            obj.__stability__ = "deprecated"  # type: ignore[union-attr]
            obj.__deprecated_since__ = since  # type: ignore[union-attr]
            obj.__deprecated_replacement__ = replacement  # type: ignore[union-attr]
            return obj  # type: ignore[return-value]
        else:
            message = _make_message(obj.__qualname__)

            @functools.wraps(obj)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            _wrapper.__stability__ = "deprecated"  # type: ignore[attr-defined]
            _wrapper.__deprecated_since__ = since  # type: ignore[attr-defined]
            _wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
            return _wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


deprecated.__stability__ = "stable"  # type: ignore[attr-defined]

_VALID_LEVELS = ("stable", "beta", "deprecated")

#: Stability levels for public symbols that cannot carry a ``__stability__``
#: attribute (constants, typing aliases, runtime-checkable Protocols).
#: Keys are the public symbol names exactly as they appear in ``__all__``.
STABILITY_REGISTRY: Dict[str, str] = {}


def register_stability(name: str, level: str) -> None:
    """Register the stability level of a symbol that cannot be decorated.

    Use this for module-level constants, typing aliases, and
    ``@runtime_checkable`` Protocol classes (decorating those would change
    ``isinstance()`` behavior on Python 3.9-3.11). Call it at the symbol's
    definition site, right after the assignment.

    Raises:
        ValueError: If ``level`` is not a recognized stability level, or if
            ``name`` was already registered with a different level.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid stability level {level!r} for {name!r}; expected one of {_VALID_LEVELS}"
        )
    existing = STABILITY_REGISTRY.get(name)
    if existing is not None and existing != level:
        raise ValueError(
            f"Conflicting stability registration for {name!r}: {existing!r} vs {level!r}"
        )
    STABILITY_REGISTRY[name] = level


def get_stability(obj: Any, name: Optional[str] = None) -> Optional[str]:
    """Return the stability level of a public symbol, or ``None`` if unmarked.

    Checks the symbol's own ``__stability__`` attribute first (the marker set
    by ``@stable``/``@beta``/``@deprecated``), then falls back to
    :data:`STABILITY_REGISTRY` keyed by ``name``. For classes and functions
    only the symbol's *own* marker counts — markers inherited from a base
    class do not, so every public subclass must be marked explicitly.
    """
    if inspect.isclass(obj) or inspect.isroutine(obj):
        level = vars(obj).get("__stability__")
    else:
        level = getattr(obj, "__stability__", None)
    if isinstance(level, str):
        return level
    if name is not None:
        return STABILITY_REGISTRY.get(name)
    return None


# The stability API is itself part of the v1.0 contract. ``stable``/``beta``/
# ``deprecated`` carry their markers at their definition sites above; the
# remaining three are marked here (the dict via the registry, the functions
# via the same direct-attribute pattern as the decorators).
register_stability("STABILITY_REGISTRY", "stable")
register_stability.__stability__ = "stable"  # type: ignore[attr-defined]
get_stability.__stability__ = "stable"  # type: ignore[attr-defined]

__stability__ = "stable"

__all__ = [
    "stable",
    "beta",
    "deprecated",
    "STABILITY_REGISTRY",
    "register_stability",
    "get_stability",
]
