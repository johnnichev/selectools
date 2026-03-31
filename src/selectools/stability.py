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
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Optional, TypeVar, Union, overload

_F = TypeVar("_F", bound=Callable[..., Any])
_C = TypeVar("_C", bound=type)


@overload
def stable(obj: _C) -> _C: ...


@overload
def stable(obj: _F) -> _F: ...


def stable(obj: Union[_F, _C]) -> Union[_F, _C]:
    """Set stability marker to 'stable' (API is frozen)."""
    obj.__stability__ = "stable"  # type: ignore[union-attr]
    return obj


stable.__stability__ = "stable"  # type: ignore[attr-defined]


@overload
def beta(obj: _C) -> _C: ...


@overload
def beta(obj: _F) -> _F: ...


def beta(obj: Union[_F, _C]) -> Union[_F, _C]:
    """Set stability marker to 'beta' (API may change in minor releases)."""
    obj.__stability__ = "beta"  # type: ignore[union-attr]
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

__all__ = ["stable", "beta", "deprecated"]
