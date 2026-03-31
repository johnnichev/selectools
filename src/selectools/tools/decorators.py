"""
Decorators for tool definition and registration.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from ..stability import stable
from .base import ParamMetadata, Tool, ToolParameter


def _unwrap_type(type_hint: Any) -> Any:
    """Unwrap Optional[T] / Union[T, None] to T.

    Also strips generic parameters from collection types so that
    ``List[str]`` → ``list``, ``Dict[str, Any]`` → ``dict``, etc.
    This allows parameters annotated as ``Optional[List[str]]`` to be
    recognised as the supported ``list`` type rather than raising
    ``ToolValidationError: Unsupported parameter type: typing.List[str]``.
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Check for Optional (Union[T, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _unwrap_type(non_none_args[0])
    # Handle Python 3.10+ X | Y syntax (types.UnionType)
    if sys.version_info >= (3, 10):
        import types  # noqa: PLC0415

        if isinstance(type_hint, types.UnionType):
            args = get_args(type_hint)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return _unwrap_type(non_none_args[0])
    # Strip generic parameters from collection types: List[str] → list,
    # Dict[str, Any] → dict, list[str] → list (Python 3.9+ native syntax).
    _SUPPORTED_ORIGINS = {list, dict}
    if origin in _SUPPORTED_ORIGINS:
        return origin
    return type_hint


def _infer_parameters_from_callable(
    func: Callable[..., Any],
    param_metadata: Optional[Dict[str, ParamMetadata]] = None,
    injected_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ToolParameter]:
    """
    Inspect function signature to create ToolParameter objects.

    Args:
        func: The function to inspect.
        param_metadata: Optional manual overrides for parameter descriptions/enums.
        injected_kwargs: Keys to exclude from inferred parameters (not visible to LLM).

    Returns:
        List of ToolParameter objects inferred from type hints.
    """
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Forward references that can't be resolved fall back to raw annotations,
        # which may be strings (PEP 563). Use an empty dict so all params default to str.
        type_hints = getattr(func, "__annotations__", {})
    sig = inspect.signature(func)
    parameters = []
    param_metadata = param_metadata or {}
    injected_names = set(injected_kwargs.keys()) if injected_kwargs else set()

    for name, param in sig.parameters.items():
        # Skip self/cls for methods
        if name in ("self", "cls"):
            continue

        # Skip *args and **kwargs — variadic collectors are not discrete LLM parameters
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Skip injected kwargs — they are provided at runtime, not by the LLM
        if name in injected_names:
            continue

        # Get type hint (default to str if missing)
        raw_type = type_hints.get(name, str)
        param_type = _unwrap_type(raw_type)

        # detailed metadata
        meta = param_metadata.get(name, {})
        description = meta.get("description", f"Parameter {name}")
        enum_values = meta.get("enum")

        # Check for optional/default values
        is_optional = param.default != inspect.Parameter.empty
        # Optional type hint (e.g. Optional[str]) handling could be added here
        # For now we rely on the default value check

        parameters.append(
            ToolParameter(
                name=name,
                param_type=param_type,
                description=description,
                required=not is_optional,
                enum=enum_values,
            )
        )

    return parameters


@stable
def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_metadata: Optional[Dict[str, ParamMetadata]] = None,
    injected_kwargs: Optional[Dict[str, Any]] = None,
    config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
    streaming: bool = False,
    screen_output: bool = False,
    terminal: bool = False,
    requires_approval: bool = False,
    cacheable: bool = False,
    cache_ttl: int = 300,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to convert a function into a Tool.

    Introspects the function signature and type hints to automatically generate
    tool parameters and JSON schema.

    Args:
        name: Optional custom name (defaults to function name).
        description: Optional description (defaults to docstring).
        param_metadata: Dict mapping parameter names to metadata (description, enum).
        injected_kwargs: Kwargs to inject nicely at runtime (hidden from LLM).
        config_injector: Callable returning kwargs to inject at runtime.
        streaming: Whether the tool streams results (returns Generator).
        screen_output: Screen this tool's output for prompt injection.
            Default: ``False``.
        terminal: If True, executing this tool stops the agent loop and
            returns the tool result as the final response.  Default: ``False``.
        requires_approval: If True, the tool always requires human approval
            before execution, regardless of ToolPolicy rules.  Default: ``False``.
        cacheable: If True, tool results are cached by name + args when the
            agent has a cache configured.  Default: ``False``.
        cache_ttl: Time-to-live in seconds for cached results.
            Default: ``300`` (5 minutes).

    Returns:
        Decorator function that returns a Tool instance.

    Example:
        >>> @tool(description="Calculate sum")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> tool_instance = add
        >>> print(tool_instance.name)
        'add'
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or inspect.getdoc(func) or f"Tool {tool_name}"
        parameters = _infer_parameters_from_callable(func, param_metadata, injected_kwargs)

        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=func,
            injected_kwargs=injected_kwargs,
            config_injector=config_injector,
            streaming=streaming,
            screen_output=screen_output,
            terminal=terminal,
            requires_approval=requires_approval,
            cacheable=cacheable,
            cache_ttl=cache_ttl,
        )
        return tool_instance

    return decorator
