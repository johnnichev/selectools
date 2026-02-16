"""
Decorators for tool definition and registration.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from .base import ParamMetadata, Tool, ToolParameter


def _unwrap_type(type_hint: Any) -> Any:
    """Unwrap Optional[T] to T."""
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Check for Optional (Union[T, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _unwrap_type(non_none_args[0])
    return type_hint


def _infer_parameters_from_callable(
    func: Callable[..., Any], param_metadata: Optional[Dict[str, ParamMetadata]] = None
) -> List[ToolParameter]:
    """
    Inspect function signature to create ToolParameter objects.

    Args:
        func: The function to inspect.
        param_metadata: Optional manual overrides for parameter descriptions/enums.

    Returns:
        List of ToolParameter objects inferred from type hints.
    """
    type_hints = get_type_hints(func)
    sig = inspect.signature(func)
    parameters = []
    param_metadata = param_metadata or {}

    for name, param in sig.parameters.items():
        # Skip self/cls for methods
        if name in ("self", "cls"):
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


def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_metadata: Optional[Dict[str, ParamMetadata]] = None,
    injected_kwargs: Optional[Dict[str, Any]] = None,
    config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
    streaming: bool = False,
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
        # Use provided name or function name
        tool_name = name or func.__name__

        # Use provided description or docstring
        tool_description = description or inspect.getdoc(func) or f"Tool {tool_name}"

        # Infer parameters
        parameters = _infer_parameters_from_callable(func, param_metadata)

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=func,
            injected_kwargs=injected_kwargs,
            config_injector=config_injector,
            streaming=streaming,
        )
        return tool_instance

    return decorator
