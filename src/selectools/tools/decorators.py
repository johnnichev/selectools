"""
Decorators for tool definition and registration.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ..stability import stable
from .base import ParamMetadata, Tool, ToolParameter


def _literal_info(type_hint: Any) -> Optional[Tuple[Any, List[Any]]]:
    """Return (base_type, enum_values) for Literal[...] hints, else None.

    Unwraps Optional[Literal[...]] as well. Base type is inferred from the
    first literal value (e.g. Literal["a", "b"] -> str).
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _literal_info(non_none[0])
    if sys.version_info >= (3, 10):
        import types as _types  # noqa: PLC0415

        if isinstance(type_hint, _types.UnionType):
            args = get_args(type_hint)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _literal_info(non_none[0])
    if origin is Literal:
        values = list(get_args(type_hint))
        if not values:
            return None
        base_type = type(values[0])
        return base_type, values
    return None


def _unwrap_type(type_hint: Any) -> Any:
    """Unwrap Optional[T] / Union[T, None] to T.

    Also strips generic parameters from collection types so that
    ``List[str]`` → ``list``, ``Dict[str, Any]`` → ``dict``, etc.
    This allows parameters annotated as ``Optional[List[str]]`` to be
    recognised as the supported ``list`` type rather than raising
    ``ToolValidationError: Unsupported parameter type: typing.List[str]``.

    BUG-11: Multi-type unions like ``Union[str, int]`` previously fell
    through to ``_validate_tool_definition`` which rejected them. We now
    default such unions to ``str`` — runtime values are then coerced by
    ``Tool._coerce_value`` (BUG-10) so int/float/bool inputs still work.
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        # Check for Optional (Union[T, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _unwrap_type(non_none_args[0])
        if len(non_none_args) > 1:
            # Multi-type union (e.g. Union[str, int]) — default to str.
            # Runtime values are coerced by tools/base.py::_coerce_value.
            return str
    # Handle Python 3.10+ X | Y syntax (types.UnionType)
    if sys.version_info >= (3, 10):
        import types  # noqa: PLC0415

        if isinstance(type_hint, types.UnionType):
            args = get_args(type_hint)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return _unwrap_type(non_none_args[0])
            if len(non_none_args) > 1:
                # Multi-type union (e.g. str | int) — default to str.
                return str
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

        # detailed metadata
        meta = param_metadata.get(name, {})
        description = meta.get("description", f"Parameter {name}")
        enum_values: Optional[List[Any]] = meta.get("enum")

        raw_type = type_hints.get(name, str)
        lit = _literal_info(raw_type)
        if lit is not None:
            param_type, literal_values = lit
            if enum_values is None:
                enum_values = literal_values
        else:
            param_type = _unwrap_type(raw_type)

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


def _build_tool_from_fn(func: Callable[..., Any], tool_kwargs: Dict[str, Any]) -> Tool:
    """Build a Tool instance from a callable and the kwargs ``@tool()`` received.

    Extracted as a helper so it can be reused both for top-level function
    tools and for per-instance bound method tools (see ``_BoundMethodTool``).
    """
    tool_name = tool_kwargs.get("name") or func.__name__
    tool_description = tool_kwargs.get("description") or inspect.getdoc(func) or f"Tool {tool_name}"
    parameters = _infer_parameters_from_callable(
        func,
        tool_kwargs.get("param_metadata"),
        tool_kwargs.get("injected_kwargs"),
    )
    return Tool(
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        function=func,
        injected_kwargs=tool_kwargs.get("injected_kwargs"),
        config_injector=tool_kwargs.get("config_injector"),
        streaming=tool_kwargs.get("streaming", False),
        screen_output=tool_kwargs.get("screen_output", False),
        terminal=tool_kwargs.get("terminal", False),
        requires_approval=tool_kwargs.get("requires_approval", False),
        cacheable=tool_kwargs.get("cacheable", False),
        cache_ttl=tool_kwargs.get("cache_ttl", 300),
    )


class _BoundMethodTool:
    """Descriptor that binds a ``@tool``-decorated method to its instance.

    Applying ``@tool()`` to a regular function returns a ``Tool`` whose
    ``function`` attribute is the raw callable — the agent executor calls
    ``tool.function(**llm_args)`` and everything works.

    Applying ``@tool()`` to a method (``def f(self, ...)``) is trickier:
    the LLM does not know about ``self``, so ``function(**llm_args)`` would
    call the method without its receiver and raise
    ``TypeError: missing 1 required positional argument: 'self'``.

    This descriptor solves it by returning a **per-instance** ``Tool`` from
    ``__get__``: the Tool's ``function`` is ``functools.partial(original_fn,
    instance)``, so the agent executor can invoke it with only the LLM's
    kwargs and the method still receives its receiver.

    Class-level access (``RAGTool.search_knowledge_base``) returns the
    descriptor itself, which proxies attribute lookups to a template ``Tool``
    so introspection (``.name``, ``.description``, ``.parameters``) keeps
    working.
    """

    def __init__(self, original_fn: Callable[..., Any], tool_kwargs: Dict[str, Any]) -> None:
        self._original_fn = original_fn
        self._tool_kwargs = tool_kwargs
        # Template Tool used for class-level introspection. ``self`` is
        # already skipped by ``_infer_parameters_from_callable`` so the
        # parameters field is correct for LLM schema generation.
        self._template = _build_tool_from_fn(original_fn, tool_kwargs)

    def __getattr__(self, name: str) -> Any:
        # Forward attribute lookups to the template Tool so that
        # ``MyClass.my_method.name`` / ``.description`` / ``.parameters``
        # still work at the class level.
        return getattr(self._template, name)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self
        bound_fn = functools.partial(self._original_fn, instance)
        functools.update_wrapper(bound_fn, self._original_fn)
        return _build_tool_from_fn(bound_fn, self._tool_kwargs)


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
) -> Callable[[Callable[..., Any]], Any]:
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
    tool_kwargs: Dict[str, Any] = {
        "name": name,
        "description": description,
        "param_metadata": param_metadata,
        "injected_kwargs": injected_kwargs,
        "config_injector": config_injector,
        "streaming": streaming,
        "screen_output": screen_output,
        "terminal": terminal,
        "requires_approval": requires_approval,
        "cacheable": cacheable,
        "cache_ttl": cache_ttl,
    }

    def decorator(func: Callable[..., Any]) -> Any:
        # Detect method: first parameter is named ``self``. If so, return a
        # descriptor that produces a per-instance bound Tool on attribute
        # access. Otherwise (regular function) build a plain Tool.
        try:
            sig_params = list(inspect.signature(func).parameters.values())
            is_method = bool(sig_params and sig_params[0].name == "self")
        except (TypeError, ValueError):
            is_method = False

        if is_method:
            return _BoundMethodTool(func, tool_kwargs)
        return _build_tool_from_fn(func, tool_kwargs)

    return decorator
