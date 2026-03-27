"""
Tool composition — chain tools into a single composite tool.

Usage::

    from selectools import tool, compose

    @tool()
    def fetch(url: str) -> str: ...

    @tool()
    def parse(html: str) -> dict: ...

    @tool()
    def extract(data: dict, field: str) -> str: ...

    # Chain into a single tool the LLM can call
    fetch_and_extract = compose(fetch, parse, extract, name="fetch_and_extract")
    agent = Agent(tools=[fetch_and_extract], ...)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .tools.base import Tool
from .tools.decorators import tool as tool_decorator


def compose(
    *tools_or_fns: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    """Chain multiple tools/functions into a single composite tool.

    The output of each tool is passed as the first argument to the next.
    The composite tool exposes the FIRST tool's parameters to the LLM.

    Args:
        *tools_or_fns: Tool objects or plain callables to chain.
        name: Name for the composite tool. Auto-generated if omitted.
        description: Description for the LLM. Auto-generated if omitted.

    Returns:
        A Tool that executes the chain sequentially.

    Example::

        fetch_and_parse = compose(fetch_url, parse_html, extract_text)
        agent = Agent(tools=[fetch_and_parse], provider=provider)
    """
    if len(tools_or_fns) < 2:
        raise ValueError("compose() requires at least 2 tools/functions")

    fns: List[Callable] = []
    tool_names: List[str] = []

    for t in tools_or_fns:
        if isinstance(t, Tool):
            fns.append(t.function)
            tool_names.append(t.name)
        elif callable(t):
            fns.append(t)
            tool_names.append(getattr(t, "__name__", "fn"))
        else:
            raise TypeError(f"compose() expects Tool or callable, got {type(t).__name__}")

    composite_name = name or "_then_".join(tool_names)
    composite_desc = description or f"Executes {' -> '.join(tool_names)} in sequence."

    # The first function's signature determines the composite tool's parameters
    first_fn = fns[0]

    def composite_fn(*args: Any, **kwargs: Any) -> Any:
        result = first_fn(*args, **kwargs)
        for fn in fns[1:]:
            result = fn(result)
        return result

    # Build composite tool
    first_tool = tools_or_fns[0] if isinstance(tools_or_fns[0], Tool) else None
    if first_tool:
        # Create wrapper that accepts the first tool's named params
        def named_composite(**kwargs: Any) -> Any:
            result = first_fn(**kwargs)
            for fn in fns[1:]:
                result = fn(result)
            return result

        return Tool(
            name=composite_name,
            description=composite_desc,
            parameters=first_tool.parameters,
            function=named_composite,
            _skip_validation=True,
        )

    # For plain callables, create a minimal tool
    return Tool(
        name=composite_name,
        description=composite_desc,
        parameters=[],
        function=composite_fn,
    )


__all__ = ["compose"]
