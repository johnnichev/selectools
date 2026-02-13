"""
Registry for managing and discovering tools.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .base import ParamMetadata, Tool
from .decorators import tool


class ToolRegistry:
    """
    Central registry for managing a collection of tools.

    Allows registering tools via method calls or decorators, and provides
    access to all registered tools for the Agent.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_instance: Tool) -> None:
        """
        Register a pre-existing Tool instance.

        Args:
            tool_instance: The Tool object to register.
        """
        if tool_instance.name in self._tools:
            # We overwrite existing tools with the same name mostly silently,
            # but in a real system we might want to log a warning.
            pass
        self._tools[tool_instance.name] = tool_instance

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return all registered tools as a list."""
        return list(self._tools.values())

    def all(self) -> List[Tool]:
        """Alias for list_tools (backward compatibility)."""
        return self.list_tools()

    def tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        param_metadata: Optional[Dict[str, ParamMetadata]] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
        config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
        streaming: bool = False,
    ) -> Callable[[Callable[..., Any]], Tool]:
        """
        Decorator to register a function as a tool in this registry.

        Args:
            name: Optional custom name.
            description: Optional description.
            param_metadata: Metadata for parameters.
            injected_kwargs: Dependency injection kwargs.
            config_injector: Dependency injection callable.
            streaming: Whether tool is streaming.

        Returns:
            Decorator that returns the registered Tool instance.
        """

        def decorator(func: Callable[..., Any]) -> Tool:
            # Use the standalone tool decorator to create the instance
            tool_instance = tool(
                name=name,
                description=description,
                param_metadata=param_metadata,
                injected_kwargs=injected_kwargs,
                config_injector=config_injector,
                streaming=streaming,
            )(func)

            # Register it
            self.register(tool_instance)
            return tool_instance

        return decorator
