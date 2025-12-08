"""
Tool metadata, schemas, and runtime validation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


JsonSchema = Dict[str, Any]
ParameterValue = Union[str, int, float, bool, dict, list]
ParamMetadata = Dict[str, Any]


def _python_type_to_json(param_type: type) -> str:
    """Map a Python type to a JSON schema type string."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(param_type, "string")


@dataclass
class ToolParameter:
    """Schema definition for a single tool parameter."""

    name: str
    param_type: type
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

    def to_schema(self) -> JsonSchema:
        """Return a JSON-schema compatible definition."""
        schema: JsonSchema = {
            "type": _python_type_to_json(self.param_type),
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


class Tool:
    """
    Encapsulates a callable tool with validation and schema generation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[..., str],
        *,
        injected_kwargs: Optional[Dict[str, Any]] = None,
        config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.injected_kwargs = injected_kwargs or {}
        self.config_injector = config_injector
        self.is_async = inspect.iscoroutinefunction(function)

    def schema(self) -> JsonSchema:
        """Return a JSON-schema style dict describing this tool."""
        properties = {param.name: param.to_schema() for param in self.parameters}
        required = [param.name for param in self.parameters if param.required]

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _validate_single(self, param: ToolParameter, value: ParameterValue) -> Optional[str]:
        """Validate a single parameter, returning an error message if invalid."""
        if value is None:
            return f"Parameter '{param.name}' is None"

        if param.param_type is float:
            if not isinstance(value, (float, int)):
                return f"Parameter '{param.name}' must be a number"
            return None

        if not isinstance(value, param.param_type):
            return f"Parameter '{param.name}' must be of type {param.param_type.__name__}, got {type(value).__name__}"
        return None

    def validate(self, params: Dict[str, ParameterValue]) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter dictionary against this tool's schema.

        Returns (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
            if param.name not in params:
                continue
            error = self._validate_single(param, params[param.name])
            if error:
                return False, error
        return True, None

    def execute(self, params: Dict[str, ParameterValue]) -> str:
        """Validate parameters then execute the underlying callable."""
        is_valid, error = self.validate(params)
        if not is_valid:
            raise ValueError(f"Invalid parameters for tool '{self.name}': {error}")

        call_args: Dict[str, Any] = dict(params)
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        return self.function(**call_args)

    async def aexecute(self, params: Dict[str, ParameterValue]) -> str:
        """
        Async version of execute().
        
        If the tool function is async, it will be awaited. If it's sync,
        it will be run in a thread pool executor to avoid blocking.
        """
        is_valid, error = self.validate(params)
        if not is_valid:
            raise ValueError(f"Invalid parameters for tool '{self.name}': {error}")

        call_args: Dict[str, Any] = dict(params)
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        if self.is_async:
            # Directly await async function
            return await self.function(**call_args)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: self.function(**call_args)
                )
            return result


def _infer_parameters_from_callable(
    func: Callable[..., Any],
    param_metadata: Optional[Dict[str, ParamMetadata]] = None,
) -> List[ToolParameter]:
    """Create ToolParameter objects from a callable signature and annotations."""
    param_metadata = param_metadata or {}
    signature = inspect.signature(func)
    parameters: List[ToolParameter] = []
    for name, param in signature.parameters.items():
        if name.startswith("_"):
            continue
        annotation = param.annotation if param.annotation is not inspect._empty else str
        meta = param_metadata.get(name, {})
        description = meta.get("description", "")
        enum = meta.get("enum")
        required = param.default is inspect._empty
        parameters.append(
            ToolParameter(
                name=name,
                param_type=annotation if isinstance(annotation, type) else str,
                description=description or f"Parameter '{name}'",
                required=required,
                enum=enum,
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
):
    """
    Decorator to register a function as a Tool with schema inference.

    Example:
        @tool(name="search")
        def search(query: str, count: int = 3) -> str:
            ...
    """

    def wrapper(func: Callable[..., str]) -> Tool:
        params = _infer_parameters_from_callable(func, param_metadata=param_metadata)
        tool_obj = Tool(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip() or f"Tool {func.__name__}",
            parameters=params,
            function=func,
            injected_kwargs=injected_kwargs,
            config_injector=config_injector,
        )
        return tool_obj

    return wrapper


class ToolRegistry:
    """Simple registry for reusable tool instances."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_obj: Tool) -> Tool:
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        param_metadata: Optional[Dict[str, ParamMetadata]] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
        config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """Decorator variant that also registers the tool in this registry."""

        def decorator(func: Callable[..., str]) -> Tool:
            tool_obj = tool(
                name=name,
                description=description,
                param_metadata=param_metadata,
                injected_kwargs=injected_kwargs,
                config_injector=config_injector,
            )(func)
            self.register(tool_obj)
            return tool_obj

        return decorator

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())


__all__ = ["Tool", "ToolParameter", "ToolRegistry", "tool"]
