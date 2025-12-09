"""
Tool metadata, schemas, and runtime validation.
"""

from __future__ import annotations

import asyncio
import difflib
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .exceptions import ToolExecutionError, ToolValidationError

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
    """
    Schema definition for a single tool parameter.

    Defines the expected type, description, and validation rules for a tool parameter.
    This information is used to generate JSON schemas for LLMs and validate
    parameters at runtime.

    Attributes:
        name: Parameter name (should match the function parameter name).
        param_type: Python type (str, int, float, bool, list, dict).
        description: Human-readable description explaining what the parameter does.
        required: Whether this parameter must be provided (default: True).
        enum: Optional list of allowed string values for enumerated parameters.

    Example:
        >>> param = ToolParameter(
        ...     name="temperature",
        ...     param_type=float,
        ...     description="Temperature in degrees Celsius",
        ...     required=True
        ... )
        >>>
        >>> # Enum parameter
        >>> param = ToolParameter(
        ...     name="units",
        ...     param_type=str,
        ...     description="Temperature units",
        ...     required=False,
        ...     enum=["celsius", "fahrenheit"]
        ... )
    """

    name: str
    param_type: type
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

    def to_schema(self) -> JsonSchema:
        """
        Convert parameter definition to JSON Schema format.

        Returns:
            Dictionary containing JSON Schema-compatible parameter definition
            including type, description, and enum if specified.
        """
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

    A Tool wraps a Python function and adds metadata, parameter validation,
    and schema generation capabilities. Tools can be sync or async functions.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does (used by LLM).
        parameters: List of ToolParameter objects defining expected inputs.
        function: The underlying Python function to execute.
        injected_kwargs: Additional kwargs to pass to the function (not visible to LLM).
        config_injector: Optional callable that returns additional kwargs at execution time.
        is_async: Whether the underlying function is async (detected automatically).

    Example:
        >>> def get_weather(location: str, units: str = "celsius") -> str:
        ...     return f"Weather in {location}: 72°{units[0].upper()}"
        >>>
        >>> tool = Tool(
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     parameters=[
        ...         ToolParameter(name="location", param_type=str, description="City name", required=True),
        ...         ToolParameter(name="units", param_type=str, description="celsius or fahrenheit", required=False),
        ...     ],
        ...     function=get_weather
        ... )
        >>>
        >>> # Execute with validation
        >>> result = tool.execute({"location": "Paris", "units": "celsius"})
        >>> print(result)
        Weather in Paris: 72°C
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
        """
        Initialize a new Tool.

        Args:
            name: Unique identifier for the tool.
            description: Description of what the tool does (shown to the LLM).
            parameters: List of ToolParameter definitions.
            function: Callable that implements the tool logic (must return str).
            injected_kwargs: Optional kwargs injected at execution (hidden from LLM).
            config_injector: Optional callable returning kwargs to inject at execution time.

        Raises:
            ToolValidationError: If tool definition is invalid
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.injected_kwargs = injected_kwargs or {}
        self.config_injector = config_injector
        self.is_async = inspect.iscoroutinefunction(function)

        # Validate tool definition at registration time
        self._validate_tool_definition()

    def _validate_tool_definition(self) -> None:
        """
        Validate tool definition at registration time to catch errors early.

        Raises:
            ToolValidationError: If the tool definition is invalid
        """
        # Check for empty name
        if not self.name or not self.name.strip():
            raise ToolValidationError(
                tool_name="<unnamed>",
                param_name="name",
                issue="Tool name cannot be empty",
                suggestion="Provide a descriptive name for the tool",
            )

        # Check for empty or missing description
        if not self.description or not self.description.strip():
            raise ToolValidationError(
                tool_name=self.name,
                param_name="description",
                issue="Tool description cannot be empty",
                suggestion="Provide a clear description explaining what the tool does",
            )

        # Check for duplicate parameter names
        param_names = [p.name for p in self.parameters]
        duplicates = [name for name in param_names if param_names.count(name) > 1]
        if duplicates:
            unique_duplicates = sorted(set(duplicates))
            raise ToolValidationError(
                tool_name=self.name,
                param_name=", ".join(unique_duplicates),
                issue="Duplicate parameter name(s)",
                suggestion="Each parameter must have a unique name",
            )

        # Validate parameter types
        supported_types = {str, int, float, bool, list, dict}
        for param in self.parameters:
            if param.param_type not in supported_types:
                type_list = ", ".join(t.__name__ for t in supported_types)
                raise ToolValidationError(
                    tool_name=self.name,
                    param_name=param.name,
                    issue=f"Unsupported parameter type: {param.param_type}",
                    suggestion=f"Use one of: {type_list}",
                )

        # Validate function signature matches parameters
        try:
            sig = inspect.signature(self.function)
        except (ValueError, TypeError):
            # Can't inspect signature (built-in function, etc.)
            return

        func_params = sig.parameters
        param_names_set = {p.name for p in self.parameters}
        injected_names = set(self.injected_kwargs.keys())

        # Check that all tool parameters exist in function signature
        for param in self.parameters:
            if param.name not in func_params and param.name not in injected_names:
                func_param_names = [p for p in func_params.keys() if p not in injected_names]
                suggestion = f"Available function parameters: {', '.join(func_param_names)}"
                if not func_param_names:
                    suggestion = "Function has no parameters"

                raise ToolValidationError(
                    tool_name=self.name,
                    param_name=param.name,
                    issue=f"Parameter '{param.name}' not found in function signature",
                    suggestion=suggestion,
                )

        # Check that required tool parameters don't have defaults in function
        for param in self.parameters:
            if param.required and param.name in func_params:
                func_param = func_params[param.name]
                if func_param.default != inspect.Parameter.empty:
                    raise ToolValidationError(
                        tool_name=self.name,
                        param_name=param.name,
                        issue=(
                            f"Parameter marked as required but has default value "
                            f"in function: {func_param.default!r}"
                        ),
                        suggestion="Either mark as optional (required=False) or remove default from function",
                    )

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

    def validate(self, params: Dict[str, ParameterValue]) -> None:
        """
        Validate a parameter dictionary against this tool's schema.

        Raises ToolValidationError if validation fails with helpful suggestions.
        """
        expected_params = {p.name for p in self.parameters}
        provided_params = set(params.keys())
        extra_params = provided_params - expected_params

        # Check for unexpected parameters (possible typos)
        if extra_params:
            suggestions = []
            for extra in extra_params:
                # Find closest match using difflib
                matches = difflib.get_close_matches(extra, expected_params, n=1, cutoff=0.6)
                if matches:
                    suggestions.append(f"'{extra}' -> Did you mean '{matches[0]}'?")
                else:
                    suggestions.append(f"'{extra}' is not a valid parameter")

            expected_list = ", ".join(f"'{p}'" for p in sorted(expected_params))
            raise ToolValidationError(
                tool_name=self.name,
                param_name=", ".join(sorted(extra_params)),
                issue="Unexpected parameter(s)",
                suggestion=f"{'; '.join(suggestions)}\nExpected parameters: {expected_list}",
            )

        # Check for missing required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                expected_list = ", ".join(f"'{p.name}'" for p in self.parameters if p.required)
                raise ToolValidationError(
                    tool_name=self.name,
                    param_name=param.name,
                    issue="Missing required parameter",
                    suggestion=f"Required parameters: {expected_list}",
                )

            if param.name not in params:
                continue

            # Validate parameter type
            error = self._validate_single(param, params[param.name])
            if error:
                # Provide helpful type conversion suggestions
                value = params[param.name]
                type_hint = ""
                if param.param_type is str and not isinstance(value, str):
                    type_hint = f"Try: {param.name}=str({repr(value)})"
                elif param.param_type is int and isinstance(value, str):
                    type_hint = f"Try: {param.name}=int('{value}')"
                elif param.param_type is float and isinstance(value, (str, int)):
                    type_hint = f"Try: {param.name}=float({repr(value)})"

                raise ToolValidationError(
                    tool_name=self.name,
                    param_name=param.name,
                    issue=error,
                    suggestion=(
                        type_hint if type_hint else f"Expected type: {param.param_type.__name__}"
                    ),
                )

    def execute(self, params: Dict[str, ParameterValue]) -> str:
        """Validate parameters then execute the underlying callable."""
        self.validate(params)

        call_args: Dict[str, Any] = dict(params)
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        try:
            return self.function(**call_args)
        except Exception as exc:
            raise ToolExecutionError(tool_name=self.name, error=exc, params=params) from exc

    async def aexecute(self, params: Dict[str, ParameterValue]) -> str:
        """
        Async version of execute().

        If the tool function is async, it will be awaited. If it's sync,
        it will be run in a thread pool executor to avoid blocking.
        """
        self.validate(params)

        call_args: Dict[str, Any] = dict(params)
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        try:
            if self.is_async:
                # Directly await async function
                return await self.function(**call_args)  # type: ignore[misc]
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor, lambda: self.function(**call_args)
                    )
                return result
        except Exception as exc:
            raise ToolExecutionError(tool_name=self.name, error=exc, params=params) from exc


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
    Decorator to register a function as a Tool with automatic schema inference.

    Automatically extracts parameter types from function signature and creates
    a Tool instance. Parameter descriptions can be provided via param_metadata.

    Args:
        name: Optional tool name (defaults to function name).
        description: Optional description (defaults to function docstring).
        param_metadata: Optional dict mapping parameter names to metadata dicts
                       containing 'description' and optionally 'enum'.
        injected_kwargs: Optional kwargs to inject at execution (hidden from LLM).
        config_injector: Optional callable returning kwargs to inject at execution.

    Returns:
        A Tool instance that wraps the decorated function.

    Example:
        >>> @tool(
        ...     name="search",
        ...     description="Search the web",
        ...     param_metadata={
        ...         "query": {"description": "Search terms"},
        ...         "limit": {"description": "Max results"},
        ...     }
        ... )
        ... def search(query: str, limit: int = 10) -> str:
        ...     return f"Found results for: {query}"
        >>>
        >>> # Use with an agent
        >>> agent = Agent(tools=[search], provider=provider)

        >>> # Async tools work too
        ... @tool(name="fetch_data")
        ... async def fetch_data(url: str) -> str:
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as resp:
        ...             return await resp.text()
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
    """
    Simple registry for organizing and reusing tool instances.

    Useful for managing large collections of tools across multiple agents
    or sharing tools between different parts of an application.

    Example:
        >>> registry = ToolRegistry()
        >>>
        >>> # Register tools
        >>> @registry.tool(name="add")
        ... def add(a: int, b: int) -> str:
        ...     return str(a + b)
        >>>
        >>> @registry.tool(name="multiply")
        ... def multiply(a: int, b: int) -> str:
        ...     return str(a * b)
        >>>
        >>> # Use all registered tools in an agent
        >>> agent = Agent(tools=registry.all(), provider=provider)
        >>>
        >>> # Or get a specific tool
        >>> add_tool = registry.get("add")
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_obj: Tool) -> Tool:
        """
        Register a tool instance in the registry.

        Args:
            tool_obj: The Tool instance to register.

        Returns:
            The same tool instance (for chaining).

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
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
