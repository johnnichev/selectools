"""
Tool metadata, schemas, and runtime validation.
"""

from __future__ import annotations

import asyncio
import contextvars
import difflib
import functools
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..exceptions import ToolExecutionError, ToolValidationError
from ..stability import stable

JsonSchema = Dict[str, Any]

# Shared executor for running async tools from sync contexts.
# Created lazily; avoids per-call thread overhead.
_ASYNC_TOOL_EXECUTOR: Optional[ThreadPoolExecutor] = None
_ASYNC_TOOL_EXECUTOR_LOCK = threading.Lock()


def _get_async_tool_executor() -> ThreadPoolExecutor:
    global _ASYNC_TOOL_EXECUTOR
    if _ASYNC_TOOL_EXECUTOR is None:
        with _ASYNC_TOOL_EXECUTOR_LOCK:
            if _ASYNC_TOOL_EXECUTOR is None:
                _ASYNC_TOOL_EXECUTOR = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="selectools_async_tool"
                )
    return _ASYNC_TOOL_EXECUTOR


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


@stable
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


@stable
class Tool:
    """
    Encapsulates a callable tool with validation and schema generation.

    A Tool wraps a Python function and adds metadata, parameter validation,
    and schema generation capabilities. Tools can be sync or async functions,
    and can optionally stream results progressively.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does (used by LLM).
        parameters: List of ToolParameter objects defining expected inputs.
        function: The underlying Python function to execute.
        injected_kwargs: Additional kwargs to pass to the function (not visible to LLM).
        config_injector: Optional callable that returns additional kwargs at execution time.
        is_async: Whether the underlying function is async (detected automatically).
        streaming: Whether the tool yields results progressively (Generator or AsyncGenerator).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[..., Any],
        *,
        injected_kwargs: Optional[Dict[str, Any]] = None,
        config_injector: Optional[Callable[[], Dict[str, Any]]] = None,
        streaming: bool = False,
        screen_output: bool = False,
        terminal: bool = False,
        requires_approval: bool = False,
        cacheable: bool = False,
        cache_ttl: int = 300,
        _skip_validation: bool = False,
    ):
        """
        Initialize a new Tool.

        Args:
            name: Unique identifier for the tool.
            description: Description of what the tool does (shown to the LLM).
            parameters: List of ToolParameter definitions.
            function: Callable that implements the tool logic (must return str or Generator).
            injected_kwargs: Optional kwargs injected at execution (hidden from LLM).
            config_injector: Optional callable returning kwargs to inject at execution time.
            streaming: Whether this tool yields results progressively (returns Generator).
            screen_output: Screen this tool's output for prompt injection before
                feeding it back to the LLM.  Default: ``False``.
            terminal: If True, executing this tool stops the agent loop and
                returns the tool result as the final response.  Default: ``False``.
            requires_approval: If True, the tool always requires human approval
                before execution, regardless of ToolPolicy rules.  Default: ``False``.
            cacheable: If True, tool results are cached by name + args when
                the agent has a cache configured.  Default: ``False``.
            cache_ttl: Time-to-live in seconds for cached results.
                Default: ``300`` (5 minutes).

        Raises:
            ToolValidationError: If tool definition is invalid
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.injected_kwargs = injected_kwargs or {}
        self.config_injector = config_injector
        self.is_async = inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(
            function
        )
        self.streaming = streaming
        self.screen_output = screen_output
        self.terminal = terminal
        self.requires_approval = requires_approval
        self.cacheable = cacheable
        self.cache_ttl = cache_ttl

        # Validate tool definition at registration time
        if not _skip_validation:
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

        # If the function accepts **kwargs it will accept any named parameter —
        # skip the per-parameter existence check entirely.
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in func_params.values())

        # Check that all tool parameters exist in function signature
        if not has_var_keyword:
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

        # Validate streaming tool return type (informational warning, not enforced)
        if self.streaming:
            # Check if function has proper return type hint for streaming
            sig = inspect.signature(self.function)
            return_annotation = sig.return_annotation

            # For now, just log a note if streaming is set but no Generator annotation
            # This is informational only, not strictly enforced
            if return_annotation != inspect.Signature.empty:
                type_str = str(return_annotation)
                has_generator = (
                    "Generator" in type_str
                    or "AsyncGenerator" in type_str
                    or "Iterable" in type_str
                )
                if not has_generator:
                    # Don't raise error, just note it - generators work even without type hints
                    pass

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

        if isinstance(value, bool) and param.param_type in (int, float):
            return (
                f"Expected {param.param_type.__name__} for '{param.name}', got bool. "
                f"Pass an integer or float value instead."
            )

        if param.param_type is float:
            if not isinstance(value, (float, int)):
                return f"Parameter '{param.name}' must be a number"
            return None

        if not isinstance(value, param.param_type):
            return f"Parameter '{param.name}' must be of type {param.param_type.__name__}, got {type(value).__name__}"
        return None

    def _coerce_value(
        self, param: ToolParameter, value: ParameterValue
    ) -> Tuple[ParameterValue, Optional[str]]:
        """Attempt safe coercion of a parameter value to its declared type.

        Returns a (coerced_value, error) tuple. ``error`` is ``None`` on
        success. Coercion is only attempted from ``str`` to primitive types
        (``int``, ``float``, ``bool``); other type mismatches fall through
        unchanged so the existing validation path can report them.

        BUG-10: Some LLMs (especially smaller local models via Ollama) emit
        numeric tool arguments as JSON strings. Without this coercion the
        agent would reject perfectly recoverable values.
        """
        if value is None:
            return value, None
        if isinstance(value, param.param_type) and not (
            isinstance(value, bool) and param.param_type in (int, float)
        ):
            return value, None
        if not isinstance(value, str):
            return value, None
        if param.param_type is bool:
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True, None
            if lowered in ("false", "0", "no", "off"):
                return False, None
            return value, (f"Cannot coerce {value!r} to bool for parameter '{param.name}'")
        if param.param_type is int:
            try:
                return int(value), None
            except (ValueError, TypeError) as exc:
                return value, (
                    f"Cannot coerce {value!r} to int for parameter '{param.name}': {exc}"
                )
        if param.param_type is float:
            try:
                return float(value), None
            except (ValueError, TypeError) as exc:
                return value, (
                    f"Cannot coerce {value!r} to float for parameter '{param.name}': {exc}"
                )
        return value, None

    @property
    def is_streaming(self) -> bool:
        """Return whether this tool streams results progressively."""
        return self.streaming

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

            # Optional params that are explicitly None should be treated as absent —
            # they will be omitted from call_args so the function default is used.
            if not param.required and params[param.name] is None:
                continue

            # BUG-10: Attempt safe coercion (str -> int/float/bool) before
            # validation. Smaller LLMs sometimes emit numeric tool arguments
            # as JSON strings; coercing here lets the tool execute normally.
            coerced, coerce_error = self._coerce_value(param, params[param.name])
            if coerce_error:
                raise ToolValidationError(
                    tool_name=self.name,
                    param_name=param.name,
                    issue=coerce_error,
                    suggestion=f"Expected type: {param.param_type.__name__}",
                )
            params[param.name] = coerced

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

    def _serialize_result(self, result: Any) -> str:
        """Serialize a tool result to a string suitable for the LLM.

        Dicts, lists, Pydantic models, and dataclasses are serialized as JSON.
        Strings pass through unchanged.  All other types fall back to ``str()``.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)
        if hasattr(result, "model_dump"):  # Pydantic v2
            return json.dumps(result.model_dump(), default=str)
        if hasattr(result, "__dataclass_fields__"):  # dataclass
            from dataclasses import asdict

            return json.dumps(asdict(result), default=str)
        return str(result)

    def execute(
        self,
        params: Dict[str, ParameterValue],
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Validate parameters then execute the underlying callable.

        Args:
            params: Parameter values to pass to the tool.
            chunk_callback: Optional callback to call for each chunk if tool is streaming.

        Returns:
            String result from the tool (or accumulated chunks if streaming).
        """
        self.validate(params)

        # Build call args, omitting None values for optional params so that
        # the function's own default is used (mirrors validate() behaviour).
        optional_none = {
            p.name for p in self.parameters if not p.required and params.get(p.name) is None
        }
        call_args: Dict[str, Any] = {k: v for k, v in params.items() if k not in optional_none}
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        try:
            result = self.function(**call_args)

            # Handle async generators called from sync context
            if inspect.isasyncgen(result):

                async def _drain_async_gen() -> str:
                    chunks: list = []
                    async for chunk in result:  # type: ignore[union-attr]
                        chunk_str = str(chunk)
                        chunks.append(chunk_str)
                        if chunk_callback:
                            chunk_callback(chunk_str)
                    return "".join(chunks)

                try:
                    _loop = asyncio.get_running_loop()
                except RuntimeError:
                    _loop = None
                if _loop and _loop.is_running():
                    return (
                        _get_async_tool_executor().submit(asyncio.run, _drain_async_gen()).result()
                    )
                else:
                    return asyncio.run(_drain_async_gen())

            # Handle async functions called from sync context
            if inspect.iscoroutine(result):
                try:
                    _loop = asyncio.get_running_loop()
                except RuntimeError:
                    _loop = None
                if _loop and _loop.is_running():
                    result = _get_async_tool_executor().submit(asyncio.run, result).result()
                else:
                    result = asyncio.run(result)

            # Handle streaming tools (generators)
            if inspect.isgenerator(result):
                chunks = []
                for chunk in result:
                    chunk_str = str(chunk)
                    chunks.append(chunk_str)
                    if chunk_callback:
                        chunk_callback(chunk_str)
                return "".join(chunks)

            return self._serialize_result(result)
        except Exception as exc:
            raise ToolExecutionError(tool_name=self.name, error=exc, params=params) from exc

    async def aexecute(
        self,
        params: Dict[str, ParameterValue],
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Async version of execute().

        If the tool function is async, it will be awaited. If it's sync,
        it will be run in a thread pool executor to avoid blocking.

        Args:
            params: Parameter values to pass to the tool.
            chunk_callback: Optional callback to call for each chunk if tool is streaming.

        Returns:
            String result from the tool (or accumulated chunks if streaming).
        """
        self.validate(params)

        # Build call args, omitting None values for optional params so that
        # the function's own default is used (mirrors validate() behaviour).
        optional_none = {
            p.name for p in self.parameters if not p.required and params.get(p.name) is None
        }
        call_args: Dict[str, Any] = {k: v for k, v in params.items() if k not in optional_none}
        call_args.update(self.injected_kwargs)
        if self.config_injector:
            call_args.update(self.config_injector() or {})

        try:
            if self.is_async:
                # Call the async function
                result = self.function(**call_args)

                # Check if it's an async generator (returns AsyncGenerator immediately)
                if inspect.isasyncgen(result):
                    chunks = []
                    async for chunk in result:
                        chunk_str = str(chunk)
                        chunks.append(chunk_str)
                        if chunk_callback:
                            chunk_callback(chunk_str)
                    return "".join(chunks)

                # Otherwise it's a regular async function, await it
                result = await result  # type: ignore[misc]
                return self._serialize_result(result)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_running_loop()
                context = contextvars.copy_context()
                func_with_args = functools.partial(self.function, **call_args)

                result = await loop.run_in_executor(None, context.run, func_with_args)

                # Handle sync streaming in async context
                if inspect.isgenerator(result):
                    chunks = []
                    for chunk in result:
                        chunk_str = str(chunk)
                        chunks.append(chunk_str)
                        if chunk_callback:
                            chunk_callback(chunk_str)
                    return "".join(chunks)

                return self._serialize_result(result)
        except Exception as exc:
            raise ToolExecutionError(tool_name=self.name, error=exc, params=params) from exc
