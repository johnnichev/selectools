"""
Tests for tool validation at registration time (v0.5.2 feature).

Tests cover:
- Empty/invalid tool names
- Empty/invalid descriptions
- Duplicate parameter names
- Unsupported parameter types
- Parameter/function signature mismatches
- Required parameters with defaults
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from selectools import Tool, ToolParameter, ToolValidationError, tool

# =============================================================================
# Tool Name Validation Tests
# =============================================================================


class TestToolNameValidation:
    """Test tool name validation."""

    def test_empty_tool_name(self) -> None:
        """Test that empty tool names are rejected."""
        with pytest.raises(ToolValidationError, match="Tool name cannot be empty"):
            Tool(
                name="",
                description="A tool",
                parameters=[],
                function=lambda: "result",
            )

    def test_whitespace_only_tool_name(self) -> None:
        """Test that whitespace-only tool names are rejected."""
        with pytest.raises(ToolValidationError, match="Tool name cannot be empty"):
            Tool(
                name="   ",
                description="A tool",
                parameters=[],
                function=lambda: "result",
            )


# =============================================================================
# Tool Description Validation Tests
# =============================================================================


class TestToolDescriptionValidation:
    """Test tool description validation."""

    def test_empty_description(self) -> None:
        """Test that empty descriptions are rejected."""
        with pytest.raises(ToolValidationError, match="description cannot be empty"):
            Tool(
                name="my_tool",
                description="",
                parameters=[],
                function=lambda: "result",
            )

    def test_whitespace_only_description(self) -> None:
        """Test that whitespace-only descriptions are rejected."""
        with pytest.raises(ToolValidationError, match="description cannot be empty"):
            Tool(
                name="my_tool",
                description="   ",
                parameters=[],
                function=lambda: "result",
            )


# =============================================================================
# Parameter Name Validation Tests
# =============================================================================


class TestParameterNameValidation:
    """Test parameter name validation."""

    def test_duplicate_parameter_names(self) -> None:
        """Test that duplicate parameter names are rejected."""
        with pytest.raises(ToolValidationError, match="Duplicate parameter name"):
            Tool(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="query", param_type=str, description="Query 1"),
                    ToolParameter(name="query", param_type=str, description="Query 2"),
                ],
                function=lambda query: query,
            )

    def test_multiple_duplicates(self) -> None:
        """Test that multiple duplicate names are all reported."""
        with pytest.raises(ToolValidationError, match="Duplicate parameter name"):
            Tool(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="foo", param_type=str, description="Foo 1"),
                    ToolParameter(name="foo", param_type=str, description="Foo 2"),
                    ToolParameter(name="bar", param_type=int, description="Bar 1"),
                    ToolParameter(name="bar", param_type=int, description="Bar 2"),
                ],
                function=lambda foo, bar: f"{foo}{bar}",
            )


# =============================================================================
# Parameter Type Validation Tests
# =============================================================================


class TestParameterTypeValidation:
    """Test parameter type validation."""

    def test_unsupported_parameter_type(self) -> None:
        """Test that unsupported types are rejected."""
        with pytest.raises(ToolValidationError, match="Unsupported parameter type"):

            class CustomType:
                pass

            Tool(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="param", param_type=CustomType, description="Custom param"),
                ],
                function=lambda param: str(param),
            )

    def test_supported_types_are_allowed(self) -> None:
        """Test that all supported types are accepted."""
        # Should not raise
        Tool(
            name="my_tool",
            description="A tool",
            parameters=[
                ToolParameter(name="s", param_type=str, description="String"),
                ToolParameter(name="i", param_type=int, description="Int"),
                ToolParameter(name="f", param_type=float, description="Float"),
                ToolParameter(name="b", param_type=bool, description="Bool"),
                ToolParameter(name="lst", param_type=list, description="List"),
                ToolParameter(name="d", param_type=dict, description="Dict"),
            ],
            function=lambda s, i, f, b, lst, d: "ok",
        )


# =============================================================================
# Function Signature Validation Tests
# =============================================================================


class TestFunctionSignatureValidation:
    """Test function signature validation."""

    def test_parameter_not_in_function_signature(self) -> None:
        """Test that parameters must exist in function signature."""
        with pytest.raises(ToolValidationError, match="not found in function signature"):
            Tool(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="missing_param", param_type=str, description="Missing"),
                ],
                function=lambda: "result",  # No parameters
            )

    def test_required_parameter_with_default_value(self) -> None:
        """Test that required parameters can't have defaults in function."""
        with pytest.raises(ToolValidationError, match="marked as required but has default value"):
            Tool(
                name="my_tool",
                description="A tool",
                parameters=[
                    ToolParameter(name="param", param_type=str, description="Param", required=True),
                ],
                function=lambda param="default": param,  # Has default
            )

    def test_optional_parameter_with_default_is_ok(self) -> None:
        """Test that optional parameters can have defaults."""
        # Should not raise
        Tool(
            name="my_tool",
            description="A tool",
            parameters=[
                ToolParameter(name="param", param_type=str, description="Param", required=False),
            ],
            function=lambda param="default": param,
        )

    def test_injected_kwargs_not_required_in_signature(self) -> None:
        """Test that injected kwargs don't need to be in parameters list."""
        # Should not raise
        Tool(
            name="my_tool",
            description="A tool",
            parameters=[
                ToolParameter(name="user_param", param_type=str, description="User param"),
            ],
            function=lambda user_param, injected_param: f"{user_param}{injected_param}",
            injected_kwargs={"injected_param": "value"},
        )


# =============================================================================
# Decorator Validation Tests
# =============================================================================


class TestDecoratorValidation:
    """Test that @tool decorator also validates."""

    def test_decorator_with_empty_description_uses_docstring(self) -> None:
        """Test that decorator uses docstring if no description provided."""

        @tool()
        def my_tool(param: str) -> str:
            """This is the docstring description."""
            return param

        assert "docstring description" in my_tool.description

    def test_decorator_validates_inferred_parameters(self) -> None:
        """Test that decorator validates auto-inferred parameters."""

        # This should work fine
        @tool(description="A tool")
        def good_tool(param: str) -> str:
            return param

        assert good_tool.name == "good_tool"
        assert len(good_tool.parameters) == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_builtin_functions_skip_signature_validation(self) -> None:
        """Test that built-in functions skip signature validation gracefully."""
        # Built-in functions can't be inspected, so validation should be skipped
        # This test just ensures no crash occurs
        try:
            Tool(
                name="len_tool",
                description="Get length",
                parameters=[
                    ToolParameter(name="obj", param_type=str, description="Object"),
                ],
                function=len,  # Built-in function
            )
        except ToolValidationError:
            # If validation fails for other reasons, that's ok
            pass

    def test_lambda_functions_can_be_validated(self) -> None:
        """Test that lambda functions work with validation."""
        # Should not raise
        Tool(
            name="lambda_tool",
            description="A lambda",
            parameters=[
                ToolParameter(name="x", param_type=int, description="Input"),
            ],
            function=lambda x: str(x * 2),
        )

    def test_async_functions_can_be_validated(self) -> None:
        """Test that async functions work with validation."""

        async def async_func(param: str) -> str:
            return param

        # Should not raise
        Tool(
            name="async_tool",
            description="An async tool",
            parameters=[
                ToolParameter(name="param", param_type=str, description="Param"),
            ],
            function=async_func,
        )


# =============================================================================
# Regression Tests — Ralph Bug Hunt
# =============================================================================


class TestInjectedKwargsNotVisibleToLLM:
    """Regression: @tool decorator with injected_kwargs must exclude injected params from LLM schema."""

    def test_injected_param_not_in_schema(self) -> None:
        """Injected kwargs must not appear in the tool's parameter list (not visible to LLM)."""
        from selectools import tool

        @tool(description="Query something", injected_kwargs={"context": "injected_value"})
        def query_something(query: str, context: str) -> str:
            return f"{query} {context}"

        param_names = [p.name for p in query_something.parameters]
        assert "context" not in param_names, (
            "Injected kwarg 'context' must not appear in LLM-visible parameters"
        )
        assert "query" in param_names, "Non-injected param 'query' must remain visible"

    def test_injected_param_still_used_at_execution(self) -> None:
        """Injected kwargs must still be passed to the function at execution time."""
        from selectools import tool

        @tool(description="Use injected", injected_kwargs={"suffix": "_injected"})
        def add_suffix(text: str, suffix: str) -> str:
            return text + suffix

        result = add_suffix.execute({"text": "hello"})
        assert result == "hello_injected"

    def test_schema_does_not_include_injected_key(self) -> None:
        """Tool schema() must not expose injected kwargs as required parameters."""
        from selectools import tool

        @tool(description="Test", injected_kwargs={"db": "conn"})
        def fetch(query: str, db: str) -> str:
            return query

        schema = fetch.schema()
        assert "db" not in schema["parameters"]["properties"]
        assert "db" not in schema["parameters"]["required"]


class TestRegistryToolCacheableParam:
    """Regression: ToolRegistry.tool() decorator must support cacheable and cache_ttl parameters."""

    def test_registry_tool_accepts_cacheable(self) -> None:
        """ToolRegistry.tool() must accept cacheable=True without TypeError."""
        from selectools.tools.registry import ToolRegistry

        reg = ToolRegistry()

        @reg.tool(description="Expensive call", cacheable=True, cache_ttl=60)
        def expensive(query: str) -> str:
            return f"result: {query}"

        assert expensive.cacheable is True
        assert expensive.cache_ttl == 60

    def test_registry_tool_cacheable_defaults_to_false(self) -> None:
        """ToolRegistry.tool() cacheable defaults to False for backward compatibility."""
        from selectools.tools.registry import ToolRegistry

        reg = ToolRegistry()

        @reg.tool(description="Regular tool")
        def regular(query: str) -> str:
            return query

        assert regular.cacheable is False
        assert regular.cache_ttl == 300


# =============================================================================
# Regression Tests — Ralph Bug Hunt
# =============================================================================


class TestGetTypeHintsFallback:
    """Regression: _infer_parameters_from_callable must not raise NameError for
    unresolvable forward references; it should raise ToolValidationError instead."""

    def test_unresolvable_forward_ref_raises_tool_validation_error(self) -> None:
        """@tool on a function with a bad forward reference raises ToolValidationError,
        not NameError.  Previously get_type_hints() propagated NameError directly."""

        # Use a function defined at test-scope so the forward ref can't resolve
        def _func(x: "AbsolutelyNonExistentType12345XYZ") -> str:  # noqa: F821
            return x

        with pytest.raises((ToolValidationError, Exception)) as exc_info:
            tool(description="test")(_func)  # type: ignore[arg-type]

        # Must NOT be a raw NameError
        assert not isinstance(exc_info.value, NameError), (
            "Expected ToolValidationError (or subclass), got raw NameError. "
            "Fix: wrap get_type_hints() call in try/except in _infer_parameters_from_callable."
        )

    def test_valid_type_hints_still_work_after_fallback_fix(self) -> None:
        """Normal functions with valid type hints are unaffected by the fallback fix."""

        @tool(description="Add two integers")
        def add(a: int, b: int) -> int:
            return a + b

        assert len(add.parameters) == 2
        assert add.parameters[0].param_type is int
        assert add.parameters[1].param_type is int


# =============================================================================
# Regression Tests — Ralph Bug Hunt Pass 3
# =============================================================================


class TestNoneOptionalParamAllowed:
    """Regression: None passed for optional params must be treated as absent (use function default).

    Previously validate() raised ToolValidationError for any None value regardless of
    whether the parameter was required or optional.  execute() then passed None explicitly
    to the function, overriding its default.
    """

    def test_execute_none_optional_uses_default(self) -> None:
        """execute() with None for optional param must use the function's default."""

        def func(query: str, limit: int = 10) -> str:
            return f"query={query} limit={limit}"

        t = Tool(
            name="search",
            description="Search tool",
            parameters=[
                ToolParameter(name="query", param_type=str, description="Query", required=True),
                ToolParameter(name="limit", param_type=int, description="Limit", required=False),
            ],
            function=func,
        )

        result = t.execute({"query": "test", "limit": None})
        assert result == "query=test limit=10", f"Expected default to be used, got: {result}"

    @pytest.mark.asyncio
    async def test_aexecute_none_optional_uses_default(self) -> None:
        """aexecute() with None for optional param must use the function's default."""

        def func(query: str, limit: int = 10) -> str:
            return f"query={query} limit={limit}"

        t = Tool(
            name="search_async",
            description="Search tool async",
            parameters=[
                ToolParameter(name="query", param_type=str, description="Query", required=True),
                ToolParameter(name="limit", param_type=int, description="Limit", required=False),
            ],
            function=func,
        )

        result = await t.aexecute({"query": "hello", "limit": None})
        assert result == "query=hello limit=10", f"Expected default to be used, got: {result}"

    def test_execute_none_required_still_raises(self) -> None:
        """execute() with None for required param must still raise ToolValidationError."""

        def func(query: str) -> str:
            return query

        t = Tool(
            name="require_query",
            description="Requires query",
            parameters=[
                ToolParameter(name="query", param_type=str, description="Query", required=True),
            ],
            function=func,
        )

        with pytest.raises(ToolValidationError):
            t.execute({"query": None})

    def test_validate_none_optional_does_not_raise(self) -> None:
        """validate() with None for optional param must not raise."""

        def func(query: str, limit: int = 5) -> str:
            return query

        t = Tool(
            name="no_raise",
            description="No raise for optional None",
            parameters=[
                ToolParameter(name="query", param_type=str, description="Query", required=True),
                ToolParameter(name="limit", param_type=int, description="Limit", required=False),
            ],
            function=func,
        )

        # Must not raise
        t.validate({"query": "test", "limit": None})


# =============================================================================
# Regression Tests — Ralph Bug Hunt Pass 4
# =============================================================================


class TestVarKeywordFunction:
    """Regression: Tools wrapping **kwargs functions must pass signature validation.

    Previously _validate_tool_definition raised ToolValidationError('not found in
    function signature') for every declared parameter when the underlying function
    used **kwargs, because the per-name lookup never found the bare parameter names.
    """

    def test_kwargs_function_tool_creates_successfully(self) -> None:
        """Tool wrapping a **kwargs function must not raise ToolValidationError."""

        def dispatch(**kwargs: object) -> str:
            return str(kwargs)

        t = Tool(
            name="dispatch",
            description="Dispatches kwargs",
            parameters=[
                ToolParameter(name="action", param_type=str, description="Action"),
                ToolParameter(name="value", param_type=int, description="Value", required=False),
            ],
            function=dispatch,
        )
        assert t.name == "dispatch"
        assert len(t.parameters) == 2

    def test_kwargs_function_tool_executes_correctly(self) -> None:
        """Tool wrapping a **kwargs function must forward parameters at execution time."""

        def dispatch(**kwargs: object) -> str:
            return ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

        t = Tool(
            name="dispatch_exec",
            description="Dispatches kwargs",
            parameters=[
                ToolParameter(name="action", param_type=str, description="Action"),
            ],
            function=dispatch,
        )
        result = t.execute({"action": "ping"})
        assert result == "action=ping"

    def test_kwargs_function_with_required_and_optional_params(self) -> None:
        """Required + optional mix works correctly for **kwargs functions."""

        def multi(**kwargs: object) -> str:
            return str(kwargs)

        t = Tool(
            name="multi_kwargs",
            description="Multi kwargs",
            parameters=[
                ToolParameter(name="q", param_type=str, description="Query"),
                ToolParameter(name="n", param_type=int, description="N", required=False),
            ],
            function=multi,
        )
        result = t.execute({"q": "test", "n": None})
        assert "q" in result
        assert "n" not in result  # None optional should be omitted

    def test_regular_function_still_validates_signature(self) -> None:
        """Non-kwargs functions still raise on unknown parameter names."""

        def regular(x: str) -> str:
            return x

        with pytest.raises(ToolValidationError, match="not found in function signature"):
            Tool(
                name="regular",
                description="Regular",
                parameters=[
                    ToolParameter(name="nonexistent", param_type=str, description="Bad param"),
                ],
                function=regular,
            )


# =============================================================================
# Regression Tests — Ralph Bug Hunt Pass 5
# =============================================================================


class TestVarArgsNotInSchema:
    """Regression: @tool on a function with *args or **kwargs must NOT expose
    those variadic collectors as discrete LLM-visible parameters.

    Previously _infer_parameters_from_callable included VAR_POSITIONAL (*args)
    and VAR_KEYWORD (**kwargs) parameters as required string parameters in the
    inferred schema, which is incorrect — the LLM cannot meaningfully pass
    values to *args or **kwargs collectors.
    """

    def test_tool_varargs_only_produces_empty_schema(self) -> None:
        """@tool on *args-only function must produce empty parameter schema."""
        from selectools import tool

        @tool(description="Accepts anything")
        def accepts_anything(*args: object) -> str:
            return str(args)

        assert accepts_anything.parameters == [], (
            "VAR_POSITIONAL (*args) must not appear in LLM-visible parameters"
        )
        schema = accepts_anything.schema()
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []

    def test_tool_kwargs_only_produces_empty_schema(self) -> None:
        """@tool on **kwargs-only function must produce empty parameter schema."""
        from selectools import tool

        @tool(description="Accepts kwargs")
        def accepts_kwargs(**kwargs: object) -> str:
            return str(kwargs)

        assert accepts_kwargs.parameters == [], (
            "VAR_KEYWORD (**kwargs) must not appear in LLM-visible parameters"
        )

    def test_tool_mixed_regular_and_varargs(self) -> None:
        """Regular params are inferred; *args/**kwargs are silently skipped."""
        from selectools import tool

        @tool(description="Mixed")
        def mixed(query: str, limit: int = 10, *args: object, **kwargs: object) -> str:
            return query

        param_names = [p.name for p in mixed.parameters]
        assert "query" in param_names
        assert "limit" in param_names
        assert "args" not in param_names, "*args must not be inferred as a parameter"
        assert "kwargs" not in param_names, "**kwargs must not be inferred as a parameter"
        assert len(mixed.parameters) == 2

    def test_infer_parameters_skips_var_positional(self) -> None:
        """_infer_parameters_from_callable must skip VAR_POSITIONAL parameters."""
        from selectools.tools.decorators import _infer_parameters_from_callable

        def func(*args: str) -> str:
            return str(args)

        params = _infer_parameters_from_callable(func)
        assert params == [], f"Expected empty list, got {params}"

    def test_infer_parameters_skips_var_keyword(self) -> None:
        """_infer_parameters_from_callable must skip VAR_KEYWORD parameters."""
        from selectools.tools.decorators import _infer_parameters_from_callable

        def func(**kwargs: str) -> str:
            return str(kwargs)

        params = _infer_parameters_from_callable(func)
        assert params == [], f"Expected empty list, got {params}"


# =============================================================================
# Regression Tests — Ralph Bug Hunt Pass 7
# =============================================================================


class TestGenericCollectionTypeAnnotations:
    """Regression: @tool on functions with List[T] / Dict[K,V] / Optional[List[T]]
    annotations must resolve to the bare ``list`` / ``dict`` types.

    Previously _unwrap_type returned ``typing.List[str]`` (or ``list[str]`` on 3.9+)
    after stripping Optional, which caused ToolValidationError: "Unsupported parameter
    type: typing.List[str]" — even though the user intended a plain list parameter.
    """

    def test_optional_list_str_creates_tool(self) -> None:
        """@tool with Optional[List[str]] param must create tool with param_type=list."""

        @tool(description="Accept optional list")
        def accept_list(items: Optional[List[str]] = None) -> str:
            return str(items)

        assert len(accept_list.parameters) == 1
        assert accept_list.parameters[0].param_type is list
        assert accept_list.parameters[0].required is False

    def test_optional_dict_creates_tool(self) -> None:
        """@tool with Optional[Dict[str, Any]] param must create tool with param_type=dict."""

        @tool(description="Accept optional dict")
        def accept_dict(config: Optional[Dict[str, Any]] = None) -> str:
            return str(config)

        assert len(accept_dict.parameters) == 1
        assert accept_dict.parameters[0].param_type is dict
        assert accept_dict.parameters[0].required is False

    def test_plain_list_str_creates_tool(self) -> None:
        """@tool with List[str] (no Optional) param must create tool with param_type=list."""

        @tool(description="Require list")
        def require_list(items: List[str]) -> str:
            return str(items)

        assert len(require_list.parameters) == 1
        assert require_list.parameters[0].param_type is list
        assert require_list.parameters[0].required is True

    def test_plain_dict_str_creates_tool(self) -> None:
        """@tool with Dict[str, int] param must create tool with param_type=dict."""

        @tool(description="Require dict")
        def require_dict(mapping: Dict[str, int]) -> str:
            return str(mapping)

        assert len(require_dict.parameters) == 1
        assert require_dict.parameters[0].param_type is dict

    def test_native_list_generic_creates_tool(self) -> None:
        """@tool with list[str] (Python 3.9+ native syntax) must create tool with param_type=list."""

        @tool(description="Native list annotation")
        def native_list(items: list[str]) -> str:  # type: ignore[type-arg]
            return str(items)

        assert native_list.parameters[0].param_type is list

    def test_native_dict_generic_creates_tool(self) -> None:
        """@tool with dict[str, int] (Python 3.9+ native syntax) must produce param_type=dict."""

        @tool(description="Native dict annotation")
        def native_dict(mapping: dict[str, int]) -> str:  # type: ignore[type-arg]
            return str(mapping)

        assert native_dict.parameters[0].param_type is dict

    def test_optional_non_collection_still_rejects_unsupported(self) -> None:
        """Optional[datetime] still raises ToolValidationError (datetime is not supported)."""
        from datetime import datetime

        with pytest.raises(ToolValidationError, match="Unsupported parameter type"):

            @tool(description="datetime param")
            def with_datetime(dt: Optional[datetime] = None) -> str:
                return str(dt)

    def test_schema_exposes_list_type_for_generic_annotation(self) -> None:
        """Tool schema must expose 'array' JSON type for List[str] params."""

        @tool(description="Schema check")
        def schema_tool(items: List[str]) -> str:
            return str(items)

        schema = schema_tool.schema()
        assert schema["parameters"]["properties"]["items"]["type"] == "array"

    def test_tool_executes_correctly_with_list_annotation(self) -> None:
        """Tool with List[str] annotation must execute and pass the value through."""

        @tool(description="Execute list tool")
        def list_tool(items: List[str]) -> str:
            return ",".join(items)

        result = list_tool.execute({"items": ["a", "b", "c"]})
        assert result == "a,b,c"
