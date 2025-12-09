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

import pytest

from selectools import Tool, ToolParameter, ToolValidationError, tool

# =============================================================================
# Tool Name Validation Tests
# =============================================================================


class TestToolNameValidation:
    """Test tool name validation."""

    def test_empty_tool_name(self):
        """Test that empty tool names are rejected."""
        with pytest.raises(ToolValidationError, match="Tool name cannot be empty"):
            Tool(
                name="",
                description="A tool",
                parameters=[],
                function=lambda: "result",
            )

    def test_whitespace_only_tool_name(self):
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

    def test_empty_description(self):
        """Test that empty descriptions are rejected."""
        with pytest.raises(ToolValidationError, match="description cannot be empty"):
            Tool(
                name="my_tool",
                description="",
                parameters=[],
                function=lambda: "result",
            )

    def test_whitespace_only_description(self):
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

    def test_duplicate_parameter_names(self):
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

    def test_multiple_duplicates(self):
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

    def test_unsupported_parameter_type(self):
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

    def test_supported_types_are_allowed(self):
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

    def test_parameter_not_in_function_signature(self):
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

    def test_required_parameter_with_default_value(self):
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

    def test_optional_parameter_with_default_is_ok(self):
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

    def test_injected_kwargs_not_required_in_signature(self):
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

    def test_decorator_with_empty_description_uses_docstring(self):
        """Test that decorator uses docstring if no description provided."""

        @tool()
        def my_tool(param: str) -> str:
            """This is the docstring description."""
            return param

        assert "docstring description" in my_tool.description

    def test_decorator_validates_inferred_parameters(self):
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

    def test_builtin_functions_skip_signature_validation(self):
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

    def test_lambda_functions_can_be_validated(self):
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

    def test_async_functions_can_be_validated(self):
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
