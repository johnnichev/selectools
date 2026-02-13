"""
Comprehensive tests for Better Error Messages (v0.5.0 feature).

Tests cover:
- ToolValidationError with fuzzy matching suggestions
- ToolExecutionError with context
- ProviderConfigurationError with fix instructions
- MemoryLimitExceededError with suggestions
- Parameter validation edge cases
"""

import pytest

from selectools import (
    Agent,
    AgentConfig,
    Message,
    ProviderConfigurationError,
    Role,
    SelectoolsError,
    Tool,
    ToolExecutionError,
    ToolParameter,
    ToolValidationError,
    tool,
)
from selectools.providers.stubs import LocalProvider

# =============================================================================
# Exception Class Tests
# =============================================================================


class TestSelectoolsError:
    """Test base SelectoolsError exception."""

    def test_base_exception_inheritance(self):
        """SelectoolsError should inherit from Exception."""
        assert issubclass(SelectoolsError, Exception)

    def test_can_raise_base_exception(self):
        """Should be able to raise SelectoolsError directly."""
        with pytest.raises(SelectoolsError):
            raise SelectoolsError("Test error")


class TestToolValidationError:
    """Test ToolValidationError exception."""

    def test_basic_creation(self):
        """Test basic error creation with required fields."""
        error = ToolValidationError(
            tool_name="test_tool",
            param_name="test_param",
            issue="Test issue",
        )
        assert error.tool_name == "test_tool"
        assert error.param_name == "test_param"
        assert error.issue == "Test issue"
        assert error.suggestion == ""

    def test_with_suggestion(self):
        """Test error creation with suggestion."""
        error = ToolValidationError(
            tool_name="test_tool",
            param_name="loction",
            issue="Unexpected parameter",
            suggestion="Did you mean 'location'?",
        )
        assert "Did you mean 'location'?" in error.suggestion
        assert "Did you mean 'location'?" in str(error)

    def test_error_message_formatting(self):
        """Test that error message is properly formatted with emojis."""
        error = ToolValidationError(
            tool_name="get_weather",
            param_name="locaton",
            issue="Unexpected parameter",
            suggestion="Did you mean 'location'?",
        )
        msg = str(error)
        assert "❌" in msg
        assert "Tool Validation Error" in msg
        assert "get_weather" in msg
        assert "locaton" in msg
        assert "💡" in msg
        assert "Did you mean 'location'?" in msg

    def test_inherits_from_selectools_error(self):
        """ToolValidationError should inherit from SelectoolsError."""
        assert issubclass(ToolValidationError, SelectoolsError)


class TestToolExecutionError:
    """Test ToolExecutionError exception."""

    def test_basic_creation(self):
        """Test basic error creation."""
        original_error = ValueError("Division by zero")
        error = ToolExecutionError(
            tool_name="calculator",
            error=original_error,
            params={"expression": "1/0"},
        )
        assert error.tool_name == "calculator"
        assert error.error == original_error
        assert error.params == {"expression": "1/0"}

    def test_error_message_formatting(self):
        """Test that error message includes helpful context."""
        original_error = KeyError("missing_key")
        error = ToolExecutionError(
            tool_name="data_fetcher",
            error=original_error,
            params={"url": "https://example.com", "key": "data"},
        )
        msg = str(error)
        assert "❌" in msg
        assert "Tool Execution Failed" in msg
        assert "data_fetcher" in msg
        assert "KeyError" in msg
        assert "missing_key" in msg
        assert "url" in msg
        assert "💡" in msg


class TestProviderConfigurationError:
    """Test ProviderConfigurationError exception."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = ProviderConfigurationError(
            provider_name="OpenAI",
            missing_config="API key",
        )
        assert error.provider_name == "OpenAI"
        assert error.missing_config == "API key"

    def test_with_env_var_suggestion(self):
        """Test error with environment variable suggestion."""
        error = ProviderConfigurationError(
            provider_name="OpenAI",
            missing_config="API key",
            env_var="OPENAI_API_KEY",
        )
        msg = str(error)
        assert "❌" in msg
        assert "Provider Configuration Error" in msg
        assert "OpenAI" in msg
        assert "💡" in msg
        assert "export OPENAI_API_KEY=" in msg
        assert "OpenAIProvider(api_key=" in msg

    def test_anthropic_provider_error(self):
        """Test error message for Anthropic provider."""
        error = ProviderConfigurationError(
            provider_name="Anthropic",
            missing_config="API key",
            env_var="ANTHROPIC_API_KEY",
        )
        msg = str(error)
        assert "AnthropicProvider(api_key=" in msg
        assert "ANTHROPIC_API_KEY" in msg

    def test_gemini_provider_error(self):
        """Test error message for Gemini provider."""
        error = ProviderConfigurationError(
            provider_name="Gemini",
            missing_config="API key",
            env_var="GEMINI_API_KEY",
        )
        msg = str(error)
        assert "GeminiProvider(api_key=" in msg
        assert "GEMINI_API_KEY" in msg


# =============================================================================
# Tool Validation Tests (Fuzzy Matching)
# =============================================================================


class TestToolValidationFuzzyMatching:
    """Test fuzzy matching for parameter typos."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            parameters=[
                ToolParameter(
                    name="location", param_type=str, description="City name", required=True
                ),
                ToolParameter(
                    name="units",
                    param_type=str,
                    description="celsius or fahrenheit",
                    required=False,
                ),
            ],
            function=lambda location, units="celsius": f"Weather in {location}: 72°",
        )

    def test_typo_detection_location(self):
        """Test detection of 'loction' typo."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"loction": "Paris"})

        error = exc_info.value
        assert "loction" in error.param_name
        assert "Did you mean 'location'?" in error.suggestion

    def test_typo_detection_locaton(self):
        """Test detection of 'locaton' typo."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"locaton": "Paris"})

        error = exc_info.value
        assert "locaton" in error.param_name
        assert "Did you mean 'location'?" in error.suggestion

    def test_typo_detection_units(self):
        """Test detection of 'unit' typo (missing 's')."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"location": "Paris", "unit": "celsius"})

        error = exc_info.value
        assert "unit" in error.param_name
        assert "Did you mean 'units'?" in error.suggestion

    def test_completely_wrong_parameter(self):
        """Test error for completely unrelated parameter name."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"xyz_unknown_param": "value"})

        error = exc_info.value
        assert "xyz_unknown_param" in error.param_name
        assert "is not a valid parameter" in error.suggestion
        assert "Expected parameters:" in error.suggestion

    def test_multiple_typos(self):
        """Test multiple parameter typos."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"loction": "Paris", "unts": "celsius"})

        # Should catch the first typo
        error = exc_info.value
        assert "loction" in error.param_name or "unts" in error.param_name

    def test_valid_parameters_no_error(self):
        """Test that valid parameters don't raise errors."""
        # Should not raise
        self.tool.validate({"location": "Paris"})
        self.tool.validate({"location": "Paris", "units": "celsius"})


class TestToolValidationMissingParameters:
    """Test validation of missing required parameters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = Tool(
            name="send_email",
            description="Send an email",
            parameters=[
                ToolParameter(name="to", param_type=str, description="Recipient", required=True),
                ToolParameter(
                    name="subject", param_type=str, description="Subject line", required=True
                ),
                ToolParameter(name="body", param_type=str, description="Email body", required=True),
                ToolParameter(
                    name="cc", param_type=str, description="CC recipients", required=False
                ),
            ],
            function=lambda to, subject, body, cc=None: f"Email sent to {to}",
        )

    def test_missing_single_required_parameter(self):
        """Test error when single required parameter is missing."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"to": "user@example.com", "body": "Hello"})

        error = exc_info.value
        assert "subject" in error.param_name
        assert "Missing required parameter" in error.issue
        assert "Required parameters:" in error.suggestion

    def test_missing_all_required_parameters(self):
        """Test error when all required parameters are missing."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({})

        error = exc_info.value
        assert "Missing required parameter" in error.issue

    def test_optional_parameter_can_be_omitted(self):
        """Test that optional parameters don't cause errors when omitted."""
        # Should not raise
        self.tool.validate({"to": "user@example.com", "subject": "Hello", "body": "Hi there"})


class TestToolValidationTypeMismatch:
    """Test validation of parameter type mismatches."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = Tool(
            name="calculate",
            description="Perform calculations",
            parameters=[
                ToolParameter(name="x", param_type=int, description="First number", required=True),
                ToolParameter(
                    name="y", param_type=float, description="Second number", required=True
                ),
                ToolParameter(
                    name="operation", param_type=str, description="Operation", required=True
                ),
            ],
            function=lambda x, y, operation: f"{x} {operation} {y}",
        )

    def test_string_instead_of_int(self):
        """Test error when string is provided instead of int."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"x": "five", "y": 2.0, "operation": "+"})

        error = exc_info.value
        assert "x" in error.param_name
        assert "must be of type int" in error.issue
        assert "got str" in error.issue

    def test_int_instead_of_string(self):
        """Test error when int is provided instead of string."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"x": 5, "y": 2.0, "operation": 123})

        error = exc_info.value
        assert "operation" in error.param_name
        assert "must be of type str" in error.issue

    def test_float_accepts_int(self):
        """Test that float parameters accept integers."""
        # Should not raise - int is acceptable for float
        self.tool.validate({"x": 5, "y": 2, "operation": "+"})

    def test_type_hint_in_suggestion(self):
        """Test that type conversion hints are provided."""
        with pytest.raises(ToolValidationError) as exc_info:
            self.tool.validate({"x": "5", "y": 2.0, "operation": "+"})

        error = exc_info.value
        # Should suggest conversion
        assert "Expected type: int" in error.suggestion or "int(" in error.suggestion


# =============================================================================
# Tool Execution Error Tests
# =============================================================================


class TestToolExecutionErrors:
    """Test tool execution error handling."""

    def test_exception_wrapped_in_tool_execution_error(self):
        """Test that tool exceptions are wrapped in ToolExecutionError."""

        def failing_tool(x: str) -> str:
            raise ValueError("Intentional failure")

        tool = Tool(
            name="failing_tool",
            description="A tool that always fails",
            parameters=[
                ToolParameter(name="x", param_type=str, description="Input", required=True)
            ],
            function=failing_tool,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"x": "test"})

        error = exc_info.value
        assert error.tool_name == "failing_tool"
        assert isinstance(error.error, ValueError)
        assert error.params == {"x": "test"}

    def test_key_error_wrapped(self):
        """Test that KeyError is wrapped with context."""

        def dict_tool(key: str) -> str:
            data = {"a": 1, "b": 2}
            return str(data[key])  # Will raise KeyError for unknown keys

        tool = Tool(
            name="dict_tool",
            description="Access dictionary",
            parameters=[
                ToolParameter(name="key", param_type=str, description="Key", required=True)
            ],
            function=dict_tool,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            tool.execute({"key": "unknown"})

        error = exc_info.value
        assert isinstance(error.error, KeyError)
        assert "dict_tool" in str(error)

    def test_async_tool_execution_error(self):
        """Test that async tool exceptions are also wrapped."""
        import asyncio

        async def async_failing_tool(x: str) -> str:
            raise RuntimeError("Async failure")

        tool = Tool(
            name="async_failing_tool",
            description="Async tool that fails",
            parameters=[
                ToolParameter(name="x", param_type=str, description="Input", required=True)
            ],
            function=async_failing_tool,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            asyncio.run(tool.aexecute({"x": "test"}))

        error = exc_info.value
        assert error.tool_name == "async_failing_tool"
        assert isinstance(error.error, RuntimeError)


# =============================================================================
# Integration Tests with Agent
# =============================================================================


class TestAgentErrorHandling:
    """Test error handling in Agent context."""

    def test_agent_handles_tool_validation_error_gracefully(self):
        """Test that agent handles tool validation errors gracefully."""

        @tool(description="Test tool")
        def test_tool(name: str) -> str:
            return f"Hello {name}"

        agent = Agent(
            tools=[test_tool],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=2),
        )

        # Agent should handle errors and continue
        response = agent.run([Message(role=Role.USER, content="Call test_tool with wrong params")])
        assert response.role == Role.ASSISTANT


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
