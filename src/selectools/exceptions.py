"""
Custom exceptions with helpful error messages and suggestions.

These exceptions provide PyTorch-style error messages with:
- Clear explanations of what went wrong
- Concrete suggestions for fixes
- Relevant context (parameter names, types, etc.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SelectoolsError(Exception):
    """Base exception for all selectools errors."""

    pass


class ToolValidationError(SelectoolsError):
    """Raised when tool parameters are invalid."""

    def __init__(self, tool_name: str, param_name: str, issue: str, suggestion: str = ""):
        self.tool_name = tool_name
        self.param_name = param_name
        self.issue = issue
        self.suggestion = suggestion

        message = f"\n{'='*60}\n"
        message += f"❌ Tool Validation Error: '{tool_name}'\n"
        message += f"{'='*60}\n\n"
        message += f"Parameter: {param_name}\n"
        message += f"Issue: {issue}\n"
        if suggestion:
            message += f"\n💡 Suggestion: {suggestion}\n"
        message += f"\n{'='*60}\n"

        super().__init__(message)


class ToolExecutionError(SelectoolsError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, error: Exception, params: Dict[str, Any]):
        self.tool_name = tool_name
        self.error = error
        self.params = params

        message = f"\n{'='*60}\n"
        message += f"❌ Tool Execution Failed: '{tool_name}'\n"
        message += f"{'='*60}\n\n"
        message += f"Error: {type(error).__name__}: {str(error)}\n"
        message += f"Parameters: {params}\n"
        message += f"\n💡 Check that:\n"
        message += f"  - All required parameters are provided\n"
        message += f"  - Parameter types match the tool's schema\n"
        message += f"  - The tool function is correctly implemented\n"
        message += f"\n{'='*60}\n"

        super().__init__(message)


class ProviderConfigurationError(SelectoolsError):
    """Raised when provider configuration is incorrect."""

    def __init__(self, provider_name: str, missing_config: str, env_var: str = ""):
        self.provider_name = provider_name
        self.missing_config = missing_config
        self.env_var = env_var

        message = f"\n{'='*60}\n"
        message += f"❌ Provider Configuration Error: '{provider_name}'\n"
        message += f"{'='*60}\n\n"
        message += f"Missing: {missing_config}\n"
        if env_var:
            message += f"\n💡 How to fix:\n"
            message += f"  1. Set the environment variable:\n"
            message += f"     export {env_var}='your-api-key'\n"
            message += f"  2. Or pass it directly:\n"
            message += f"     provider = {provider_name}Provider(api_key='your-api-key')\n"
        message += f"\n{'='*60}\n"

        super().__init__(message)


class MemoryLimitExceededError(SelectoolsError):
    """Raised when memory limits are exceeded."""

    def __init__(self, current: int, limit: int, limit_type: str):
        self.current = current
        self.limit = limit
        self.limit_type = limit_type

        message = f"\n{'='*60}\n"
        message += f"⚠️  Memory Limit Exceeded\n"
        message += f"{'='*60}\n\n"
        message += f"Limit Type: {limit_type}\n"
        message += f"Current: {current}\n"
        message += f"Limit: {limit}\n"
        message += f"\n💡 Suggestions:\n"
        if limit_type == "messages":
            message += f"  - Increase max_messages: ConversationMemory(max_messages={limit * 2})\n"
            message += f"  - Clear older messages manually: memory.clear()\n"
        elif limit_type == "tokens":
            message += f"  - Increase max_tokens: ConversationMemory(max_tokens={limit * 2})\n"
            message += f"  - Use shorter messages\n"
            message += f"  - Enable conversation summarization (coming in v0.6.0)\n"
        message += f"\n{'='*60}\n"

        super().__init__(message)


__all__ = [
    "SelectoolsError",
    "ToolValidationError",
    "ToolExecutionError",
    "ProviderConfigurationError",
    "MemoryLimitExceededError",
]
