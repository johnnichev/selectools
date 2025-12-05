"""
Tool metadata, schemas, and runtime validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


JsonSchema = Dict[str, Any]
ParameterValue = Union[str, int, float, bool, dict, list]


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
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

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
        return self.function(**params)


__all__ = ["Tool", "ToolParameter"]
