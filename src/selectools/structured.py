"""
Structured output parsing and validation.

Extracts typed objects from LLM responses using Pydantic models or raw
JSON Schema dicts, with automatic retry on validation failure.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type, Union

ResponseFormat = Union[Type[Any], Dict[str, Any]]


def schema_from_response_format(response_format: ResponseFormat) -> Dict[str, Any]:
    """Convert a Pydantic model class or raw dict into a JSON Schema dict."""
    if isinstance(response_format, dict):
        return response_format
    if hasattr(response_format, "model_json_schema"):
        schema: Dict[str, Any] = response_format.model_json_schema()  # type: ignore[union-attr]
        return schema
    raise TypeError(
        f"response_format must be a Pydantic BaseModel class or a dict JSON Schema, "
        f"got {type(response_format).__name__}"
    )


def build_schema_instruction(schema: Dict[str, Any]) -> str:
    """Build a system prompt suffix instructing the LLM to produce valid JSON."""
    compact = json.dumps(schema, indent=2)
    return (
        "\n\nYou MUST respond with valid JSON that conforms to this schema:\n"
        f"```json\n{compact}\n```\n"
        "Return ONLY the JSON object, no extra text."
    )


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[str]:
    """Extract the first complete JSON object from text.

    Tries fenced code blocks first, then scans for the first brace-balanced
    ``{...}`` span so that trailing text or multiple JSON objects never produce
    a spurious parse error.
    """
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return None


def parse_and_validate(
    text: str,
    response_format: ResponseFormat,
) -> Any:
    """Parse JSON from *text* and validate against *response_format*.

    Returns:
        A Pydantic model instance (if *response_format* is a BaseModel subclass)
        or a plain ``dict`` (if *response_format* is a dict schema).

    Raises:
        ValueError: If no JSON could be extracted or validation fails.
    """
    raw_json = extract_json(text)
    if raw_json is None:
        raise ValueError(
            "No JSON object found in the response. " "Please respond with a valid JSON object."
        )

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(response_format, dict):
        return data

    if hasattr(response_format, "model_validate"):
        return response_format.model_validate(data)  # type: ignore[union-attr]

    raise TypeError(
        f"Cannot validate against {type(response_format).__name__}. "
        f"Use a Pydantic BaseModel class or a dict JSON Schema."
    )


def validation_retry_message(error: Exception) -> str:
    """Build a user-message telling the LLM what went wrong so it can fix it."""
    return (
        f"Your previous response was not valid: {error}\n"
        "Please try again, returning ONLY a valid JSON object matching the schema."
    )


__all__ = [
    "ResponseFormat",
    "schema_from_response_format",
    "build_schema_instruction",
    "extract_json",
    "parse_and_validate",
    "validation_retry_message",
]
