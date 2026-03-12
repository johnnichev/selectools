"""
Unit tests for structured.py: extract_json(), parse_and_validate(),
schema_from_response_format(), build_schema_instruction(), validation_retry_message().

Previously only covered by E2E tests that were always skipped in CI.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from selectools.structured import (
    build_schema_instruction,
    extract_json,
    parse_and_validate,
    schema_from_response_format,
    validation_retry_message,
)


class TestExtractJson:
    def test_fenced_code_block(self) -> None:
        text = 'Here is the result:\n```json\n{"name": "Alice", "age": 30}\n```\nDone.'
        result = extract_json(text)
        assert result is not None
        assert '"name": "Alice"' in result

    def test_fenced_block_no_language_tag(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result is not None
        assert '"key": "value"' in result

    def test_inline_json(self) -> None:
        text = 'The answer is {"temperature": 72, "unit": "F"} as shown.'
        result = extract_json(text)
        assert result is not None
        assert '"temperature": 72' in result

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": [1, 2, 3]}, "key": "val"}'
        result = extract_json(text)
        assert result is not None
        import json

        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == [1, 2, 3]

    def test_json_with_strings_containing_braces(self) -> None:
        text = '{"code": "if (x) { return y; }", "lang": "js"}'
        result = extract_json(text)
        assert result is not None
        import json

        parsed = json.loads(result)
        assert parsed["lang"] == "js"

    def test_no_json(self) -> None:
        assert extract_json("No JSON here at all") is None

    def test_no_opening_brace(self) -> None:
        assert extract_json("just text") is None

    def test_multiple_json_objects(self) -> None:
        text = '{"first": 1} some text {"second": 2}'
        result = extract_json(text)
        assert result is not None
        import json

        parsed = json.loads(result)
        assert parsed == {"first": 1}

    def test_escaped_quotes_in_strings(self) -> None:
        text = r'{"msg": "He said \"hello\""}'
        result = extract_json(text)
        assert result is not None


class TestSchemaFromResponseFormat:
    def test_dict_passthrough(self) -> None:
        schema: Dict[str, Any] = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = schema_from_response_format(schema)
        assert result is schema

    def test_pydantic_model(self) -> None:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class User(BaseModel):
            name: str
            age: int

        schema = schema_from_response_format(User)
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="response_format must be"):
            schema_from_response_format("not a schema")  # type: ignore


class TestParseAndValidate:
    def test_dict_schema(self) -> None:
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        schema: Dict[str, Any] = {"type": "object"}
        result = parse_and_validate(text, schema)
        assert result == {"name": "Alice", "age": 30}

    def test_pydantic_model(self) -> None:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class City(BaseModel):
            name: str
            population: int

        text = '{"name": "London", "population": 9000000}'
        result = parse_and_validate(text, City)
        assert result.name == "London"
        assert result.population == 9000000

    def test_pydantic_validation_error(self) -> None:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("pydantic not installed")

        class Strict(BaseModel):
            count: int

        with pytest.raises((ValueError, Exception)):
            parse_and_validate('{"count": "not_a_number"}', Strict)

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            parse_and_validate("Just plain text", {"type": "object"})

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object found"):
            parse_and_validate("{broken json", {"type": "object"})

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot validate"):
            parse_and_validate('{"key": 1}', int)  # type: ignore


class TestBuildSchemaInstruction:
    def test_contains_schema(self) -> None:
        schema: Dict[str, Any] = {"type": "object", "properties": {"x": {"type": "integer"}}}
        instruction = build_schema_instruction(schema)
        assert "JSON" in instruction
        assert '"type": "object"' in instruction
        assert "schema" in instruction.lower()


class TestValidationRetryMessage:
    def test_includes_error(self) -> None:
        msg = validation_retry_message(ValueError("missing field 'name'"))
        assert "missing field 'name'" in msg
        assert "valid JSON" in msg
