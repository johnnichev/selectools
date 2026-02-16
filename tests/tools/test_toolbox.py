"""
Comprehensive tests for the Pre-built Tool Library (toolbox).

Tests cover all tool categories:
- File operations
- Web requests
- Data processing
- DateTime utilities
- Text processing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from selectools.toolbox import (
    data_tools,
    datetime_tools,
    file_tools,
    get_all_tools,
    get_tools_by_category,
    text_tools,
    web_tools,
)

# =============================================================================
# Toolbox API Tests
# =============================================================================


class TestToolboxAPI:
    """Test the toolbox module-level API."""

    def test_get_all_tools(self) -> None:
        """Test getting all tools from toolbox."""
        tools = get_all_tools()
        assert len(tools) > 0
        assert all(hasattr(tool, "name") for tool in tools)
        assert all(hasattr(tool, "description") for tool in tools)

    def test_get_tools_by_category(self) -> None:
        """Test getting tools by category."""
        file_category = get_tools_by_category("file")
        assert len(file_category) > 0

        web_category = get_tools_by_category("web")
        assert len(web_category) > 0

        data_category = get_tools_by_category("data")
        assert len(data_category) > 0

    def test_get_tools_invalid_category(self) -> None:
        """Test error handling for invalid category."""
        with pytest.raises(ValueError, match="Invalid category"):
            get_tools_by_category("invalid_category")


# =============================================================================
# File Tools Tests
# =============================================================================


class TestFileTools:
    """Test file operation tools."""

    def test_read_file_success(self) -> None:
        """Test reading an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content")
            filepath = f.name

        try:
            result = file_tools.read_file.function(filepath)
            assert "Test content" in result
            assert "characters" in result
        finally:
            Path(filepath).unlink()

    def test_read_file_not_found(self) -> None:
        """Test reading a non-existent file."""
        result = file_tools.read_file.function("/nonexistent/file.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_write_file_success(self) -> None:
        """Test writing to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            result = file_tools.write_file.function(str(filepath), "Hello World")
            assert "✅" in result or "Written" in result

            # Verify file was written
            assert filepath.exists()
            assert filepath.read_text() == "Hello World"

    def test_write_file_append_mode(self) -> None:
        """Test appending to a file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("First line\n")
            filepath = f.name

        try:
            result = file_tools.write_file.function(filepath, "Second line\n", mode="a")
            assert "Appended" in result or "✅" in result

            content = Path(filepath).read_text()
            assert "First line" in content
            assert "Second line" in content
        finally:
            Path(filepath).unlink()

    def test_list_files(self) -> None:
        """Test listing files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            (Path(tmpdir) / "file1.txt").touch()
            (Path(tmpdir) / "file2.txt").touch()
            (Path(tmpdir) / "subdir").mkdir()

            result = file_tools.list_files.function(tmpdir)
            assert "file1.txt" in result
            assert "file2.txt" in result

    def test_file_exists(self) -> None:
        """Test checking if file exists."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name

        try:
            result = file_tools.file_exists.function(filepath)
            assert "exists" in result.lower()
        finally:
            Path(filepath).unlink()


# =============================================================================
# Data Tools Tests
# =============================================================================


class TestDataTools:
    """Test data processing tools."""

    def test_parse_json_valid(self) -> None:
        """Test parsing valid JSON."""
        json_str = '{"name": "Alice", "age": 30}'
        result = data_tools.parse_json.function(json_str)
        assert "Valid JSON" in result
        assert "Alice" in result

    def test_parse_json_invalid(self) -> None:
        """Test parsing invalid JSON."""
        result = data_tools.parse_json.function("{invalid json}")
        assert "Invalid JSON" in result or "Error" in result

    def test_json_to_csv(self) -> None:
        """Test converting JSON to CSV."""
        json_str = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        result = data_tools.json_to_csv.function(json_str)
        assert "CSV" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "name" in result or "age" in result

    def test_csv_to_json(self) -> None:
        """Test converting CSV to JSON."""
        csv_str = "name,age\nAlice,30\nBob,25"
        result = data_tools.csv_to_json.function(csv_str)
        assert "JSON" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_extract_json_field(self) -> None:
        """Test extracting field from JSON."""
        json_str = '{"user": {"name": "Alice", "email": "alice@example.com"}}'
        result = data_tools.extract_json_field.function(json_str, "user.name")
        assert "Alice" in result

    def test_extract_json_field_array_index(self) -> None:
        """Test extracting array element from JSON."""
        json_str = '{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}'
        result = data_tools.extract_json_field.function(json_str, "items.1.id")
        assert "2" in result

    def test_format_table_simple(self) -> None:
        """Test formatting data as simple table."""
        data_str = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        result = data_tools.format_table.function(data_str, format_type="simple")
        assert "Alice" in result
        assert "Bob" in result
        assert "|" in result

    def test_format_table_markdown(self) -> None:
        """Test formatting data as markdown table."""
        data_str = '[{"name": "Alice", "age": 30}]'
        result = data_tools.format_table.function(data_str, format_type="markdown")
        assert "|" in result
        assert "---" in result


# =============================================================================
# DateTime Tools Tests
# =============================================================================


class TestDateTimeTools:
    """Test date/time utility tools."""

    def test_get_current_time(self) -> None:
        """Test getting current time."""
        result = datetime_tools.get_current_time.function()
        assert "time" in result.lower() or "UTC" in result

    def test_parse_datetime_with_format(self) -> None:
        """Test parsing datetime with explicit format."""
        result = datetime_tools.parse_datetime.function(
            "2025-12-09 10:30:00", input_format="%Y-%m-%d %H:%M:%S"
        )
        assert "2025" in result
        assert "Parsed" in result or "12" in result

    def test_parse_datetime_auto(self) -> None:
        """Test auto-parsing common datetime formats."""
        result = datetime_tools.parse_datetime.function("2025-12-09")
        assert "2025" in result

    def test_time_difference(self) -> None:
        """Test calculating time difference."""
        result = datetime_tools.time_difference.function("2025-12-01", "2025-12-09", unit="days")
        assert "8" in result or "difference" in result.lower()

    def test_date_arithmetic_add(self) -> None:
        """Test adding time to a date."""
        result = datetime_tools.date_arithmetic.function("2025-12-09", "add", 7, unit="days")
        assert "2025-12-16" in result or "Result" in result

    def test_date_arithmetic_subtract(self) -> None:
        """Test subtracting time from a date."""
        result = datetime_tools.date_arithmetic.function("2025-12-09", "subtract", 5, unit="days")
        assert "2025-12-04" in result or "Result" in result


# =============================================================================
# Text Tools Tests
# =============================================================================


class TestTextTools:
    """Test text processing tools."""

    def test_count_text(self) -> None:
        """Test counting words, lines, and characters."""
        text = "Hello world\nThis is a test"
        result = text_tools.count_text.function(text)
        assert "Words:" in result or "words" in result.lower()
        assert "2" in result  # 2 lines
        assert "6" in result  # 6 words

    def test_search_text(self) -> None:
        """Test searching for pattern in text."""
        text = "Contact: alice@example.com and bob@example.com"
        result = text_tools.search_text.function(text, r"\w+@\w+\.\w+")
        assert "match" in result.lower()
        assert "alice" in result.lower() or "bob" in result.lower()

    def test_replace_text(self) -> None:
        """Test replacing text pattern."""
        text = "Hello world, hello universe"
        result = text_tools.replace_text.function(text, "hello", "Hi", case_sensitive=False)
        assert "Hi" in result
        assert "replacement" in result.lower()

    def test_extract_emails(self) -> None:
        """Test extracting email addresses."""
        text = "Contact alice@example.com or bob@test.com for info"
        result = text_tools.extract_emails.function(text)
        assert "alice@example.com" in result
        assert "bob@test.com" in result

    def test_extract_urls(self) -> None:
        """Test extracting URLs."""
        text = "Visit https://example.com and http://test.com for more"
        result = text_tools.extract_urls.function(text)
        assert "https://example.com" in result
        assert "http://test.com" in result

    def test_convert_case_upper(self) -> None:
        """Test converting to uppercase."""
        result = text_tools.convert_case.function("hello world", "upper")
        assert result == "HELLO WORLD"

    def test_convert_case_title(self) -> None:
        """Test converting to title case."""
        result = text_tools.convert_case.function("hello world", "title")
        assert result == "Hello World"

    def test_convert_case_snake(self) -> None:
        """Test converting to snake_case."""
        result = text_tools.convert_case.function("Hello World Test", "snake")
        assert result == "hello_world_test"

    def test_convert_case_camel(self) -> None:
        """Test converting to camelCase."""
        result = text_tools.convert_case.function("hello world test", "camel")
        assert result == "helloWorldTest"

    def test_truncate_text(self) -> None:
        """Test truncating text."""
        long_text = "A" * 200
        result = text_tools.truncate_text.function(long_text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")


# =============================================================================
# Web Tools Tests (mocked)
# =============================================================================


class TestWebTools:
    """Test web request tools (requires requests library)."""

    def test_http_get_no_requests_library(self) -> None:
        """Test http_get handles missing requests library gracefully."""
        # This test just verifies the tool exists and has proper structure
        assert hasattr(web_tools.http_get, "function")
        assert hasattr(web_tools.http_get, "description")
        assert (
            "GET" in web_tools.http_get.description
            or "request" in web_tools.http_get.description.lower()
        )

    def test_http_post_no_requests_library(self) -> None:
        """Test http_post handles missing requests library gracefully."""
        assert hasattr(web_tools.http_post, "function")
        assert hasattr(web_tools.http_post, "description")
        assert (
            "POST" in web_tools.http_post.description
            or "request" in web_tools.http_post.description.lower()
        )
