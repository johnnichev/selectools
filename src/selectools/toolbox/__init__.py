"""
Selectools Toolbox - Pre-built tools for common tasks.

This module provides a collection of ready-to-use tools for:
- File operations (read, write, list files)
- Web requests (HTTP GET/POST)
- Data processing (JSON, CSV parsing and formatting)
- Date/time utilities (current time, parsing, arithmetic)
- Text processing (search, replace, extract, case conversion)

Usage:
    from selectools.toolbox import get_all_tools, file_tools, web_tools

    # Get all tools
    all_tools = get_all_tools()

    # Get specific category
    tools = file_tools.get_tools()

    # Get individual tools
    from selectools.toolbox.file_tools import read_file, write_file
"""

from typing import List

from ..tools import Tool

# Import all tool modules
from . import data_tools, datetime_tools, file_tools, text_tools, web_tools

__all__ = [
    "file_tools",
    "web_tools",
    "data_tools",
    "datetime_tools",
    "text_tools",
    "get_all_tools",
    "get_tools_by_category",
]


def get_all_tools() -> List[Tool]:
    """
    Get all available pre-built tools from the toolbox.

    Returns:
        List of all Tool objects from all categories
    """
    tools = []

    # File tools
    tools.extend(
        [
            file_tools.read_file,
            file_tools.write_file,
            file_tools.list_files,
            file_tools.file_exists,
        ]
    )

    # Web tools
    tools.extend([web_tools.http_get, web_tools.http_post])

    # Data tools
    tools.extend(
        [
            data_tools.parse_json,
            data_tools.json_to_csv,
            data_tools.csv_to_json,
            data_tools.extract_json_field,
            data_tools.format_table,
        ]
    )

    # DateTime tools
    tools.extend(
        [
            datetime_tools.get_current_time,
            datetime_tools.parse_datetime,
            datetime_tools.time_difference,
            datetime_tools.date_arithmetic,
        ]
    )

    # Text tools
    tools.extend(
        [
            text_tools.count_text,
            text_tools.search_text,
            text_tools.replace_text,
            text_tools.extract_emails,
            text_tools.extract_urls,
            text_tools.convert_case,
            text_tools.truncate_text,
        ]
    )

    return tools


def get_tools_by_category(category: str) -> List[Tool]:
    """
    Get tools from a specific category.

    Args:
        category: Tool category ('file', 'web', 'data', 'datetime', 'text')

    Returns:
        List of Tool objects from the specified category

    Raises:
        ValueError: If category is invalid
    """
    categories = {
        "file": [
            file_tools.read_file,
            file_tools.write_file,
            file_tools.list_files,
            file_tools.file_exists,
        ],
        "web": [web_tools.http_get, web_tools.http_post],
        "data": [
            data_tools.parse_json,
            data_tools.json_to_csv,
            data_tools.csv_to_json,
            data_tools.extract_json_field,
            data_tools.format_table,
        ],
        "datetime": [
            datetime_tools.get_current_time,
            datetime_tools.parse_datetime,
            datetime_tools.time_difference,
            datetime_tools.date_arithmetic,
        ],
        "text": [
            text_tools.count_text,
            text_tools.search_text,
            text_tools.replace_text,
            text_tools.extract_emails,
            text_tools.extract_urls,
            text_tools.convert_case,
            text_tools.truncate_text,
        ],
    }

    if category not in categories:
        valid_categories = ", ".join(categories.keys())
        raise ValueError(f"Invalid category '{category}'. Valid categories: {valid_categories}")

    return categories[category]
