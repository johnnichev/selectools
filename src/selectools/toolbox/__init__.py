"""
Selectools Toolbox - Pre-built tools for common tasks.

This module provides a collection of ready-to-use tools for:
- File operations (read, write, list files)
- Web requests (HTTP GET/POST)
- Data processing (JSON, CSV parsing and formatting)
- Date/time utilities (current time, parsing, arithmetic)
- Text processing (search, replace, extract, case conversion)
- Code execution (Python, shell commands)
- Web search and scraping (DuckDuckGo, URL scraping)
- GitHub API (repos, files, issues)
- Database queries (SQLite, PostgreSQL)
- Calculator (safe math evaluation, unit conversion)
- Email (SMTP send, IMAP inbox reading)
- PDF extraction (text, tables)
- Slack (send, read channel, search)
- Notion (create, search, update pages)
- Linear (create, list, update issues)
- Discord (send, read channel)
- Amazon S3 (list, get, put objects)
- Browser (scrape rendered pages, screenshots)
- Image generation (OpenAI Images API)

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

from ..stability import stable
from ..tools import Tool

# Import all tool modules
from . import (
    browser_tools,
    calculator_tools,
    code_tools,
    data_tools,
    datetime_tools,
    db_tools,
    discord_tools,
    email_tools,
    file_tools,
    github_tools,
    image_tools,
    linear_tools,
    notion_tools,
    pdf_tools,
    s3_tools,
    search_tools,
    slack_tools,
    text_tools,
    web_tools,
)

__stability__ = "stable"

__all__ = [
    "file_tools",
    "web_tools",
    "data_tools",
    "datetime_tools",
    "text_tools",
    "code_tools",
    "search_tools",
    "github_tools",
    "db_tools",
    "calculator_tools",
    "email_tools",
    "pdf_tools",
    "slack_tools",
    "notion_tools",
    "linear_tools",
    "discord_tools",
    "s3_tools",
    "browser_tools",
    "image_tools",
    "get_all_tools",
    "get_tools_by_category",
]


@stable
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
            file_tools.read_file_stream,
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
            data_tools.process_csv_stream,
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

    # Code tools
    tools.extend([code_tools.execute_python, code_tools.execute_shell])

    # Search tools
    tools.extend([search_tools.web_search, search_tools.scrape_url])

    # GitHub tools
    tools.extend(
        [
            github_tools.github_search_repos,
            github_tools.github_get_file,
            github_tools.github_list_issues,
        ]
    )

    # Database tools
    tools.extend([db_tools.query_sqlite, db_tools.query_postgres])

    # Calculator tools
    tools.extend([calculator_tools.evaluate_expression, calculator_tools.unit_convert])

    # Email tools
    tools.extend([email_tools.send_email, email_tools.read_inbox])

    # PDF tools
    tools.extend([pdf_tools.extract_pdf_text, pdf_tools.extract_pdf_tables])

    # Slack tools
    tools.extend(
        [
            slack_tools.slack_send_message,
            slack_tools.slack_read_channel,
            slack_tools.slack_search_messages,
        ]
    )

    # Notion tools
    tools.extend(
        [
            notion_tools.notion_create_page,
            notion_tools.notion_search,
            notion_tools.notion_update_page,
        ]
    )

    # Linear tools
    tools.extend(
        [
            linear_tools.linear_create_issue,
            linear_tools.linear_list_issues,
            linear_tools.linear_update_issue,
        ]
    )

    # Discord tools
    tools.extend(
        [
            discord_tools.discord_send_message,
            discord_tools.discord_read_channel,
        ]
    )

    # S3 tools
    tools.extend(
        [
            s3_tools.s3_list_objects,
            s3_tools.s3_get_object,
            s3_tools.s3_put_object,
        ]
    )

    # Browser tools
    tools.extend(
        [
            browser_tools.browser_scrape_page,
            browser_tools.browser_screenshot,
        ]
    )

    # Image tools
    tools.extend([image_tools.generate_image])

    return tools


@stable
def get_tools_by_category(category: str) -> List[Tool]:
    """
    Get tools from a specific category.

    Args:
        category: Tool category ('file', 'web', 'data', 'datetime', 'text',
            'code', 'search', 'github', 'database', 'calculator', 'email',
            'pdf', 'slack', 'notion', 'linear', 'discord', 's3', 'browser',
            'image')

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
            file_tools.read_file_stream,
        ],
        "web": [web_tools.http_get, web_tools.http_post],
        "data": [
            data_tools.parse_json,
            data_tools.json_to_csv,
            data_tools.csv_to_json,
            data_tools.extract_json_field,
            data_tools.format_table,
            data_tools.process_csv_stream,
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
        "code": [code_tools.execute_python, code_tools.execute_shell],
        "search": [search_tools.web_search, search_tools.scrape_url],
        "github": [
            github_tools.github_search_repos,
            github_tools.github_get_file,
            github_tools.github_list_issues,
        ],
        "database": [db_tools.query_sqlite, db_tools.query_postgres],
        "calculator": [calculator_tools.evaluate_expression, calculator_tools.unit_convert],
        "email": [email_tools.send_email, email_tools.read_inbox],
        "pdf": [pdf_tools.extract_pdf_text, pdf_tools.extract_pdf_tables],
        "slack": [
            slack_tools.slack_send_message,
            slack_tools.slack_read_channel,
            slack_tools.slack_search_messages,
        ],
        "notion": [
            notion_tools.notion_create_page,
            notion_tools.notion_search,
            notion_tools.notion_update_page,
        ],
        "linear": [
            linear_tools.linear_create_issue,
            linear_tools.linear_list_issues,
            linear_tools.linear_update_issue,
        ],
        "discord": [
            discord_tools.discord_send_message,
            discord_tools.discord_read_channel,
        ],
        "s3": [
            s3_tools.s3_list_objects,
            s3_tools.s3_get_object,
            s3_tools.s3_put_object,
        ],
        "browser": [
            browser_tools.browser_scrape_page,
            browser_tools.browser_screenshot,
        ],
        "image": [image_tools.generate_image],
    }

    if category not in categories:
        valid_categories = ", ".join(categories.keys())
        raise ValueError(f"Invalid category '{category}'. Valid categories: {valid_categories}")

    return categories[category]
