"""
Text processing and manipulation tools.
"""

import re
from typing import List, Optional

from ..tools import tool


@tool(description="Count words, characters, and lines in text")
def count_text(text: str, detailed: bool = True) -> str:
    """
    Count words, characters, lines, and other statistics in text.

    Args:
        text: The text to analyze
        detailed: Include detailed statistics

    Returns:
        Text statistics
    """
    try:
        lines = text.split("\n")
        words = text.split()
        chars = len(text)
        chars_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

        result = [
            f"Lines: {len(lines)}",
            f"Words: {len(words)}",
            f"Characters: {chars}",
        ]

        if detailed:
            result.extend(
                [
                    f"Characters (no spaces): {chars_no_spaces}",
                    f"Average word length: {chars_no_spaces / len(words):.1f}" if words else "N/A",
                    f"Average words per line: {len(words) / len(lines):.1f}" if lines else "N/A",
                ]
            )

        return "\n".join(result)
    except Exception as e:
        return f"❌ Error counting text: {e}"


@tool(description="Search for text pattern using regex")
def search_text(
    text: str, pattern: str, case_sensitive: bool = True, return_matches: bool = True
) -> str:
    """
    Search for a pattern in text using regular expressions.

    Args:
        text: The text to search
        pattern: Regular expression pattern to search for
        case_sensitive: Whether search is case-sensitive
        return_matches: Return matching text (vs just count)

    Returns:
        Search results or error message
    """
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        matches = re.findall(pattern, text, flags=flags)

        if not matches:
            return f"No matches found for pattern: {pattern}"

        result = [f"Found {len(matches)} match(es)"]
        if return_matches:
            result.append("\nMatches:")
            for i, match in enumerate(matches[:20], 1):  # Limit to 20 matches
                result.append(f"  {i}. {match}")
            if len(matches) > 20:
                result.append(f"  ... and {len(matches) - 20} more")

        return "\n".join(result)
    except re.error as e:
        return f"❌ Invalid regex pattern: {e}"
    except Exception as e:
        return f"❌ Error searching text: {e}"


@tool(description="Replace text pattern with replacement")
def replace_text(
    text: str,
    pattern: str,
    replacement: str,
    case_sensitive: bool = True,
    max_replacements: int = 0,
) -> str:
    """
    Replace occurrences of a pattern with replacement text.

    Args:
        text: The text to process
        pattern: Pattern to search for (regex supported)
        replacement: Text to replace matches with
        case_sensitive: Whether search is case-sensitive
        max_replacements: Max number of replacements (0 = unlimited)

    Returns:
        Modified text with replacement count
    """
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        count = max_replacements if max_replacements > 0 else 0

        new_text, num_replacements = re.subn(pattern, replacement, text, count=count, flags=flags)

        return f"✅ Made {num_replacements} replacement(s)\n\nResult:\n{new_text}"
    except re.error as e:
        return f"❌ Invalid regex pattern: {e}"
    except Exception as e:
        return f"❌ Error replacing text: {e}"


@tool(description="Extract email addresses from text")
def extract_emails(text: str) -> str:
    """
    Extract all email addresses from text.

    Args:
        text: Text to extract emails from

    Returns:
        List of found email addresses
    """
    try:
        # Email regex pattern
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(pattern, text)

        if not emails:
            return "No email addresses found"

        # Remove duplicates while preserving order
        seen = set()
        unique_emails = []
        for email in emails:
            if email.lower() not in seen:
                seen.add(email.lower())
                unique_emails.append(email)

        result = [f"Found {len(unique_emails)} unique email(s):\n"]
        for i, email in enumerate(unique_emails, 1):
            result.append(f"  {i}. {email}")

        return "\n".join(result)
    except Exception as e:
        return f"❌ Error extracting emails: {e}"


@tool(description="Extract URLs from text")
def extract_urls(text: str) -> str:
    """
    Extract all URLs from text.

    Args:
        text: Text to extract URLs from

    Returns:
        List of found URLs
    """
    try:
        # URL regex pattern
        pattern = r"https?://[^\s<>\"']+"
        urls = re.findall(pattern, text)

        if not urls:
            return "No URLs found"

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        result = [f"Found {len(unique_urls)} unique URL(s):\n"]
        for i, url in enumerate(unique_urls, 1):
            result.append(f"  {i}. {url}")

        return "\n".join(result)
    except Exception as e:
        return f"❌ Error extracting URLs: {e}"


@tool(description="Convert text case")
def convert_case(text: str, case_type: str) -> str:
    """
    Convert text to different cases.

    Args:
        text: The text to convert
        case_type: Target case ('upper', 'lower', 'title', 'sentence', 'camel', 'snake', 'kebab')

    Returns:
        Converted text or error message
    """
    try:
        if case_type == "upper":
            return text.upper()
        elif case_type == "lower":
            return text.lower()
        elif case_type == "title":
            return text.title()
        elif case_type == "sentence":
            # Capitalize first letter of each sentence
            sentences = re.split(r"([.!?]\s+)", text)
            result = []
            for i, s in enumerate(sentences):
                if i % 2 == 0 and s:  # Actual sentence (not delimiter)
                    result.append(s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper())
                else:
                    result.append(s)
            return "".join(result)
        elif case_type == "camel":
            # Convert to camelCase
            words: List[str] = re.findall(r"[a-zA-Z0-9]+", text)
            if not words:
                return text
            return words[0].lower() + "".join(w.capitalize() for w in words[1:])
        elif case_type == "snake":
            # Convert to snake_case
            words = re.findall(r"[a-zA-Z0-9]+", text)
            return "_".join(w.lower() for w in words)
        elif case_type == "kebab":
            # Convert to kebab-case
            words = re.findall(r"[a-zA-Z0-9]+", text)
            return "-".join(w.lower() for w in words)
        else:
            return f"❌ Error: Invalid case type '{case_type}'. Options: upper, lower, title, sentence, camel, snake, kebab"
    except Exception as e:
        return f"❌ Error converting case: {e}"


@tool(description="Truncate text to a maximum length")
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: String to append if truncated (default: ...)

    Returns:
        Truncated text
    """
    try:
        if len(text) <= max_length:
            return text

        truncate_at = max_length - len(suffix)
        if truncate_at < 1:
            return suffix[:max_length]

        return text[:truncate_at] + suffix
    except Exception as e:
        return f"❌ Error truncating text: {e}"
