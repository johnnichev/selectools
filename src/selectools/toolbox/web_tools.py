"""
Web request tools for making HTTP requests.

Requires the 'requests' library for external HTTP calls.
"""

import json
from typing import Optional

from ..tools import tool


@tool(description="Make an HTTP GET request to a URL")
def http_get(
    url: str,
    headers: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP GET request and return the response.

    Args:
        url: The URL to request
        headers: JSON string of headers to include (optional)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Response content or error message
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return "❌ Error: 'requests' library not installed. Run: pip install requests"

    try:
        # Parse headers if provided
        headers_dict = None
        if headers:
            try:
                headers_dict = json.loads(headers)
            except json.JSONDecodeError:
                return f"❌ Error: Invalid JSON in headers: {headers}"

        # Make request
        response = requests.get(url, headers=headers_dict, timeout=timeout)

        # Format response
        content_type = response.headers.get("Content-Type", "")
        lines = [
            f"Status: {response.status_code} {response.reason}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(response.content)} bytes",
            "",
        ]

        # Try to format JSON responses nicely
        if "application/json" in content_type:
            try:
                json_data = response.json()
                lines.append(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                lines.append(response.text)
        else:
            # For text responses, truncate if very long
            text = response.text
            if len(text) > 5000:
                lines.append(text[:5000] + "\n... (truncated)")
            else:
                lines.append(text)

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return f"❌ Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError:
        return f"❌ Error: Could not connect to {url}"
    except requests.exceptions.RequestException as e:
        return f"❌ Error making request: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


@tool(description="Make an HTTP POST request with JSON data")
def http_post(
    url: str,
    data: str,
    headers: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    Make an HTTP POST request with JSON data.

    Args:
        url: The URL to request
        data: JSON string of data to send in request body
        headers: JSON string of headers to include (optional)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Response content or error message
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return "❌ Error: 'requests' library not installed. Run: pip install requests"

    try:
        # Parse data
        try:
            data_dict = json.loads(data)
        except json.JSONDecodeError:
            return f"❌ Error: Invalid JSON in data: {data}"

        # Parse headers if provided
        headers_dict = {"Content-Type": "application/json"}
        if headers:
            try:
                headers_dict.update(json.loads(headers))
            except json.JSONDecodeError:
                return f"❌ Error: Invalid JSON in headers: {headers}"

        # Make request
        response = requests.post(url, json=data_dict, headers=headers_dict, timeout=timeout)

        # Format response
        content_type = response.headers.get("Content-Type", "")
        lines = [
            f"Status: {response.status_code} {response.reason}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(response.content)} bytes",
            "",
        ]

        # Try to format JSON responses nicely
        if "application/json" in content_type:
            try:
                json_data = response.json()
                lines.append(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                lines.append(response.text)
        else:
            text = response.text
            if len(text) > 5000:
                lines.append(text[:5000] + "\n... (truncated)")
            else:
                lines.append(text)

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return f"❌ Error: Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError:
        return f"❌ Error: Could not connect to {url}"
    except requests.exceptions.RequestException as e:
        return f"❌ Error making request: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"
