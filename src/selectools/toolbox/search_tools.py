"""
Web search and scraping tools using only the standard library.

No external dependencies required -- uses ``urllib`` for HTTP requests
and regex-based HTML parsing (no BeautifulSoup needed).
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from ..tools import tool

# Private IP networks that must be blocked to prevent SSRF
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_url(url: str) -> str | None:
    """Validate a URL to prevent SSRF attacks.

    Returns an error message if the URL is invalid/blocked, or None if safe.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return f"Error: URL scheme {parsed.scheme!r} is not allowed."

    hostname = parsed.hostname
    if not hostname:
        return "Error: URL has no hostname."

    lower_host = hostname.lower()
    if lower_host in ("localhost", "0.0.0.0"):
        return f"Error: Requests to {hostname!r} are blocked (loopback/internal address)."

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return f"Error: Could not resolve hostname {hostname!r}: {e}"

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                return (
                    f"Error: URL resolves to private/reserved address {ip} "
                    f"(network {network}). Requests to internal networks are blocked."
                )

    return None


_MAX_OUTPUT_BYTES = 10 * 1024  # 10 KB
_DEFAULT_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; selectools/0.21; +https://github.com/johnnichev/selectools)"


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate text to max_bytes, appending a notice if truncated."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n... (output truncated to 10 KB)"


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities from text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@tool(description="Search the web using DuckDuckGo (no API key needed)")
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo's HTML interface.

    No API key or external dependencies required.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return (default: 5, max: 20).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    if not query or not query.strip():
        return "Error: No search query provided."

    num_results = max(1, min(num_results, 20))

    try:
        encoded_query = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded_query}"

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")

        # Parse results from DuckDuckGo HTML
        # Results are in <a class="result__a" ...> for titles/URLs
        # and <a class="result__snippet" ...> for snippets
        results: list[dict[str, str]] = []

        # Match result blocks: each has a link and optional snippet
        title_pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        titles = title_pattern.findall(raw_html)
        snippets = snippet_pattern.findall(raw_html)

        for i, (href, title_html) in enumerate(titles[:num_results]):
            title = _strip_html_tags(title_html).strip()
            snippet = _strip_html_tags(snippets[i]).strip() if i < len(snippets) else ""

            # DuckDuckGo wraps URLs in a redirect; extract the real URL
            real_url = href
            if "uddg=" in href:
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    real_url = urllib.parse.unquote(match.group(1))

            results.append({"title": title, "url": real_url, "snippet": snippet})

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines).strip()

    except urllib.error.URLError as e:
        return f"Error: Could not connect to DuckDuckGo: {e}"
    except Exception as e:
        return f"Error performing web search: {e}"


@tool(description="Fetch a URL and extract text content")
def scrape_url(url: str, selector: Optional[str] = None) -> str:
    """
    Fetch a URL and extract its text content.

    HTML tags are stripped using regex (no BeautifulSoup dependency).
    Output is truncated to 10 KB.

    Args:
        url: The URL to fetch.
        selector: Optional CSS-like tag filter (e.g. ``"p"`` to extract only
            ``<p>`` blocks). Only simple tag names are supported.

    Returns:
        Extracted text content or an error message.
    """
    if not url or not url.strip():
        return "Error: No URL provided."

    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    ssrf_error = _validate_url(url)
    if ssrf_error:
        return ssrf_error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")

        # If selector is provided, extract matching tag blocks first
        if selector and selector.strip():
            tag = re.escape(selector.strip())
            pattern = re.compile(rf"<{tag}[^>]*>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(raw)
            if matches:
                raw = "\n".join(matches)
            else:
                return f"No <{selector}> elements found on the page."

        text = _strip_html_tags(raw)

        if not text:
            return "No text content found at the URL."

        lines = [
            f"URL: {url}",
            f"Content-Type: {content_type}",
            f"Length: {len(text)} characters",
            "",
            text,
        ]
        return _truncate("\n".join(lines))

    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: Could not connect to {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching URL: {e}"
