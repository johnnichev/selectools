"""
Browser tools -- scrape rendered pages and take screenshots.

Requires the optional ``playwright`` library, installed via
``pip install selectools[browser]`` followed by
``playwright install chromium`` (downloads the browser binary). The
import is lazy: the module loads fine without the dependency.

Pages are rendered in headless Chromium, so JavaScript-heavy sites work
where plain HTTP scraping (``search_tools.scrape_url``) does not. No
credentials are required.
"""

from __future__ import annotations

from pathlib import Path

from ..stability import beta
from ..tools import tool

_MISSING_DEP_ERROR = (
    "Error: 'playwright' library not installed. "
    "Run: pip install selectools[browser] && playwright install chromium"
)
_DEFAULT_TIMEOUT_S = 30
_MAX_TEXT_CHARS = 5000


@beta
@tool(description="Scrape the visible text of a web page using a headless browser")
def browser_scrape_page(url: str, timeout: int = _DEFAULT_TIMEOUT_S) -> str:
    """
    Load a page in headless Chromium and return its visible text.

    JavaScript is executed before extraction, so dynamically rendered
    content is included. Output is truncated to 5000 characters.

    Args:
        url: URL of the page to scrape (http/https).
        timeout: Page load timeout in seconds (default: 30, max: 120).

    Returns:
        The page's visible text content, or a readable error string.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError  # type: ignore[import-untyped]
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    if not url or not url.strip():
        return "Error: No URL provided."
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    timeout_ms = max(1, min(timeout, 120)) * 1000

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=timeout_ms)
                title = page.title()
                text = page.inner_text("body")
            finally:
                browser.close()

        text = text.strip()
        if not text:
            return f"No visible text found at {url}."
        truncated = len(text) > _MAX_TEXT_CHARS
        text = text[:_MAX_TEXT_CHARS]
        suffix = f"\n\n(truncated at {_MAX_TEXT_CHARS} characters)" if truncated else ""
        return f"Page: {title} ({url})\n\n{text}{suffix}"
    except PlaywrightError as exc:
        return f"Error: Browser failed to load {url}: {type(exc).__name__}"
    except Exception as exc:
        return f"Error scraping page: {type(exc).__name__}"


@beta
@tool(description="Take a screenshot of a web page using a headless browser")
def browser_screenshot(
    url: str, path: str, full_page: bool = False, timeout: int = _DEFAULT_TIMEOUT_S
) -> str:
    """
    Load a page in headless Chromium and save a PNG screenshot.

    Args:
        url: URL of the page to capture (http/https).
        path: Destination file path for the PNG (parent directories are
            created if missing).
        full_page: Capture the full scrollable page instead of just the
            viewport (default: False).
        timeout: Page load timeout in seconds (default: 30, max: 120).

    Returns:
        Confirmation with the saved file path, or a readable error string.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError  # type: ignore[import-untyped]
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    if not url or not url.strip():
        return "Error: No URL provided."
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    if not path or not path.strip():
        return "Error: No output path provided."

    timeout_ms = max(1, min(timeout, 120)) * 1000
    destination = Path(path.strip())

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=timeout_ms)
                page.screenshot(path=str(destination), full_page=full_page)
            finally:
                browser.close()
        return f"Screenshot of {url} saved to {destination}"
    except PlaywrightError as exc:
        return f"Error: Browser failed to capture {url}: {type(exc).__name__}"
    except OSError as exc:
        return f"Error: Could not write screenshot to {destination}: {type(exc).__name__}"
    except Exception as exc:
        return f"Error taking screenshot: {type(exc).__name__}"


__stability__ = "beta"

__all__ = [
    "browser_scrape_page",
    "browser_screenshot",
]
