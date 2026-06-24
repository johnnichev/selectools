"""
Tests for browser tools (browser_scrape_page, browser_screenshot).

playwright is mocked via sys.modules -- no browser binary, no network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.browser_tools import browser_scrape_page, browser_screenshot


class _FakePlaywrightError(Exception):
    pass


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch, page: MagicMock, patch_ssrf: bool = True
) -> MagicMock:
    """Install a fake playwright.sync_api whose chromium page is *page*.

    By default the SSRF ``validate_url`` guard is stubbed to a no-op so these
    tests stay network-free (no real DNS resolution). SSRF-specific tests pass
    ``patch_ssrf=False`` to exercise the real guard.
    """
    browser = MagicMock()
    browser.new_page.return_value = page
    handle = MagicMock()
    handle.chromium.launch.return_value = browser

    sync_playwright = MagicMock()
    sync_playwright.return_value.__enter__.return_value = handle
    sync_playwright.return_value.__exit__.return_value = False

    playwright = types.ModuleType("playwright")
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = sync_playwright  # type: ignore[attr-defined]
    sync_api.Error = _FakePlaywrightError  # type: ignore[attr-defined]
    playwright.sync_api = sync_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    if patch_ssrf:
        monkeypatch.setattr("selectools.toolbox.browser_tools.validate_url", lambda _url: None)
    return browser


class TestBrowserSSRF:
    """The SSRF guard blocks internal targets in both browser tools."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1:8080/",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://[::1]/",
        ],
    )
    def test_scrape_blocks_internal_targets(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        page = MagicMock()
        browser = _install_fake_playwright(monkeypatch, page, patch_ssrf=False)
        result = browser_scrape_page.function(url)
        assert "Error" in result
        # Browser is never launched for a blocked URL.
        browser.new_page.assert_not_called()
        page.goto.assert_not_called()

    def test_screenshot_blocks_internal_targets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = MagicMock()
        _install_fake_playwright(monkeypatch, page, patch_ssrf=False)
        result = browser_screenshot.function("http://127.0.0.1/", str(tmp_path / "p.png"))
        assert "Error" in result
        page.goto.assert_not_called()
        page.screenshot.assert_not_called()


class TestBrowserScrapePage:
    def test_tool_metadata(self) -> None:
        assert browser_scrape_page.name == "browser_scrape_page"
        assert "browser" in browser_scrape_page.description.lower()

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        result = browser_scrape_page.function("https://example.com")
        assert "Error" in result
        assert "selectools[browser]" in result

    def test_scrape_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = MagicMock()
        page.title.return_value = "Example Domain"
        page.inner_text.return_value = "Example Domain\nThis domain is for examples."
        browser = _install_fake_playwright(monkeypatch, page)
        result = browser_scrape_page.function("https://example.com")
        assert "Example Domain" in result
        assert "for examples" in result
        page.goto.assert_called_once()
        assert page.goto.call_args[0][0] == "https://example.com"
        browser.close.assert_called_once()

    def test_text_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = MagicMock()
        page.title.return_value = "Big"
        page.inner_text.return_value = "x" * 10000
        _install_fake_playwright(monkeypatch, page)
        result = browser_scrape_page.function("https://example.com")
        assert "truncated at 5000 characters" in result
        assert "x" * 5001 not in result

    def test_empty_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = MagicMock()
        page.title.return_value = ""
        page.inner_text.return_value = "   "
        _install_fake_playwright(monkeypatch, page)
        result = browser_scrape_page.function("https://example.com")
        assert "No visible text" in result

    def test_invalid_url_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_playwright(monkeypatch, MagicMock())
        assert "Error" in browser_scrape_page.function("ftp://example.com")
        assert "Error" in browser_scrape_page.function("")

    def test_navigation_error_readable_and_browser_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = MagicMock()
        page.goto.side_effect = _FakePlaywrightError("net::ERR_NAME_NOT_RESOLVED")
        browser = _install_fake_playwright(monkeypatch, page)
        result = browser_scrape_page.function("https://does-not-resolve.invalid")
        assert "Error" in result
        assert "failed to load" in result
        browser.close.assert_called_once()

    def test_timeout_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = MagicMock()
        page.title.return_value = "t"
        page.inner_text.return_value = "body"
        _install_fake_playwright(monkeypatch, page)
        browser_scrape_page.function("https://example.com", timeout=9999)
        assert page.goto.call_args[1]["timeout"] == 120000


class TestBrowserScreenshot:
    def test_tool_metadata(self) -> None:
        assert browser_screenshot.name == "browser_screenshot"
        assert "screenshot" in browser_screenshot.description.lower()

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        result = browser_screenshot.function("https://example.com", "/tmp/shot.png")
        assert "Error" in result
        assert "selectools[browser]" in result

    def test_screenshot_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        page = MagicMock()
        browser = _install_fake_playwright(monkeypatch, page)
        destination = tmp_path / "shots" / "page.png"
        result = browser_screenshot.function("https://example.com", str(destination))
        assert "saved to" in result
        assert str(destination) in result
        page.screenshot.assert_called_once_with(path=str(destination), full_page=False)
        assert destination.parent.is_dir()
        browser.close.assert_called_once()

    def test_full_page_flag_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = MagicMock()
        _install_fake_playwright(monkeypatch, page)
        browser_screenshot.function("https://example.com", str(tmp_path / "p.png"), full_page=True)
        assert page.screenshot.call_args[1]["full_page"] is True

    def test_missing_path_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_playwright(monkeypatch, MagicMock())
        result = browser_screenshot.function("https://example.com", "  ")
        assert "Error" in result

    def test_invalid_url_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_playwright(monkeypatch, MagicMock())
        result = browser_screenshot.function("not-a-url", "/tmp/shot.png")
        assert "Error" in result

    def test_capture_error_readable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        page = MagicMock()
        page.goto.side_effect = _FakePlaywrightError("Timeout 30000ms exceeded")
        browser = _install_fake_playwright(monkeypatch, page)
        result = browser_screenshot.function("https://slow.example.com", str(tmp_path / "p.png"))
        assert "Error" in result
        assert "failed to capture" in result
        browser.close.assert_called_once()
