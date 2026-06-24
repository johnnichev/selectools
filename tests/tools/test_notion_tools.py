"""
Tests for Notion tools (notion_create_page, notion_search, notion_update_page).

The requests library is mocked via sys.modules -- no network, no API keys.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.notion_tools import notion_create_page, notion_search, notion_update_page

_KEY = "secret_fake_notion_key_do_not_leak"


class _FakeRequestException(Exception):
    pass


def _fake_response(status_code: int, payload: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _install_fake_requests(
    monkeypatch: pytest.MonkeyPatch,
    post: Optional[MagicMock] = None,
    patch_fn: Optional[MagicMock] = None,
) -> types.ModuleType:
    fake = types.ModuleType("requests")
    exceptions = types.ModuleType("requests.exceptions")
    exceptions.RequestException = _FakeRequestException  # type: ignore[attr-defined]
    fake.exceptions = exceptions  # type: ignore[attr-defined]
    fake.post = post or MagicMock()  # type: ignore[attr-defined]
    fake.patch = patch_fn or MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    monkeypatch.setitem(sys.modules, "requests.exceptions", exceptions)
    return fake


@pytest.fixture(autouse=True)
def _clear_notion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTION_API_KEY", raising=False)


class TestNotionCreatePage:
    def test_tool_metadata(self) -> None:
        assert notion_create_page.name == "notion_create_page"
        assert "Notion" in notion_create_page.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "requests", None)
        result = notion_create_page.function("parent-id", "Title", api_key=_KEY)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = notion_create_page.function("parent-id", "Title")
        assert "Error" in result
        assert "NOTION_API_KEY" in result

    def test_create_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200, {"id": "page-123", "url": "https://notion.so/page-123"}
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = notion_create_page.function(
            "parent-id", "My Page", content="line one\nline two", api_key=_KEY
        )
        assert "page-123" in result
        assert "My Page" in result
        url = post.call_args[0][0]
        assert url.endswith("/pages")
        headers = post.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {_KEY}"
        assert "Notion-Version" in headers

    def test_env_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"id": "p", "url": "u"}))
        _install_fake_requests(monkeypatch, post=post)
        monkeypatch.setenv("NOTION_API_KEY", _KEY)
        result = notion_create_page.function("parent-id", "Title")
        assert "created" in result

    def test_api_error_readable_no_key_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                404, {"code": "object_not_found", "message": "Could not find page"}
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = notion_create_page.function("bad-parent", "Title", api_key=_KEY)
        assert "Error" in result
        assert "404" in result
        assert "object_not_found" in result
        assert _KEY not in result

    def test_connection_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(side_effect=_FakeRequestException("boom"))
        _install_fake_requests(monkeypatch, post=post)
        result = notion_create_page.function("parent-id", "Title", api_key=_KEY)
        assert "Error" in result
        assert "Notion" in result
        assert _KEY not in result

    def test_missing_title_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = notion_create_page.function("parent-id", "", api_key=_KEY)
        assert "Error" in result


class TestNotionSearch:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = notion_search.function("roadmap")
        assert "Error" in result
        assert "NOTION_API_KEY" in result

    def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200,
                {
                    "results": [
                        {
                            "object": "page",
                            "id": "page-1",
                            "url": "https://notion.so/page-1",
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": "Roadmap 2026"}],
                                }
                            },
                        }
                    ]
                },
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = notion_search.function("roadmap", api_key=_KEY)
        assert "Roadmap 2026" in result
        assert "page-1" in result

    def test_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"results": []}))
        _install_fake_requests(monkeypatch, post=post)
        result = notion_search.function("nothing", api_key=_KEY)
        assert "No Notion pages found" in result

    def test_auth_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(401, {"code": "unauthorized", "message": "Invalid token"})
        )
        _install_fake_requests(monkeypatch, post=post)
        result = notion_search.function("roadmap", api_key=_KEY)
        assert "Error" in result
        assert "401" in result
        assert _KEY not in result


class TestNotionUpdatePage:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = notion_update_page.function("page-1", title="New")
        assert "Error" in result
        assert "NOTION_API_KEY" in result

    def test_nothing_to_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = notion_update_page.function("page-1", api_key=_KEY)
        assert "Error" in result
        assert "Nothing to update" in result

    def test_update_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_fn = MagicMock(return_value=_fake_response(200, {"id": "page-1"}))
        _install_fake_requests(monkeypatch, patch_fn=patch_fn)
        result = notion_update_page.function("page-1", title="Renamed", api_key=_KEY)
        assert "updated" in result
        assert "Renamed" in result
        assert patch_fn.call_args[0][0].endswith("/pages/page-1")

    def test_archive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_fn = MagicMock(return_value=_fake_response(200, {"id": "page-1"}))
        _install_fake_requests(monkeypatch, patch_fn=patch_fn)
        result = notion_update_page.function("page-1", archived=True, api_key=_KEY)
        assert "archived" in result

    def test_api_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_fn = MagicMock(
            return_value=_fake_response(
                404, {"code": "object_not_found", "message": "Could not find page"}
            )
        )
        _install_fake_requests(monkeypatch, patch_fn=patch_fn)
        result = notion_update_page.function("page-x", title="T", api_key=_KEY)
        assert "Error" in result
        assert "404" in result
        assert _KEY not in result
