"""
Tests for Linear tools (linear_create_issue, linear_list_issues, linear_update_issue).

The requests library is mocked via sys.modules -- no network, no API keys.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.linear_tools import (
    linear_create_issue,
    linear_list_issues,
    linear_update_issue,
)

_KEY = "lin_api_fake_key_do_not_leak"


class _FakeRequestException(Exception):
    pass


def _fake_response(status_code: int, payload: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _install_fake_requests(
    monkeypatch: pytest.MonkeyPatch, post: Optional[MagicMock] = None
) -> types.ModuleType:
    fake = types.ModuleType("requests")
    exceptions = types.ModuleType("requests.exceptions")
    exceptions.RequestException = _FakeRequestException  # type: ignore[attr-defined]
    fake.exceptions = exceptions  # type: ignore[attr-defined]
    fake.post = post or MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    monkeypatch.setitem(sys.modules, "requests.exceptions", exceptions)
    return fake


@pytest.fixture(autouse=True)
def _clear_linear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)


class TestLinearCreateIssue:
    def test_tool_metadata(self) -> None:
        assert linear_create_issue.name == "linear_create_issue"
        assert "Linear" in linear_create_issue.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "requests", None)
        result = linear_create_issue.function("team-1", "Bug", api_key=_KEY)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = linear_create_issue.function("team-1", "Bug")
        assert "Error" in result
        assert "LINEAR_API_KEY" in result

    def test_create_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200,
                {
                    "data": {
                        "issueCreate": {
                            "success": True,
                            "issue": {
                                "id": "uuid-1",
                                "identifier": "ENG-42",
                                "title": "Fix the bug",
                                "url": "https://linear.app/x/issue/ENG-42",
                            },
                        }
                    }
                },
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_create_issue.function(
            "team-1", "Fix the bug", description="details", api_key=_KEY
        )
        assert "ENG-42" in result
        assert "Fix the bug" in result

        body = json.loads(post.call_args[1]["data"])
        assert body["variables"]["input"]["teamId"] == "team-1"
        assert post.call_args[1]["headers"]["Authorization"] == _KEY

    def test_graphql_error_readable_no_key_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200, {"errors": [{"message": "Argument teamId is invalid"}]}
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_create_issue.function("bad-team", "Bug", api_key=_KEY)
        assert "Error" in result
        assert "teamId is invalid" in result
        assert _KEY not in result

    def test_auth_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(401, {}))
        _install_fake_requests(monkeypatch, post=post)
        result = linear_create_issue.function("team-1", "Bug", api_key=_KEY)
        assert "Error" in result
        assert "authentication" in result.lower()
        assert _KEY not in result

    def test_connection_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(side_effect=_FakeRequestException("boom"))
        _install_fake_requests(monkeypatch, post=post)
        result = linear_create_issue.function("team-1", "Bug", api_key=_KEY)
        assert "Error" in result
        assert "Linear" in result

    def test_missing_title_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = linear_create_issue.function("team-1", "", api_key=_KEY)
        assert "Error" in result


class TestLinearListIssues:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = linear_list_issues.function()
        assert "Error" in result
        assert "LINEAR_API_KEY" in result

    def test_list_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200,
                {
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "uuid-1",
                                    "identifier": "ENG-1",
                                    "title": "First issue",
                                    "url": "u1",
                                    "state": {"name": "In Progress"},
                                    "assignee": {"name": "John"},
                                },
                                {
                                    "id": "uuid-2",
                                    "identifier": "ENG-2",
                                    "title": "Second issue",
                                    "url": "u2",
                                    "state": {"name": "Todo"},
                                    "assignee": None,
                                },
                            ]
                        }
                    }
                },
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_list_issues.function(api_key=_KEY)
        assert "ENG-1" in result
        assert "In Progress" in result
        assert "John" in result
        assert "ENG-2" in result
        assert "unassigned" in result

    def test_team_filter_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"data": {"issues": {"nodes": []}}}))
        _install_fake_requests(monkeypatch, post=post)
        linear_list_issues.function(team_id="team-9", api_key=_KEY)
        body = json.loads(post.call_args[1]["data"])
        assert body["variables"]["filter"]["team"]["id"]["eq"] == "team-9"

    def test_no_issues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"data": {"issues": {"nodes": []}}}))
        _install_fake_requests(monkeypatch, post=post)
        result = linear_list_issues.function(api_key=_KEY)
        assert "No Linear issues" in result

    def test_api_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(200, {"errors": [{"message": "rate limited"}]})
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_list_issues.function(api_key=_KEY)
        assert "Error" in result
        assert "rate limited" in result


class TestLinearUpdateIssue:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = linear_update_issue.function("uuid-1", title="New")
        assert "Error" in result
        assert "LINEAR_API_KEY" in result

    def test_nothing_to_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = linear_update_issue.function("uuid-1", api_key=_KEY)
        assert "Error" in result
        assert "Nothing to update" in result

    def test_update_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(
                200,
                {
                    "data": {
                        "issueUpdate": {
                            "success": True,
                            "issue": {
                                "identifier": "ENG-42",
                                "title": "Renamed",
                                "url": "u",
                                "state": {"name": "Done"},
                            },
                        }
                    }
                },
            )
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_update_issue.function(
            "uuid-1", title="Renamed", state_id="state-1", api_key=_KEY
        )
        assert "ENG-42" in result
        assert "Renamed" in result
        assert "Done" in result

        body = json.loads(post.call_args[1]["data"])
        assert body["variables"]["input"]["title"] == "Renamed"
        assert body["variables"]["input"]["stateId"] == "state-1"

    def test_api_error_readable_no_key_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(200, {"errors": [{"message": "Entity not found"}]})
        )
        _install_fake_requests(monkeypatch, post=post)
        result = linear_update_issue.function("uuid-x", title="T", api_key=_KEY)
        assert "Error" in result
        assert "Entity not found" in result
        assert _KEY not in result
