"""Tests for shared toolbox HTTP helpers."""

from unittest.mock import MagicMock

from selectools.toolbox._http import DEFAULT_TIMEOUT, USER_AGENT, format_api_error


def test_shared_http_defaults_match_the_current_package() -> None:
    assert DEFAULT_TIMEOUT == 30
    assert "selectools/1.3.0" in USER_AGENT


def test_format_api_error_includes_service_response_details() -> None:
    response = MagicMock(status_code=404)
    response.json.return_value = {"code": "missing", "message": "Not found"}

    assert (
        format_api_error("Example", response)
        == "Error: Example API returned HTTP 404 (missing: Not found)"
    )


def test_format_api_error_handles_non_json_responses() -> None:
    response = MagicMock(status_code=502)
    response.json.side_effect = ValueError("invalid JSON")

    assert format_api_error("Example", response) == "Error: Example API returned HTTP 502"
