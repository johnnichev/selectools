"""Tests for the shared SSRF URL validator (selectools._ssrf)."""

from __future__ import annotations

import pytest

from selectools._ssrf import check_url, validate_url


class TestValidateUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/",
            "http://127.0.0.1/",
            "http://127.0.0.1:8080/admin",
            "http://0.0.0.0/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
            "http://[::1]/",
        ],
    )
    def test_blocks_internal_targets(self, url: str) -> None:
        result = validate_url(url)
        assert result is not None
        assert result.startswith("Error")

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/file",
            "file:///etc/passwd",
            "gopher://example.com/",
            "http://",  # no hostname
        ],
    )
    def test_blocks_bad_scheme_or_missing_host(self, url: str) -> None:
        assert validate_url(url) is not None

    def test_unresolvable_host_reported(self) -> None:
        result = validate_url("http://does-not-resolve.invalid/")
        assert result is not None
        assert "resolve" in result.lower()

    def test_public_ip_literal_allowed(self) -> None:
        # A public IP literal needs no DNS and is not in any blocked range.
        assert validate_url("https://93.184.216.34/") is None


class TestCheckUrl:
    def test_raises_on_internal(self) -> None:
        with pytest.raises(ValueError):
            check_url("http://127.0.0.1/")

    def test_error_message_has_no_error_prefix(self) -> None:
        with pytest.raises(ValueError) as exc:
            check_url("ftp://example.com/")
        assert not str(exc.value).startswith("Error:")

    def test_public_ip_literal_passes(self) -> None:
        check_url("https://93.184.216.34/")  # no raise
