"""Tests for the shared built-in PII patterns."""

from selectools._pii_patterns import PII_PATTERNS


def test_shared_email_pattern_matches_valid_addresses() -> None:
    assert PII_PATTERNS["email"].search("Reach me at hello.world+test@example.co.uk")


def test_shared_email_pattern_rejects_pipe_domain_suffixes() -> None:
    assert PII_PATTERNS["email"].search("Reach me at hello@example.|a") is None


def test_shared_patterns_cover_the_guardrail_defaults() -> None:
    assert {"email", "phone_us", "ssn", "credit_card", "ipv4"} <= PII_PATTERNS.keys()
