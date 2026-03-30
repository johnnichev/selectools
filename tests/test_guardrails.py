"""Comprehensive tests for the guardrails engine."""

from __future__ import annotations

import json

import pytest

from selectools.guardrails import (
    FormatGuardrail,
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    GuardrailsPipeline,
    LengthGuardrail,
    PIIGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
)

# ── TopicGuardrail ──────────────────────────────────────────────────────


class TestTopicGuardrail:
    def test_passes_clean_content(self) -> None:
        g = TopicGuardrail(deny=["politics", "religion"])
        result = g.check("Tell me about Python programming")
        assert result.passed is True

    def test_blocks_denied_topic(self) -> None:
        g = TopicGuardrail(deny=["politics"])
        result = g.check("What is your opinion on politics?")
        assert result.passed is False
        assert "politics" in (result.reason or "")

    def test_case_insensitive_by_default(self) -> None:
        g = TopicGuardrail(deny=["religion"])
        result = g.check("Tell me about RELIGION")
        assert result.passed is False

    def test_case_sensitive_mode(self) -> None:
        g = TopicGuardrail(deny=["Religion"], case_sensitive=True)
        result = g.check("tell me about religion")
        assert result.passed is True  # lowercase doesn't match

    def test_multiple_denied_topics(self) -> None:
        g = TopicGuardrail(deny=["politics", "gambling"])
        result = g.check("I want to discuss politics and gambling")
        assert result.passed is False
        assert "politics" in (result.reason or "")
        assert "gambling" in (result.reason or "")

    def test_empty_deny_list_passes(self) -> None:
        g = TopicGuardrail(deny=[])
        result = g.check("Anything goes")
        assert result.passed is True

    def test_word_boundary_matching(self) -> None:
        g = TopicGuardrail(deny=["war"])
        result = g.check("I need a software tool")
        assert result.passed is True  # "software" doesn't match "war"


# ── PIIGuardrail ────────────────────────────────────────────────────────


class TestPIIGuardrail:
    def test_detects_email(self) -> None:
        g = PIIGuardrail()
        matches = g.detect("Contact me at user@example.com")
        assert any(m.pii_type == "email" for m in matches)

    def test_detects_ssn(self) -> None:
        g = PIIGuardrail()
        matches = g.detect("My SSN is 123-45-6789")
        assert any(m.pii_type == "ssn" for m in matches)

    def test_detects_credit_card(self) -> None:
        g = PIIGuardrail()
        matches = g.detect("Card: 4111-1111-1111-1111")
        assert any(m.pii_type == "credit_card" for m in matches)

    def test_redacts_email(self) -> None:
        g = PIIGuardrail(action=GuardrailAction.REWRITE)
        result = g.check("Email: user@example.com ok")
        assert result.passed is False
        assert "user@example.com" not in result.content
        assert "[EMAIL:" in result.content

    def test_blocks_on_block_action(self) -> None:
        g = PIIGuardrail(action=GuardrailAction.BLOCK)
        result = g.check("SSN: 123-45-6789")
        assert result.passed is False
        assert "ssn" in (result.reason or "").lower()

    def test_no_pii_passes(self) -> None:
        g = PIIGuardrail()
        result = g.check("Hello, how are you?")
        assert result.passed is True

    def test_selective_detection(self) -> None:
        g = PIIGuardrail(detect=["email"])
        matches = g.detect("SSN 123-45-6789 email user@x.com")
        assert all(m.pii_type == "email" for m in matches)

    def test_custom_patterns(self) -> None:
        g = PIIGuardrail(custom_patterns={"badge_id": r"BADGE-\d{6}"})
        matches = g.detect("My badge is BADGE-123456")
        assert any(m.pii_type == "badge_id" for m in matches)

    def test_redact_overlapping_matches_no_double_redaction(self) -> None:
        """Regression: overlapping custom patterns must not double-redact content."""
        g = PIIGuardrail(custom_patterns={"digits_long": r"\d+", "digits_short": r"\d{3}"})
        content = "abc123def"
        result = g.redact(content)
        # The 3-digit run should appear exactly once in the redacted output
        # and the original 'def' tail must be present
        assert "def" in result
        assert result.count("[DIGITS") == 1

    def test_redact_non_overlapping_matches_all_redacted(self) -> None:
        """Non-overlapping matches must all be redacted independently."""
        g = PIIGuardrail(detect=["email", "ssn"])
        content = "Email: user@example.com and SSN: 123-45-6789"
        result = g.redact(content)
        assert "user@example.com" not in result
        assert "123-45-6789" not in result
        assert "[EMAIL:" in result
        assert "[SSN:" in result


# ── ToxicityGuardrail ──────────────────────────────────────────────────


class TestToxicityGuardrail:
    def test_blocks_toxic_content(self) -> None:
        g = ToxicityGuardrail(threshold=0.0)
        result = g.check("I will attack and harass them")
        assert result.passed is False

    def test_passes_clean_content(self) -> None:
        g = ToxicityGuardrail(threshold=0.0)
        result = g.check("Hello, how are you today?")
        assert result.passed is True

    def test_custom_blocklist(self) -> None:
        g = ToxicityGuardrail(blocklist={"badword"}, threshold=0.0)
        result = g.check("This contains badword in it")
        assert result.passed is False

    def test_threshold_sensitivity(self) -> None:
        g = ToxicityGuardrail(threshold=0.5)
        result = g.check("The word hate appears once")
        assert result.passed is True  # 1/16 < 0.5

    def test_score_method(self) -> None:
        g = ToxicityGuardrail()
        score = g.score("clean content with no issues")
        assert score == 0.0

    def test_matched_words(self) -> None:
        g = ToxicityGuardrail()
        matched = g.matched_words("hate and violence")
        assert "hate" in matched
        assert "violence" in matched

    def test_default_threshold_is_0_1(self) -> None:
        """Regression: docstring previously said default was 0.0 but actual default is 0.1."""
        g = ToxicityGuardrail()
        assert g.threshold == 0.1

    def test_negative_threshold_raises(self) -> None:
        """Regression: negative threshold would cause score > threshold always True."""
        with pytest.raises(ValueError, match="threshold must be between"):
            ToxicityGuardrail(threshold=-0.1)

    def test_threshold_above_one_raises(self) -> None:
        """Regression: threshold > 1.0 would silently never block any content."""
        with pytest.raises(ValueError, match="threshold must be between"):
            ToxicityGuardrail(threshold=1.1)

    def test_threshold_boundary_zero_valid(self) -> None:
        """threshold=0.0 and threshold=1.0 are both valid boundary values."""
        g0 = ToxicityGuardrail(threshold=0.0)
        assert g0.threshold == 0.0
        g1 = ToxicityGuardrail(threshold=1.0)
        assert g1.threshold == 1.0


# ── FormatGuardrail ─────────────────────────────────────────────────────


class TestFormatGuardrail:
    def test_valid_json_passes(self) -> None:
        g = FormatGuardrail(require_json=True)
        result = g.check('{"key": "value"}')
        assert result.passed is True

    def test_invalid_json_fails(self) -> None:
        g = FormatGuardrail(require_json=True)
        result = g.check("not json")
        assert result.passed is False
        assert "Invalid JSON" in (result.reason or "")

    def test_required_keys_present(self) -> None:
        g = FormatGuardrail(require_json=True, required_keys=["name", "age"])
        result = g.check('{"name": "Alice", "age": 30}')
        assert result.passed is True

    def test_required_keys_missing(self) -> None:
        g = FormatGuardrail(require_json=True, required_keys=["name", "age"])
        result = g.check('{"name": "Alice"}')
        assert result.passed is False
        assert "age" in (result.reason or "")

    def test_max_length_enforced(self) -> None:
        g = FormatGuardrail(max_length=10)
        result = g.check("a" * 20)
        assert result.passed is False

    def test_min_length_enforced(self) -> None:
        g = FormatGuardrail(min_length=10)
        result = g.check("short")
        assert result.passed is False

    def test_within_length_bounds(self) -> None:
        g = FormatGuardrail(min_length=2, max_length=100)
        result = g.check("hello world")
        assert result.passed is True

    def test_negative_max_length_raises(self) -> None:
        """Regression: negative max_length would cause all content to fail silently."""
        with pytest.raises(ValueError, match="max_length must be non-negative"):
            FormatGuardrail(max_length=-1)

    def test_negative_min_length_raises(self) -> None:
        """Regression: negative min_length is nonsensical and should be rejected."""
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            FormatGuardrail(min_length=-5)

    def test_max_less_than_min_raises(self) -> None:
        """Regression: max_length < min_length is an impossible constraint."""
        with pytest.raises(ValueError, match="max_length.*must be >= min_length"):
            FormatGuardrail(min_length=10, max_length=5)


# ── LengthGuardrail ────────────────────────────────────────────────────


class TestLengthGuardrail:
    def test_max_chars_blocks(self) -> None:
        g = LengthGuardrail(max_chars=5)
        result = g.check("too long")
        assert result.passed is False

    def test_max_chars_truncates_on_rewrite(self) -> None:
        g = LengthGuardrail(max_chars=5, action=GuardrailAction.REWRITE)
        result = g.check("too long string")
        assert result.passed is False
        assert len(result.content) == 5

    def test_min_chars(self) -> None:
        g = LengthGuardrail(min_chars=10)
        result = g.check("hi")
        assert result.passed is False

    def test_max_words_blocks(self) -> None:
        g = LengthGuardrail(max_words=3)
        result = g.check("one two three four five")
        assert result.passed is False

    def test_max_words_truncates_on_rewrite(self) -> None:
        g = LengthGuardrail(max_words=2, action=GuardrailAction.REWRITE)
        result = g.check("one two three four")
        assert result.content == "one two"

    def test_min_words(self) -> None:
        g = LengthGuardrail(min_words=5)
        result = g.check("one two")
        assert result.passed is False

    def test_within_bounds(self) -> None:
        g = LengthGuardrail(min_chars=1, max_chars=100, min_words=1, max_words=50)
        result = g.check("hello world")
        assert result.passed is True

    def test_negative_max_chars_raises(self) -> None:
        """Regression: negative max_chars causes all content to fail check."""
        with pytest.raises(ValueError, match="max_chars must be non-negative"):
            LengthGuardrail(max_chars=-1)

    def test_negative_min_chars_raises(self) -> None:
        """Regression: negative min_chars is nonsensical."""
        with pytest.raises(ValueError, match="min_chars must be non-negative"):
            LengthGuardrail(min_chars=-1)

    def test_negative_max_words_raises(self) -> None:
        """Regression: negative max_words causes all content to fail check."""
        with pytest.raises(ValueError, match="max_words must be non-negative"):
            LengthGuardrail(max_words=-5)

    def test_negative_min_words_raises(self) -> None:
        """Regression: negative min_words is nonsensical."""
        with pytest.raises(ValueError, match="min_words must be non-negative"):
            LengthGuardrail(min_words=-1)

    def test_max_chars_less_than_min_chars_raises(self) -> None:
        """Regression: max_chars < min_chars is an impossible constraint."""
        with pytest.raises(ValueError, match="max_chars.*must be >= min_chars"):
            LengthGuardrail(min_chars=10, max_chars=5)

    def test_max_words_less_than_min_words_raises(self) -> None:
        """Regression: max_words < min_words is an impossible constraint."""
        with pytest.raises(ValueError, match="max_words.*must be >= min_words"):
            LengthGuardrail(min_words=10, max_words=5)


# ── GuardrailsPipeline ──────────────────────────────────────────────────


class TestGuardrailsPipeline:
    def test_empty_pipeline_passes(self) -> None:
        pipeline = GuardrailsPipeline()
        result = pipeline.check_input("hello")
        assert result.passed is True

    def test_input_guardrail_blocks(self) -> None:
        pipeline = GuardrailsPipeline(
            input=[TopicGuardrail(deny=["politics"])],
        )
        with pytest.raises(GuardrailError) as exc_info:
            pipeline.check_input("Tell me about politics")
        assert "topic" in exc_info.value.guardrail_name

    def test_output_guardrail_blocks(self) -> None:
        pipeline = GuardrailsPipeline(
            output=[FormatGuardrail(require_json=True)],
        )
        with pytest.raises(GuardrailError):
            pipeline.check_output("not json")

    def test_rewrite_action_modifies_content(self) -> None:
        pipeline = GuardrailsPipeline(
            input=[PIIGuardrail(action=GuardrailAction.REWRITE)],
        )
        result = pipeline.check_input("Email: user@test.com")
        assert "user@test.com" not in result.content

    def test_warn_action_passes_through(self) -> None:
        g = TopicGuardrail(deny=["politics"], action=GuardrailAction.WARN)
        pipeline = GuardrailsPipeline(input=[g])
        result = pipeline.check_input("Tell me about politics")
        assert result.passed is True

    def test_chain_executes_in_order(self) -> None:
        pipeline = GuardrailsPipeline(
            input=[
                PIIGuardrail(action=GuardrailAction.REWRITE),
                LengthGuardrail(max_chars=200, action=GuardrailAction.REWRITE),
            ],
        )
        result = pipeline.check_input("Email: user@test.com is short enough")
        assert result.passed is True
        assert "user@test.com" not in result.content


# ── GuardrailResult ─────────────────────────────────────────────────────


class TestGuardrailResult:
    def test_dataclass_fields(self) -> None:
        r = GuardrailResult(passed=True, content="test")
        assert r.passed is True
        assert r.content == "test"
        assert r.reason is None
        assert r.guardrail_name is None


# ── GuardrailError ──────────────────────────────────────────────────────


class TestGuardrailError:
    def test_error_message(self) -> None:
        err = GuardrailError("topic", "Denied topics detected")
        assert "topic" in str(err)
        assert "Denied topics detected" in str(err)
        assert err.guardrail_name == "topic"
        assert err.reason == "Denied topics detected"


# ── Custom guardrail ────────────────────────────────────────────────────


class TestCustomGuardrail:
    def test_subclass_with_custom_check(self) -> None:
        class NoNumbersGuardrail(Guardrail):
            name = "no_numbers"
            action = GuardrailAction.BLOCK

            def check(self, content: str) -> GuardrailResult:
                import re

                if re.search(r"\d", content):
                    return GuardrailResult(
                        passed=False,
                        content=content,
                        reason="Numbers not allowed",
                        guardrail_name=self.name,
                    )
                return GuardrailResult(passed=True, content=content, guardrail_name=self.name)

        g = NoNumbersGuardrail()
        assert g.check("hello").passed is True
        assert g.check("hello 123").passed is False


# ── Regression: falsy empty-list/set bugs ──────────────────────────────


class TestPIIGuardrailEmptyDetectList:
    def test_empty_detect_list_detects_nothing(self) -> None:
        """Regression: PIIGuardrail(detect=[]) must use NO patterns, not all patterns.

        Previously, an empty detect list was falsy, so the 'if detect:' branch was
        skipped and ALL built-in patterns were loaded — the opposite of the intent.
        """
        g = PIIGuardrail(detect=[])
        result = g.check("user@example.com")
        # With detect=[], no patterns are active, so PII should not be detected
        assert result.passed is True

    def test_empty_detect_list_no_patterns(self) -> None:
        """PIIGuardrail(detect=[]) must have empty _patterns dict."""
        g = PIIGuardrail(detect=[])
        assert g._patterns == {}

    def test_none_detect_loads_all_defaults(self) -> None:
        """PIIGuardrail(detect=None) must still use all built-in patterns."""
        g = PIIGuardrail(detect=None)
        assert len(g._patterns) > 0
        assert "email" in g._patterns


class TestToxicityGuardrailEmptyBlocklist:
    def test_empty_blocklist_detects_nothing(self) -> None:
        """Regression: ToxicityGuardrail(blocklist=set()) must use NO blocklist words.

        Previously, an empty set was falsy, so 'blocklist or _DEFAULT_BLOCKLIST' fell
        back to the default list — the opposite of the intent.
        """
        g = ToxicityGuardrail(blocklist=set(), threshold=0.0)
        result = g.check("kill murder attack bomb")
        # With empty blocklist, score should be 0.0 and content should pass
        assert result.passed is True

    def test_empty_blocklist_score_is_zero(self) -> None:
        """ToxicityGuardrail(blocklist=set()) must have score=0.0 for any input."""
        g = ToxicityGuardrail(blocklist=set(), threshold=0.0)
        assert g.score("kill murder attack") == 0.0

    def test_none_blocklist_loads_defaults(self) -> None:
        """ToxicityGuardrail(blocklist=None) must still load the default blocklist."""
        g = ToxicityGuardrail(blocklist=None)
        assert len(g._blocklist) > 0
