"""Tests for the internal ISO-8601 time helpers (selectools._time).

Regression coverage for the Python 3.10 ``Z``-suffix bug: ``fromisoformat`` did
not accept a trailing ``Z`` until 3.11, so a UTC-designated timestamp returned by
Postgres/Supabase/Redis crashed the knowledge and checkpoint loaders on the
project's minimum interpreter (3.10).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from selectools._time import ensure_aware, parse_iso


class TestParseIso:
    def test_z_suffix_is_parsed_as_utc(self):
        # The bug: this raised ValueError on Python 3.10 before the fix.
        dt = parse_iso("2026-06-15T12:00:00Z")
        assert dt == datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert dt.tzinfo is not None

    def test_z_suffix_with_fractional_seconds(self):
        dt = parse_iso("2026-06-15T12:00:00.123456Z")
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timezone.utc.utcoffset(None)

    def test_explicit_offset_is_preserved(self):
        dt = parse_iso("2026-06-15T12:00:00+00:00")
        assert dt == datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_non_utc_offset_is_preserved(self):
        dt = parse_iso("2026-06-15T09:00:00-03:00")
        assert dt.utcoffset().total_seconds() == -3 * 3600

    def test_naive_string_stays_naive(self):
        # Awareness is preserved exactly as encoded; no implicit tz is attached.
        dt = parse_iso("2026-06-15T12:00:00")
        assert dt.tzinfo is None

    def test_datetime_passthrough(self):
        original = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert parse_iso(original) is original

    def test_invalid_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            parse_iso("not-a-timestamp")


class TestEnsureAware:
    def test_naive_gets_utc(self):
        naive = datetime(2026, 6, 15, 12, 0, 0)
        aware = ensure_aware(naive)
        assert aware.tzinfo is timezone.utc

    def test_aware_passes_through_unchanged(self):
        aware = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert ensure_aware(aware) == aware
        assert ensure_aware(aware).tzinfo is timezone.utc

    def test_parse_then_ensure_aware_round_trip(self):
        # The exact path sessions._iso_to_ts and Episode.from_dict now use.
        ts = ensure_aware(parse_iso("2026-06-15T12:00:00Z")).timestamp()
        assert ts == datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
