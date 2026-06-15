"""Internal time helpers.

Centralizes ISO-8601 timestamp parsing so every persistence backend handles a
trailing ``Z`` (UTC designator) the same way.

``datetime.fromisoformat`` did not accept the ``Z`` suffix until Python 3.11.
On Python 3.10 (the project's minimum) a ``Z``-suffixed timestamp such as
``"2026-06-15T12:00:00Z"`` raises ``ValueError``. Postgres/Supabase/Redis and
other stores routinely return that form, so parsing it inline crashed the
knowledge and checkpoint loaders on 3.10. ``parse_iso`` normalizes the suffix
before delegating, so the same code works on every supported interpreter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union


def parse_iso(value: Union[str, datetime]) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` (UTC).

    Accepts either a string or an already-parsed ``datetime`` (returned as-is,
    so callers reading columns that may already be typed do not need their own
    ``isinstance`` guard). Timezone awareness is preserved exactly as encoded:
    an offset-bearing string (including ``Z``) yields an aware datetime; a naive
    string yields a naive datetime. Use :func:`ensure_aware` when a guaranteed
    aware value is required.
    """
    if isinstance(value, datetime):
        return value
    s = str(value)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def ensure_aware(dt: datetime, *, tz: timezone = timezone.utc) -> datetime:
    """Attach ``tz`` (default UTC) to a naive datetime; pass aware ones through."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt
