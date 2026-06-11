#!/usr/bin/env python3
"""
Quick-reply button flows for deferred confirmation (issue #82).

Example 100 covers FREE-TEXT confirmation ("sim", "yes") through the
RegexConfirmParser. But chat channels with quick-reply buttons (Twilio
WhatsApp quick replies, Telegram inline keyboards) deliver the decision as
a STRUCTURED payload — the webhook already knows the intent, and parsing
the button label as text would be both redundant and locale-fragile.

`pop_if_intent` bypasses the parser entirely:

1. intent="confirm"  -> atomically claims and runs the stashed executor.
2. intent="cancel"   -> claims and drops it (status "cancelled").
3. intent="ignore"   -> PRESERVES the pending but tightens its TTL to a
   few seconds (status "ignored") — a mis-tap is recoverable, yet the
   destructive op cannot stay armed for the original window.
4. expected_kind / expected_id pins: a button minted for a DIFFERENT
   prompt (stale replay, out-of-order delivery) returns "kind_mismatch"
   and PRESERVES the pending — unlike digest_mismatch, which disarms,
   because here the user never addressed THIS pending at all.

`tighten_ttl` is also available standalone: it only ever SHORTENS the
window and keeps the executor, so the action stays confirmable inside
the tightened window.

No API key needed. Runs offline against the store directly.

Run: python examples/110_quick_reply_buttons.py
"""

from __future__ import annotations

import time
from typing import Any, Dict

from selectools.pending import InMemoryPendingStore

# ---------------------------------------------------------------------------
# A fake invoice "database" and the destructive side effect
# ---------------------------------------------------------------------------

_INVOICES: Dict[str, Dict[str, Any]] = {
    "INV-42": {"amount": 199.0, "customer": "ACME"},
}


def _delete_invoice(invoice_id: str) -> str:
    _INVOICES.pop(invoice_id, None)
    return f"Deleted invoice {invoice_id}"


# ---------------------------------------------------------------------------
# Simulated button webhook
# ---------------------------------------------------------------------------


def handle_button_webhook(
    store: InMemoryPendingStore,
    user_id: str,
    intent: str,
    expected_kind: str,
) -> str:
    """What a Twilio quick-reply webhook does with a ButtonPayload.

    The channel layer classified the payload into a structured intent and
    derived ``expected_kind`` from the payload prefix — the kind the button
    was minted for. No text parsing happens here.
    """
    outcome = store.pop_if_intent(user_id, intent, expected_kind=expected_kind)
    if outcome is None:
        return "Nothing pending (or a duplicate delivery already handled it)."
    if outcome.executed:
        return f"Done: {outcome.result}"
    if outcome.status == "cancelled":
        return f"Cancelled: {outcome.record.preview}"
    if outcome.status == "kind_mismatch":
        # PRESERVED — the button belonged to a different prompt.
        return "That button was for an earlier prompt. Your pending action is unchanged."
    if outcome.status == "ignored":
        # PRESERVED with a tightened TTL.
        remaining = outcome.record.expires_at - time.time()
        return f"Okay — reply sim/nao within {remaining:.0f}s to finish, or it lapses."
    if outcome.status == "expired":
        return f"That confirmation expired: {outcome.record.preview}. Please start over."
    return f"Could not confirm safely ({outcome.status}). Please request it again."


def main() -> None:
    store = InMemoryPendingStore()
    user = "user-1"

    # Turn 1: the agent's destructive tool stashed a pending action and the
    # channel sent the user a quick-reply template (Confirm / Cancel).
    record = store.stash(
        user,
        kind="delete_invoice",
        preview="Delete invoice INV-42 (R$199.00, ACME)",
        executor=lambda: _delete_invoice("INV-42"),
        args={"invoice_id": "INV-42"},
        ttl_seconds=300.0,
    )
    print(f"[stash]   pending: {record.preview}")

    # A STALE button from an earlier "cancel subscription" prompt replays.
    # The pin mismatch preserves the live pending.
    print("[stale ]  ", handle_button_webhook(store, user, "confirm", "cancel_subscription"))
    assert store.get(user) is not None

    # The user taps a button the template doesn't recognize (or Twilio sends
    # a malformed payload): preserve, but tighten the TTL.
    print("[ignore]  ", handle_button_webhook(store, user, "ignore", "delete_invoice"))
    assert store.get(user) is not None

    # Within the tightened window the user taps the real Confirm button.
    print("[confirm] ", handle_button_webhook(store, user, "confirm", "delete_invoice"))
    assert "INV-42" not in _INVOICES

    # Duplicate webhook delivery: the twin claim finds nothing — exactly-once.
    print("[dup   ]  ", handle_button_webhook(store, user, "confirm", "delete_invoice"))

    # Standalone tighten_ttl: shorten a long-lived window after a reminder.
    store.stash(
        user,
        kind="cancel_subscription",
        preview="Cancel the ACME subscription",
        executor=lambda: "Subscription cancelled",
        ttl_seconds=3600.0,
    )
    tightened = store.tighten_ttl(user, 15.0)
    assert tightened is not None
    print(f"[tighten] window now {tightened.expires_at - time.time():.0f}s (was 3600s)")


if __name__ == "__main__":
    main()
