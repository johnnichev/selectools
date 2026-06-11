"""
Deferred confirmation flow for chat-channel destructive tools (issue #58).

The built-in approval gate (``Tool.requires_approval`` +
``AgentConfig.confirm_action``) is synchronous and in-loop: it blocks the
agent worker until a confirm decision arrives. That model is wrong for
chat-channel agents (WhatsApp, Telegram, Slack DM, SMS) where the user's
"yes" arrives as a *separate webhook turn* — the agent loop has already
returned and there is nothing to block on.

This module provides the out-of-loop pattern every chat-channel consumer
otherwise rebuilds (proven shape: Sheriff's ``core/pending_actions.py``):

1. A destructive tool does NOT execute on the first call. It stashes a
   pending action (preview + executor + TTL) via :func:`stash_pending` and
   returns a :class:`PendingConfirmation` so the LLM asks the user.
2. The next webhook turn goes through :class:`ChannelAgent.ask_channel`.
   If the message confirms, the stashed executor runs and the LLM is
   bypassed; if it cancels, the action is dropped; anything else drops the
   pending (the user moved on) and falls through to the normal agent path.

Button flows (issue #82): quick-reply buttons (Twilio, Telegram inline
keyboards) deliver the decision as a STRUCTURED payload, not free text.
``pop_if_intent`` bypasses the parser and takes ``"confirm" | "cancel" |
"ignore"`` plus optional ``expected_kind``/``expected_id`` pins minted into
the button; a pin mismatch PRESERVES the pending (``kind_mismatch``) — a
stale button replay must neither fire nor disarm the user's live flow. An
"ignore" tap keeps the pending but tightens its TTL via ``tighten_ttl``
(default 10s) so a mis-tap is recoverable without leaving a destructive op
armed for the original window.

Safety model (rpelevin's review spec): the confirmation is bound to ONE
exact proposed side effect. :class:`PendingAction` records who asked
(user/channel/conversation scope), what was previewed, a canonical
``args_digest`` of the proposed arguments, a TTL window, and a
pending → confirmed | cancelled | expired | consumed status lifecycle.
``pop_if_confirmed`` executes only when ALL guards pass — same scope,
still pending, not expired, digest unchanged — and duplicate webhook
delivery executes ONCE (atomic claim).

Executor closures and Redis: a closure cannot be serialized, and pickling
callables into Redis is a code-injection footgun. :class:`RedisPendingStore`
therefore persists only the RECORD (id, scope, kind, preview, args, digest,
timestamps) and keeps executor closures in a process-local registry keyed by
``pending_action_id``. When the confirming webhook lands on a different
process, the closure is gone — register an executor *factory* per ``kind``
via :meth:`RedisPendingStore.register_executor_factory`; it rebuilds the
executor from the persisted record (``record.args``). Without a closure or
factory the action is claimed-and-dropped (``no_executor``), never executed.

Example::

    from selectools import Agent
    from selectools.pending import (
        ChannelAgent, InMemoryPendingStore, PendingConfirmation, stash_pending,
    )

    @tool()
    def delete_invoice(invoice_id: str) -> PendingConfirmation:
        preview = f"Delete invoice {invoice_id}"
        stash_pending(
            kind="delete_invoice",
            preview=preview,
            executor=lambda: do_delete(invoice_id),
            args={"invoice_id": invoice_id},
        )
        return PendingConfirmation(
            action="delete_invoice",
            preview=preview,
            user_prompt="Reply 'yes' to confirm or 'no' to cancel.",
        )

    channel = ChannelAgent(agent, store=InMemoryPendingStore())
    channel.ask_channel("user-1", "delete invoice INV-42")  # turn 1: preview
    channel.ask_channel("user-1", "yes")                    # turn 2: executes
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

from .results import ToolResult
from .stability import beta, register_stability

if TYPE_CHECKING:
    from .agent.core import Agent
from .types import AgentResult, Message, Role

logger = logging.getLogger(__name__)

__stability__ = "beta"

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_IGNORE_TTL_SECONDS",
    "PendingAction",
    "PendingActionExistsError",
    "ConfirmOutcome",
    "ConfirmParser",
    "RegexConfirmParser",
    "PendingActionStore",
    "InMemoryPendingStore",
    "RedisPendingStore",
    "PendingConfirmation",
    "ChannelAgent",
    "stash_pending",
    "compute_args_digest",
]

# Short TTL by default (Sheriff bug-hunt #10): a tight window keeps a casual
# "ok"/"yes" in a later, unrelated conversational turn from firing a
# destructive action proposed minutes earlier. Users who take longer simply
# re-ask — annoying, not destructive.
DEFAULT_TTL_SECONDS: float = 60.0
register_stability("DEFAULT_TTL_SECONDS", "beta")

# TTL applied by the "ignore" intent (issue #82, Sheriff round-13 F11): an
# unrecognized/stale button tap PRESERVES the pending but tightens its window
# to a few seconds. Pre-fix in Sheriff, the original TTL (up to 24h) stuck,
# so a user who tapped the wrong button could come back minutes later and a
# "sim" to an unrelated question auto-fired the stale destructive action.
# 10s keeps the legitimate "oh, let me retype my answer" path alive without
# leaving a destructive op armed for a long window.
DEFAULT_IGNORE_TTL_SECONDS: float = 10.0
register_stability("DEFAULT_IGNORE_TTL_SECONDS", "beta")

# Statuses for the PendingAction lifecycle.
_PENDING = "pending"
_CONFIRMED = "confirmed"
_CANCELLED = "cancelled"
_EXPIRED = "expired"
_CONSUMED = "consumed"

Executor = Callable[[], str]
ExecutorFactory = Callable[["PendingAction"], Executor]


@beta
class PendingActionExistsError(RuntimeError):
    """Raised by ``stash`` when an unexpired pending action already exists.

    Refusing to overwrite is deliberate (Sheriff bug-hunt iter-2 T4): if the
    LLM calls two destructive tools in one turn, "last wins" means the user
    sees the preview for action A, says "yes", and action B fires. Tools
    should catch this and ask the user to finish the prior confirmation.
    """


@beta
def compute_args_digest(args: Optional[Mapping[str, Any]]) -> str:
    """Canonical SHA-256 digest of a proposed side effect's arguments.

    Key order does not affect the digest. ``None`` and ``{}`` are
    equivalent (no arguments).
    """
    canonical = json.dumps(dict(args or {}), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@beta
@dataclass(frozen=True)
class PendingAction:
    """One proposed destructive side effect awaiting user confirmation.

    The record binds the confirmation to ONE exact action: the scope that
    requested it, the preview the user was shown, and a canonical digest of
    the proposed arguments. Confirmation guards check all of them.

    Attributes:
        pending_action_id: Unique id for this pending action.
        user_id: User who must confirm.
        kind: Tool/action kind (e.g. ``"delete_invoice"``).
        preview: Human-readable description shown to the user.
        args_digest: ``compute_args_digest`` of the proposed arguments.
        requested_at: Epoch seconds when the action was stashed.
        expires_at: Epoch seconds after which confirmation is refused.
        channel_id: Optional channel scope (e.g. ``"whatsapp"``).
        conversation_id: Optional conversation/thread scope.
        parser_version: Version/locale of the confirm parser in effect when
            the preview was issued (audit: which patterns gated this action).
        status: ``pending | confirmed | cancelled | expired | consumed``.
        outcome: Final executor outcome text once consumed.
        args: JSON-serializable proposed arguments. Persisted so a
            multi-process store can rebuild the executor via a factory.
    """

    pending_action_id: str
    user_id: str
    kind: str
    preview: str
    args_digest: str
    requested_at: float
    expires_at: float
    channel_id: Optional[str] = None
    conversation_id: Optional[str] = None
    parser_version: str = ""
    status: str = _PENDING
    outcome: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Whether the confirmation window has closed."""
        return (time.time() if now is None else now) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation (for Redis persistence)."""
        return {
            "pending_action_id": self.pending_action_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "preview": self.preview,
            "args_digest": self.args_digest,
            "requested_at": self.requested_at,
            "expires_at": self.expires_at,
            "channel_id": self.channel_id,
            "conversation_id": self.conversation_id,
            "parser_version": self.parser_version,
            "status": self.status,
            "outcome": self.outcome,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingAction":
        """Inverse of :meth:`to_dict`."""
        return cls(
            pending_action_id=data["pending_action_id"],
            user_id=data["user_id"],
            kind=data["kind"],
            preview=data["preview"],
            args_digest=data["args_digest"],
            requested_at=data["requested_at"],
            expires_at=data["expires_at"],
            channel_id=data.get("channel_id"),
            conversation_id=data.get("conversation_id"),
            parser_version=data.get("parser_version", ""),
            status=data.get("status", _PENDING),
            outcome=data.get("outcome"),
            args=data.get("args"),
        )


@beta
@dataclass(frozen=True)
class ConfirmOutcome:
    """Result of a ``pop_if_confirmed`` or ``pop_if_intent`` claim.

    ``status`` values:

    - ``"executed"``: all guards passed; ``result`` holds the executor output.
    - ``"expired"``: the "yes" arrived after the TTL. Nothing executed.
    - ``"digest_mismatch"``: the action changed after the preview. Nothing
      executed; a fresh confirmation cycle is required.
    - ``"no_executor"``: no closure and no factory could run the action
      (e.g. the confirming webhook landed on a fresh process). Nothing
      executed; a fresh confirmation cycle is required.
    - ``"failed"``: guards passed but the executor RAISED. The record is
      consumed (one-shot — a duplicate webhook neither retries nor falls
      through to the LLM). ``record.outcome`` carries only the exception
      type name (``"error: <TypeName>"``); full detail goes to logging.
    - ``"cancelled"`` (``pop_if_intent`` only): a structured cancel claimed
      and dropped the pending. Nothing executed.
    - ``"kind_mismatch"`` (``pop_if_intent`` only): the button's
      ``expected_kind``/``expected_id`` does not match the pending action.
      Nothing executed and — unlike every other status — the pending action
      is PRESERVED. Rationale: a digest mismatch means THIS action mutated
      after the user previewed it, so the confirmation is tainted and the
      action is disarmed; a kind/id mismatch means the button belonged to a
      DIFFERENT prompt entirely (stale Twilio replay, out-of-order
      delivery), so the observed pending is still the user's live,
      previewed intent and a stale button must be able to neither fire nor
      disarm it. The user's next text reply can still confirm or cancel.
      Both pins share this one status: an ``expected_id`` mismatch also
      reports ``kind_mismatch``, not a separate code.
    - ``"ignored"`` (``pop_if_intent`` only): an "ignore" (or unrecognized)
      intent. Nothing executed; the pending action is PRESERVED with its
      TTL tightened to at most ``ignore_ttl_seconds`` (executor kept), so a
      mis-tap doesn't kill the flow but the destructive op cannot stay
      armed for the original long window. ``record`` carries the tightened
      ``expires_at``.

    Except for ``kind_mismatch`` and ``ignored``, every non-``executed``
    status means the pending action is disarmed — a stale destructive
    action is never left armed. On :class:`InMemoryPendingStore` (and on
    every claiming path) disarmed also means *removed*. The one precise
    exception is ``"expired"`` from :class:`RedisPendingStore`'s
    ``pop_if_intent``, which deliberately reports expiry WITHOUT claiming:
    the key persists inert for up to ~1s until the whole-second server-side
    TTL reaps it (the expiry guard refuses execution on every path in the
    meantime; claiming would risk disarming a live same-scope re-stash).
    """

    status: str
    record: PendingAction
    result: Optional[str] = None

    @property
    def executed(self) -> bool:
        """Whether the side effect actually ran."""
        return self.status == "executed"


# ---------------------------------------------------------------------------
# Confirm parsing
# ---------------------------------------------------------------------------


@beta
class ConfirmParser(Protocol):
    """Pluggable confirm/cancel intent classifier for inbound messages."""

    version: str

    def is_confirm(self, msg: str) -> bool:
        """Whether the message is an unambiguous confirmation."""
        ...

    def is_cancel(self, msg: str) -> bool:
        """Whether the message is an unambiguous cancellation."""
        ...


# Only unambiguous confirmations (Sheriff bug-hunt #10): "ok", "claro",
# "pode", "isso" are common PT-BR acknowledgments in NON-destructive replies
# and must never fire a pending destructive action.
#
# v2 anchoring (PR #73 review BLOCKER): bare confirmation tokens confirm
# only as the WHOLE message. v1 used a prefix match (``re.match`` + ``\b``),
# so "yes, but tell me more first", "yeah right, as if", "confirm what
# exactly?" and "sim, mas antes me explica" all fired a destructive action.
# Allowed trailing content after a bare token is EXACTLY: optional
# punctuation (``.,!``), at most one politeness marker ("please",
# "por favor", "porfa"), and optional punctuation again. Anything else
# (a clause, a question, a condition) means the message is NOT a pure
# confirmation and must not execute.
_CONFIRM_WORD = r"(?:sim|s[ií]|yes|yep|yeah|confirm(?:ado|ar|ed|o|a)?)"
_TRAILING_POLITENESS = r"(?:please|por\s+favor|porfa)"
_CONFIRM_BARE_RE = re.compile(
    rf"^\s*{_CONFIRM_WORD}\s*[.,!]*\s*(?:{_TRAILING_POLITENESS}\s*[.,!]*\s*)?$",
    flags=re.IGNORECASE,
)
# Destructive verb phrases RESTATE the proposed intent ("pode apagar",
# "puedes borrar", "yes, delete it", "sí, borra"), so they may appear with
# other words around them: the first two branches match anywhere in the
# message; the third requires a leading confirm token immediately followed
# by a destructive imperative.
_CONFIRM_VERB_RE = re.compile(
    r"(?:"
    r"\bpode\s+(?:apagar|deletar|excluir|cancelar|remover)\b"
    r"|\bpuedes?\s+(?:borrar|eliminar|cancelar)\b"
    rf"|^\s*{_CONFIRM_WORD}\s*[.,!]*\s+(?:please\s+)?"
    r"(?:delete|remove|cancel|erase|drop"
    r"|apaga|apague|deleta|delete|exclui|remove|cancela"
    r"|borra|b[oó]rralo|elimina|elim[ií]nalo)\b"
    r")",
    flags=re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^(?:ainda\s+n[aã]o|n[aã]o|nao|no|cancel\w*|deixa|nada|nunca|never|olvida\w*)\b",
    flags=re.IGNORECASE,
)


@beta
class RegexConfirmParser:
    """Regex confirm/cancel matcher for PT-BR, English, and Spanish.

    Patterns lifted from Sheriff's proven set and extended with Spanish.
    The bias is conservative: a false negative costs the user a retype; a
    false positive fires a destructive action.

    v2 semantics (PR #73 review): a BARE confirmation token ("sim", "sí",
    "si", "yes", "yep", "yeah", "confirm", "confirmed", "confirmo",
    "confirmar", "confirmado", "confirma") confirms only as the whole
    message — trailing ``.,!`` punctuation and a single politeness marker
    ("please" / "por favor" / "porfa") are the only extra content allowed.
    Destructive verb phrases that restate the intent ("pode apagar",
    "puedes borrar", "yes, delete it", "sí, borra") still confirm with
    surrounding words. A message whose first word is a negation
    ("no, delete it") NEVER confirms — ambiguity resolves to not-confirm.
    """

    version: str = "regex-v2:pt-en-es"

    def is_confirm(self, msg: str) -> bool:
        stripped = (msg or "").strip()
        if not stripped or _CANCEL_RE.match(stripped):
            # A leading negation ("no, delete it", "não, pode apagar")
            # makes the message ambiguous at best — never confirm on it.
            return False
        return bool(_CONFIRM_BARE_RE.match(stripped) or _CONFIRM_VERB_RE.search(stripped))

    def is_cancel(self, msg: str) -> bool:
        stripped = (msg or "").strip()
        return bool(stripped and _CANCEL_RE.match(stripped))


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def _scope_key(user_id: str, channel_id: Optional[str], conversation_id: Optional[str]) -> str:
    """Collision-free scope key. One pending action per scope.

    Confirmation guards for "same user / same channel / same conversation"
    fall out of key identity: stash and pop must present the same scope
    triple, so a confirm from another user or conversation simply misses.
    """
    if not user_id:
        raise ValueError("user_id must not be empty")
    return json.dumps([user_id, channel_id, conversation_id], separators=(",", ":"))


@beta
class PendingActionStore(Protocol):
    """Protocol for deferred-confirmation backends."""

    parser: ConfirmParser

    def stash(
        self,
        user_id: str,
        *,
        kind: str,
        preview: str,
        executor: Executor,
        args: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> PendingAction:
        """Store a pending action with its executor and TTL.

        Raises :class:`PendingActionExistsError` when an unexpired pending
        action already exists for this scope.
        """
        ...

    def get(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        """Return the unexpired pending action for this scope, if any."""
        ...

    def pop_if_confirmed(
        self,
        user_id: str,
        message: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        args_digest: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[ConfirmOutcome]:
        """Atomically claim and execute the pending action iff the message
        confirms AND every guard passes. ``None`` means: no pending for this
        scope, the message is not a confirmation, a twin request won the
        claim, or the claimed record's id differs from ``expected_id`` — in
        all cases the caller must NOT execute anything.

        ``expected_id`` pins the claim to the exact record the caller
        observed (e.g. via ``get``); if a different record was re-stashed
        in between, the claim disarms it and returns ``None`` instead of
        executing an action the user never previewed in this turn.

        A non-``None`` outcome may still be non-executed: ``expired``,
        ``digest_mismatch``, ``no_executor``, or ``failed`` (the executor
        raised; the action is consumed and never retried). Check
        ``outcome.executed`` before treating it as success.
        """
        ...

    def pop_if_intent(
        self,
        user_id: str,
        intent: str,
        *,
        expected_kind: Optional[str] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
        ignore_ttl_seconds: float = DEFAULT_IGNORE_TTL_SECONDS,
    ) -> Optional[ConfirmOutcome]:
        """Claim the pending action given a pre-classified structured intent.

        Issue #82: chat-channel button webhooks (Twilio quick replies,
        Telegram inline keyboards) deliver the user's decision as a
        STRUCTURED payload, not free text — the text parser is bypassed
        entirely. ``intent`` is normalized (strip + lowercase) and must be
        ``"confirm"``, ``"cancel"``, or ``"ignore"``; any other value is
        treated as ``"ignore"`` with a logged warning (a malformed or
        future payload must never fire or drop a pending — but never
        silently either).

        ``expected_kind``/``expected_id`` pin the claim to the action the
        button was minted for. On mismatch the pending is PRESERVED and the
        outcome is ``"kind_mismatch"`` — see :class:`ConfirmOutcome` for why
        this deliberately differs from the disarm-on-``digest_mismatch``
        behavior of :meth:`pop_if_confirmed`.

        Returns ``None`` when there is no pending action for this scope (or
        a twin webhook won the claim). Otherwise a :class:`ConfirmOutcome`
        with status ``executed | expired | no_executor | failed | cancelled
        | kind_mismatch | ignored``.
        """
        ...

    def tighten_ttl(
        self,
        user_id: str,
        seconds: float,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        """Shorten the pending action's ``expires_at`` to now + ``seconds``.

        Never lengthens: if the action already expires sooner, it is left
        untouched (and returned as-is). The executor/registry entry is KEPT
        — the action stays confirmable inside the tightened window.

        ``expected_id`` restricts the tighten to the exact record the caller
        observed; on mismatch nothing changes and ``None`` is returned.

        Returns the (possibly updated) record, or ``None`` when there is no
        unexpired pending action for this scope.
        """
        ...

    def drop(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        """Cancel and remove the pending action for this scope, if any."""
        ...


def _build_record(
    user_id: str,
    kind: str,
    preview: str,
    args: Optional[Mapping[str, Any]],
    ttl_seconds: Optional[float],
    default_ttl: float,
    channel_id: Optional[str],
    conversation_id: Optional[str],
    parser_version: str,
) -> PendingAction:
    now = time.time()
    ttl = default_ttl if ttl_seconds is None else ttl_seconds
    return PendingAction(
        pending_action_id=uuid.uuid4().hex,
        user_id=user_id,
        kind=kind,
        preview=preview,
        args_digest=compute_args_digest(args),
        requested_at=now,
        expires_at=now + ttl,
        channel_id=channel_id,
        conversation_id=conversation_id,
        parser_version=parser_version,
        status=_PENDING,
        args=dict(args) if args is not None else None,
    )


def _target_mismatch(
    record: PendingAction,
    expected_kind: Optional[str],
    expected_id: Optional[str],
) -> bool:
    """Whether a button's pinned target differs from the pending action.

    True means the tap was minted for a DIFFERENT prompt (stale replay,
    out-of-order delivery) — the caller must preserve the pending and report
    ``kind_mismatch`` instead of claiming it.
    """
    if expected_kind is not None and expected_kind != record.kind:
        logger.warning(
            "pending: button kind %r does not match pending %s (kind=%s) — "
            "preserving the pending action",
            expected_kind,
            record.pending_action_id,
            record.kind,
        )
        return True
    if expected_id is not None and expected_id != record.pending_action_id:
        logger.warning(
            "pending: button target id %s does not match pending %s (kind=%s) — "
            "preserving the pending action",
            expected_id,
            record.pending_action_id,
            record.kind,
        )
        return True
    return False


_VALID_INTENTS = ("confirm", "cancel", "ignore")


def _normalize_intent(intent: str) -> str:
    """Normalize a structured button intent; coerce unknowns to ``ignore``.

    Case and surrounding whitespace are forgiven ("Confirm\\n" from a sloppy
    webhook payload is still a confirm). Anything else is fail-safe coerced
    to ``ignore`` — a malformed or future payload must never fire or drop a
    pending — but LOUDLY (review finding 2): silent coercion hid integration
    bugs where a misspelled payload key downgraded every confirm to ignore.
    """
    normalized = (intent or "").strip().lower()
    if normalized in _VALID_INTENTS:
        return normalized
    logger.warning("pending: unrecognized intent %r treated as ignore", intent)
    return "ignore"


def _guarded_execute(
    record: PendingAction,
    executor: Optional[Executor],
    args_digest: Optional[str],
) -> ConfirmOutcome:
    """Run the post-claim guards and (only then) the executor.

    The caller has already atomically claimed the record — whatever the
    outcome here, the action is no longer armed.
    """
    if record.is_expired():
        return ConfirmOutcome(status="expired", record=replace(record, status=_EXPIRED))
    if args_digest is not None and args_digest != record.args_digest:
        logger.warning(
            "pending: digest mismatch for %s (kind=%s) — action changed after "
            "preview, refusing to execute",
            record.pending_action_id,
            record.kind,
        )
        return ConfirmOutcome(status="digest_mismatch", record=replace(record, status=_CANCELLED))
    if executor is None:
        logger.warning(
            "pending: no executor available for %s (kind=%s) — register an "
            "executor factory for this kind to confirm across processes",
            record.pending_action_id,
            record.kind,
        )
        return ConfirmOutcome(status="no_executor", record=replace(record, status=_CANCELLED))
    confirmed = replace(record, status=_CONFIRMED)
    try:
        result = executor()
    except Exception as exc:
        # PR #73 review finding 5: an executor raise must not propagate raw
        # to the webhook layer with no audit trail (a webhook retry would
        # then fall through to the LLM as a fresh message). The record is
        # consumed — one-shot semantics, no re-arm, no automatic retry —
        # and the outcome records only the exception TYPE name; the full
        # detail goes to logging, never into user-facing/persisted text.
        logger.warning(
            "pending: executor for %s (kind=%s) raised %s: %s",
            record.pending_action_id,
            record.kind,
            type(exc).__name__,
            exc,
        )
        return ConfirmOutcome(
            status="failed",
            record=replace(confirmed, status=_CONSUMED, outcome=f"error: {type(exc).__name__}"),
        )
    return ConfirmOutcome(
        status="executed",
        record=replace(confirmed, status=_CONSUMED, outcome=result),
        result=result,
    )


@beta
class InMemoryPendingStore:
    """Thread-safe, TTL-bounded in-process pending-action store.

    The proven Sheriff shape: a lock-guarded ``OrderedDict`` capped at
    ``max_entries`` (LRU eviction) so unconfirmed actions from many users
    cannot grow the map unboundedly. A process restart drops pending
    actions — the user re-asks, a mild inconvenience that is the right
    trade for destructive actions.

    Atomicity: the claim (remove-from-map + status flip) happens inside the
    lock, so duplicate webhook delivery executes ONCE — the twin caller sees
    ``None``. The executor itself runs OUTSIDE the lock so a slow side
    effect cannot block other users' confirmations.
    """

    def __init__(
        self,
        parser: Optional[ConfirmParser] = None,
        default_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = 5000,
    ) -> None:
        self.parser: ConfirmParser = parser if parser is not None else RegexConfirmParser()
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._pending: "OrderedDict[str, Tuple[PendingAction, Executor]]" = OrderedDict()

    def stash(
        self,
        user_id: str,
        *,
        kind: str,
        preview: str,
        executor: Executor,
        args: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> PendingAction:
        key = _scope_key(user_id, channel_id, conversation_id)
        record = _build_record(
            user_id,
            kind,
            preview,
            args,
            ttl_seconds,
            self._default_ttl,
            channel_id,
            conversation_id,
            self.parser.version,
        )
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None and not existing[0].is_expired():
                raise PendingActionExistsError(
                    f"scope {key} already has an unexpired pending "
                    f"{existing[0].kind!r}; refusing to replace with {kind!r}"
                )
            self._pending[key] = (record, executor)
            self._pending.move_to_end(key)
            while len(self._pending) > self._max_entries:
                evicted_key, _ = self._pending.popitem(last=False)
                logger.warning(
                    "pending: LRU-evicted pending for scope %s (cap=%d reached)",
                    evicted_key,
                    self._max_entries,
                )
        return record

    def get(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        key = _scope_key(user_id, channel_id, conversation_id)
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                return None
            if entry[0].is_expired():
                self._pending.pop(key, None)
                return None
            return entry[0]

    def pop_if_confirmed(
        self,
        user_id: str,
        message: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        args_digest: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[ConfirmOutcome]:
        if not self.parser.is_confirm(message):
            return None
        key = _scope_key(user_id, channel_id, conversation_id)
        with self._lock:
            entry = self._pending.pop(key, None)
        if entry is None:
            # No pending for this scope, or a twin webhook won the claim.
            return None
        record, executor = entry
        if expected_id is not None and record.pending_action_id != expected_id:
            # PR #73 review finding 6: the record was re-stashed between the
            # caller's get() and this claim. The claimed action was never
            # previewed in this turn — disarm it (already popped) and treat
            # as no-pending. Never execute under an identity mismatch.
            logger.warning(
                "pending: claimed record %s does not match expected id %s "
                "(kind=%s) — disarming without execution",
                record.pending_action_id,
                expected_id,
                record.kind,
            )
            return None
        return _guarded_execute(record, executor, args_digest)

    def pop_if_intent(
        self,
        user_id: str,
        intent: str,
        *,
        expected_kind: Optional[str] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
        ignore_ttl_seconds: float = DEFAULT_IGNORE_TTL_SECONDS,
    ) -> Optional[ConfirmOutcome]:
        if ignore_ttl_seconds < 0:
            raise ValueError("ignore_ttl_seconds must be >= 0")
        effective = _normalize_intent(intent)
        key = _scope_key(user_id, channel_id, conversation_id)
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                return None
            record, executor = entry
            if record.is_expired():
                self._pending.pop(key, None)
                return ConfirmOutcome(status="expired", record=replace(record, status=_EXPIRED))
            if _target_mismatch(record, expected_kind, expected_id):
                # PRESERVE: the button belonged to a different prompt (see
                # ConfirmOutcome docstring). The record stays armed as-is.
                return ConfirmOutcome(status="kind_mismatch", record=record)
            if effective == "ignore":
                now = time.time()
                new_expires = min(record.expires_at, now + ignore_ttl_seconds)
                if new_expires < record.expires_at:
                    record = replace(record, expires_at=new_expires)
                    self._pending[key] = (record, executor)
                return ConfirmOutcome(status="ignored", record=record)
            self._pending.pop(key, None)
            if effective == "cancel":
                return ConfirmOutcome(status="cancelled", record=replace(record, status=_CANCELLED))
        # "confirm": the claim happened inside the lock; the executor runs
        # outside it (same rationale as pop_if_confirmed).
        return _guarded_execute(record, executor, None)

    def tighten_ttl(
        self,
        user_id: str,
        seconds: float,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        key = _scope_key(user_id, channel_id, conversation_id)
        now = time.time()
        with self._lock:
            entry = self._pending.get(key)
            if entry is None:
                return None
            record, executor = entry
            if record.is_expired(now):
                self._pending.pop(key, None)
                return None
            if expected_id is not None and record.pending_action_id != expected_id:
                return None
            new_expires = min(record.expires_at, now + seconds)
            if new_expires >= record.expires_at:
                return record
            updated = replace(record, expires_at=new_expires)
            self._pending[key] = (updated, executor)
            return updated

    def drop(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        key = _scope_key(user_id, channel_id, conversation_id)
        with self._lock:
            entry = self._pending.pop(key, None)
        if entry is None:
            return None
        return replace(entry[0], status=_CANCELLED)


# Id-pinned compare-and-set for tighten_ttl (review finding 1). SET XX only
# proves that *a* key exists — not that it is still the observed record. In
# the GET -> SET window a twin webhook can claim record A and the same scope
# can stash a NEW record B; a plain SET XX would then overwrite live B with
# re-armed A (destroying B and, via a registered factory, letting A execute
# twice). The script rewrites the key ONLY while it still holds the observed
# pending_action_id, atomically server-side.
_TIGHTEN_TTL_LUA = (
    "local v = redis.call('GET', KEYS[1]) "
    "if v and cjson.decode(v).pending_action_id == ARGV[1] then "
    "redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3]) return 1 end "
    "return 0"
)


@beta
class RedisPendingStore:
    """Redis-backed pending-action store for multi-instance deployments.

    Follows the ``RedisSessionStore`` pattern: lazy ``import redis``, prefix
    namespace, server-side TTL via ``SETEX``.

    Closures vs Redis: only the :class:`PendingAction` record (a JSON
    document) is persisted. Executor closures live in a process-local
    registry keyed by ``pending_action_id`` — pickling callables into Redis
    would be both fragile and a deserialization attack surface. When the
    confirming webhook lands on the SAME process, the closure is found and
    runs. When it lands on a DIFFERENT process (or after a restart), register
    an executor factory per ``kind`` with :meth:`register_executor_factory`;
    it rebuilds the executor from the persisted record (``record.args``).
    With neither closure nor factory the claim resolves to ``no_executor``
    and nothing executes.

    Atomic consume: the claim is a single ``GETDEL`` (Redis >= 6.2) — read
    and remove in one server-side command, so concurrent duplicate webhooks
    can never both observe the record. Guards (TTL, digest, executor
    resolution) run AFTER the claim; any guard failure means the record has
    already been removed, so a stale destructive action is never left armed.
    ``GETDEL`` was chosen over ``WATCH``/``MULTI`` (retry loops, more round
    trips) because the desired semantics — exactly-once claim — map onto one
    command. The only operation with no single-command equivalent is the
    id-pinned TTL rewrite in :meth:`tighten_ttl`, which uses a four-line Lua
    ``EVAL`` (compare ``pending_action_id``, then ``SET EX``) so a record
    claimed-or-replaced mid-rewrite is neither resurrected nor overwritten.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "selectools:pending:",
        parser: Optional[ConfirmParser] = None,
        default_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_executor_entries: int = 5000,
    ) -> None:
        try:
            import redis as redis_lib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RedisPendingStore requires the 'redis' package. "
                "Install it with: pip install selectools[cache]"
            ) from exc

        self._client: Any = redis_lib.from_url(url, decode_responses=True)
        self._prefix = prefix
        self.parser: ConfirmParser = parser if parser is not None else RegexConfirmParser()
        self._default_ttl = default_ttl_seconds
        self._max_executor_entries = max_executor_entries
        self._lock = threading.RLock()
        # Process-local. See class docstring — closures are never persisted.
        # PR #73 review finding 4: entries carry their expires_at so the
        # no-reply path (TTL expiry with no confirming webhook) cannot leak
        # closures forever — expired entries are purged on each stash, and
        # an LRU cap (mirroring InMemoryPendingStore's max_entries) bounds
        # the registry even before expiry. An evicted closure degrades to
        # the documented ``no_executor``/factory path, never to a leak.
        self._executors: "OrderedDict[str, Tuple[Executor, float]]" = OrderedDict()
        self._factories: Dict[str, ExecutorFactory] = {}

    def _purge_expired_executors_locked(self) -> None:
        """Drop registry entries whose record TTL has passed. Caller holds the lock."""
        now = time.time()
        stale = [pid for pid, (_, expires_at) in self._executors.items() if now > expires_at]
        for pid in stale:
            del self._executors[pid]

    def _key(self, user_id: str, channel_id: Optional[str], conversation_id: Optional[str]) -> str:
        return f"{self._prefix}{_scope_key(user_id, channel_id, conversation_id)}"

    def register_executor_factory(self, kind: str, factory: ExecutorFactory) -> None:
        """Register a factory that rebuilds the executor for ``kind``.

        Required for cross-process confirmation: the factory receives the
        persisted :class:`PendingAction` (including ``args``) and returns a
        zero-arg executor. Register factories at process startup, before
        webhooks are served.
        """
        with self._lock:
            self._factories[kind] = factory

    def stash(
        self,
        user_id: str,
        *,
        kind: str,
        preview: str,
        executor: Executor,
        args: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> PendingAction:
        key = self._key(user_id, channel_id, conversation_id)
        record = _build_record(
            user_id,
            kind,
            preview,
            args,
            ttl_seconds,
            self._default_ttl,
            channel_id,
            conversation_id,
            self.parser.version,
        )
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        # Server-side TTL is the primary expiry; expires_at in the record is
        # the precise sub-second guard checked at claim time.
        ttl = max(1, math.ceil(record.expires_at - record.requested_at))
        # PR #73 review finding 3: the write must be atomic SET NX EX, not
        # GET -> check -> SETEX. Two concurrent duplicate initiating
        # webhooks in the old check-then-act window could both pass the
        # check and the loser's SETEX overwrote the winner — the user then
        # saw preview A while executor B was armed. With NX exactly one
        # writer wins; the loser raises PendingActionExistsError.
        for _ in range(2):
            if self._client.set(key, payload, nx=True, ex=ttl):
                with self._lock:
                    self._purge_expired_executors_locked()
                    self._executors[record.pending_action_id] = (executor, record.expires_at)
                    while len(self._executors) > self._max_executor_entries:
                        evicted_id, _ = self._executors.popitem(last=False)
                        logger.warning(
                            "pending: LRU-evicted executor closure %s (cap=%d reached); "
                            "confirmation will need a registered factory",
                            evicted_id,
                            self._max_executor_entries,
                        )
                return record
            existing_raw = self._client.get(key)
            if existing_raw is not None:
                existing = PendingAction.from_dict(json.loads(existing_raw))
                if not existing.is_expired():
                    raise PendingActionExistsError(
                        f"scope for user {user_id!r} already has an unexpired pending "
                        f"{existing.kind!r}; refusing to replace with {kind!r}"
                    )
                # Sub-second expired (record guard) but the whole-second
                # server TTL has not fired yet: clear it and retry the
                # atomic SET NX once.
                self._client.delete(key)
            # else: the key vanished between SET NX and GET (claimed or
            # server-expired) — retry the atomic write once.
        raise PendingActionExistsError(
            f"scope for user {user_id!r} is contended; could not stash {kind!r} atomically"
        )

    def get(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        raw = self._client.get(self._key(user_id, channel_id, conversation_id))
        if raw is None:
            return None
        record = PendingAction.from_dict(json.loads(raw))
        if record.is_expired():
            return None
        return record

    def pop_if_confirmed(
        self,
        user_id: str,
        message: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        args_digest: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[ConfirmOutcome]:
        if not self.parser.is_confirm(message):
            return None
        # Atomic claim: GETDEL reads and removes in one server-side command,
        # so duplicate webhook delivery executes ONCE (the twin gets None).
        raw = self._client.getdel(self._key(user_id, channel_id, conversation_id))
        if raw is None:
            return None
        record = PendingAction.from_dict(json.loads(raw))
        with self._lock:
            entry = self._executors.pop(record.pending_action_id, None)
            factory = self._factories.get(record.kind)
        if expected_id is not None and record.pending_action_id != expected_id:
            # PR #73 review finding 6: see InMemoryPendingStore — never
            # execute under an identity mismatch; the claimed record (and
            # its closure) are disarmed and the caller sees no-pending.
            logger.warning(
                "pending: claimed record %s does not match expected id %s "
                "(kind=%s) — disarming without execution",
                record.pending_action_id,
                expected_id,
                record.kind,
            )
            return None
        executor: Optional[Executor] = entry[0] if entry is not None else None
        if executor is None and factory is not None:
            executor = factory(record)
        return _guarded_execute(record, executor, args_digest)

    def pop_if_intent(
        self,
        user_id: str,
        intent: str,
        *,
        expected_kind: Optional[str] = None,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
        ignore_ttl_seconds: float = DEFAULT_IGNORE_TTL_SECONDS,
    ) -> Optional[ConfirmOutcome]:
        if ignore_ttl_seconds < 0:
            raise ValueError("ignore_ttl_seconds must be >= 0")
        effective = _normalize_intent(intent)
        key = self._key(user_id, channel_id, conversation_id)
        # Preserve-guards (kind/id pins, ignore) must run BEFORE the claim,
        # so the flow is observe-then-act: GET, validate, then GETDEL only
        # for claiming intents. The post-claim id re-check below closes the
        # observe/claim race (review finding 6 semantics).
        raw = self._client.get(key)
        if raw is None:
            return None
        record = PendingAction.from_dict(json.loads(raw))
        if record.is_expired():
            # Report expired WITHOUT claiming (review NOTE 3a): the record
            # guard already refuses execution and the server-side TTL reaps
            # the key on its own — a GETDEL here could race a same-scope
            # re-stash and disarm the user's LIVE record instead of the
            # expired one. The stash path tolerates sub-second-expired keys.
            return ConfirmOutcome(status="expired", record=replace(record, status=_EXPIRED))
        if _target_mismatch(record, expected_kind, expected_id):
            # PRESERVE: nothing was claimed; the record (and its closure)
            # stay armed. See ConfirmOutcome for the kind-vs-digest rationale.
            return ConfirmOutcome(status="kind_mismatch", record=record)
        if effective == "ignore":
            updated = self.tighten_ttl(
                user_id,
                ignore_ttl_seconds,
                channel_id=channel_id,
                conversation_id=conversation_id,
                expected_id=record.pending_action_id,
            )
            if updated is None:
                # Claimed (or replaced) between the GET and the tighten —
                # a twin webhook won; report no-pending.
                return None
            return ConfirmOutcome(status="ignored", record=updated)
        claimed = self._claim_observed(key, record.pending_action_id)
        if claimed is None:
            return None
        # From here on, use the CLAIMED record, not the observed snapshot
        # (review NOTE 3b): same id, but a concurrent tighten_ttl may have
        # rewritten expires_at between the GET and the claim — guards and
        # outcomes must reflect what was actually removed from Redis.
        with self._lock:
            entry = self._executors.pop(claimed.pending_action_id, None)
            factory = self._factories.get(claimed.kind)
        if effective == "cancel":
            return ConfirmOutcome(status="cancelled", record=replace(claimed, status=_CANCELLED))
        executor: Optional[Executor] = entry[0] if entry is not None else None
        if executor is None and factory is not None:
            executor = factory(claimed)
        return _guarded_execute(claimed, executor, None)

    def _claim_observed(self, key: str, observed_id: str) -> Optional[PendingAction]:
        """Atomically claim ``key`` iff it still holds the observed record.

        ``GETDEL`` then an id re-check: when a different record was stashed
        between the caller's GET and this claim, the claimed action was
        never validated against the caller's guards — it is disarmed (kept
        popped, closure purged) and ``None`` is returned, mirroring the
        ``expected_id`` semantics of ``pop_if_confirmed`` (finding 6). Never
        executes or restores anything.
        """
        raw = self._client.getdel(key)
        if raw is None:
            return None
        claimed = PendingAction.from_dict(json.loads(raw))
        if claimed.pending_action_id != observed_id:
            with self._lock:
                self._executors.pop(claimed.pending_action_id, None)
            logger.warning(
                "pending: claimed record %s does not match observed id %s "
                "(kind=%s) — disarming without execution",
                claimed.pending_action_id,
                observed_id,
                claimed.kind,
            )
            return None
        return claimed

    def tighten_ttl(
        self,
        user_id: str,
        seconds: float,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        expected_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        key = self._key(user_id, channel_id, conversation_id)
        raw = self._client.get(key)
        if raw is None:
            return None
        record = PendingAction.from_dict(json.loads(raw))
        if record.is_expired():
            return None
        if expected_id is not None and record.pending_action_id != expected_id:
            return None
        now = time.time()
        new_expires = min(record.expires_at, now + seconds)
        if new_expires >= record.expires_at:
            return record
        updated = replace(record, expires_at=new_expires)
        payload = json.dumps(updated.to_dict(), ensure_ascii=False)
        ttl = max(1, math.ceil(new_expires - now))
        # Id-pinned compare-and-set (review finding 1): the Lua script
        # rewrites the key ONLY while it still holds the observed record's
        # pending_action_id. A record claimed between the GET above and this
        # write is never resurrected, and a same-scope re-stash inside that
        # window is never overwritten — the rewrite simply misses and the
        # caller sees ``None`` (a twin won).
        if not self._client.eval(
            _TIGHTEN_TTL_LUA, 1, key, record.pending_action_id, payload, str(ttl)
        ):
            return None
        with self._lock:
            entry = self._executors.get(record.pending_action_id)
            if entry is not None:
                # Keep the registry purge horizon in sync with the new TTL.
                self._executors[record.pending_action_id] = (entry[0], new_expires)
        return updated

    def drop(
        self,
        user_id: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[PendingAction]:
        raw = self._client.getdel(self._key(user_id, channel_id, conversation_id))
        if raw is None:
            return None
        record = PendingAction.from_dict(json.loads(raw))
        with self._lock:
            self._executors.pop(record.pending_action_id, None)
        return replace(record, status=_CANCELLED)


# ---------------------------------------------------------------------------
# PendingConfirmation result type
# ---------------------------------------------------------------------------


@beta
@dataclass(frozen=True)
class PendingConfirmation(ToolResult):
    """Typed tool return signalling "previewed, awaiting user confirmation".

    Destructive tools return this INSTEAD of executing: the LLM sees the
    preview and relays ``user_prompt`` to the user; the actual side effect
    was stashed via :func:`stash_pending` and runs only when a later turn
    confirms it.

    Attributes:
        action: The action kind that was stashed (e.g. ``"delete_invoice"``).
        preview: Human-readable description of what will happen.
        user_prompt: What to ask the user (e.g. "Reply 'yes' to confirm.").
    """

    kind: ClassVar[str] = "pending_confirmation"

    action: str
    preview: str
    user_prompt: str


# ---------------------------------------------------------------------------
# Channel wiring: contextvar + ChannelAgent
# ---------------------------------------------------------------------------


@dataclass
class _PendingRunContext:
    """Per-run channel scope installed by ChannelAgent around agent dispatch."""

    store: PendingActionStore
    user_id: str
    channel_id: Optional[str] = None
    conversation_id: Optional[str] = None
    stashed: List[PendingAction] = field(default_factory=list)


# Mirrors the results.py artifact-collector pattern: a ContextVar keeps
# concurrent runs isolated, and thread-pool tool execution sites copy the
# caller's context so the scope is visible inside worker threads
# (pitfall #28 / BUG-32).
_pending_run_context: contextvars.ContextVar[Optional[_PendingRunContext]] = contextvars.ContextVar(
    "selectools_pending_run_context", default=None
)


@beta
def stash_pending(
    *,
    kind: str,
    preview: str,
    executor: Executor,
    args: Optional[Mapping[str, Any]] = None,
    ttl_seconds: Optional[float] = None,
) -> Optional[PendingAction]:
    """Stash a deferred destructive action from inside a tool.

    Call this from a tool function while a :class:`ChannelAgent` run is
    active — the channel scope (store, user, channel, conversation) is
    injected via a ContextVar, the same mechanism ``emit_artifact`` uses.

    Outside a channel run (direct tool calls in tests/scripts) this is a
    no-op returning ``None``, so tools remain directly callable.

    Raises:
        PendingActionExistsError: an unexpired pending action already exists
            for this user/scope. Catch it and return a "finish the previous
            confirmation first" message instead of silently re-arming.

    Returns:
        The stashed :class:`PendingAction`, or ``None`` outside a channel run.
    """
    ctx = _pending_run_context.get()
    if ctx is None:
        return None
    record = ctx.store.stash(
        ctx.user_id,
        kind=kind,
        preview=preview,
        executor=executor,
        args=args,
        ttl_seconds=ttl_seconds,
        channel_id=ctx.channel_id,
        conversation_id=ctx.conversation_id,
    )
    ctx.stashed.append(record)
    return record


def _ack(content: str) -> AgentResult:
    """Build a channel-layer AgentResult that bypassed the LLM."""
    return AgentResult(message=Message(role=Role.ASSISTANT, content=content), iterations=0)


@beta
class ChannelAgent:
    """Thin wrapper routing webhook turns through the confirmation flow.

    Per inbound message:

    1. If a pending action exists and the message CANCELS it: drop + ack.
    2. If a pending action exists and the message CONFIRMS it: atomically
       claim + execute, returning the executor outcome (LLM bypassed).
    3. Otherwise: any existing pending is dropped (the user moved on — a
       destructive action must never stay armed behind an unrelated reply)
       and the message dispatches to the wrapped :class:`Agent` normally,
       with the channel scope installed so tools can :func:`stash_pending`.

    Args:
        agent: The wrapped ``selectools.Agent``.
        store: Pending-action backend. Defaults to a fresh
            :class:`InMemoryPendingStore`.
        parser: Confirm parser. When given, it is assigned onto the store
            as well — the store re-parses the message inside
            ``pop_if_confirmed``, so the channel and the store MUST share
            one parser (PR #73 review finding 2: a divergent pair acked
            "No pending action to confirm", swallowed the message, and
            left the action armed). Defaults to the store's parser.
    """

    def __init__(
        self,
        agent: "Agent",
        store: Optional[PendingActionStore] = None,
        parser: Optional[ConfirmParser] = None,
    ) -> None:
        self.agent = agent
        self.store: PendingActionStore = store if store is not None else InMemoryPendingStore()
        if parser is not None:
            # Single source of decision: the store's pop_if_confirmed is the
            # final gate, so the channel-level parser must BE the store's.
            self.store.parser = parser
        self.parser: ConfirmParser = self.store.parser

    def ask_channel(
        self,
        user_id: str,
        message: str,
        *,
        channel_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        args_digest: Optional[str] = None,
    ) -> AgentResult:
        """Handle one inbound channel turn (see class docstring).

        Args:
            user_id: Stable id of the sender (phone, chat id, ...).
            message: The inbound message text.
            channel_id: Optional channel scope; pass the SAME value used
                when the action was stashed.
            conversation_id: Optional conversation/thread scope.
            args_digest: Optional recomputed digest of the side effect that
                would execute NOW; on mismatch with the previewed digest the
                action is refused and a fresh confirmation is requested.

        Returns:
            AgentResult — either a channel-layer ack/outcome (``iterations``
            is 0 and the LLM was bypassed) or the wrapped agent's result.
        """
        scope = {"channel_id": channel_id, "conversation_id": conversation_id}
        pending = self.store.get(user_id, **scope)
        if pending is not None:
            if self.parser.is_cancel(message):
                dropped = self.store.drop(user_id, **scope)
                preview = dropped.preview if dropped is not None else pending.preview
                return _ack(f"Cancelled: {preview}")
            if self.parser.is_confirm(message):
                outcome = self.store.pop_if_confirmed(
                    user_id,
                    message,
                    args_digest=args_digest,
                    # Pin the claim to the exact record observed above: if a
                    # different action was re-stashed in between, the store
                    # disarms it and reports no-pending instead of executing
                    # something this turn never previewed (review finding 6).
                    expected_id=pending.pending_action_id,
                    **scope,
                )
                if outcome is None:
                    # Twin webhook won the claim between get() and the pop,
                    # or the claimed record was not the one observed.
                    return _ack("No pending action to confirm.")
                if outcome.executed:
                    return _ack(outcome.result or "")
                if outcome.status == "expired":
                    return _ack(
                        f"That confirmation expired: {outcome.record.preview}. Please start over."
                    )
                if outcome.status == "failed":
                    # The executor raised (review finding 5): the action is
                    # consumed — one-shot, no automatic retry — and the raw
                    # exception never reaches the webhook layer.
                    return _ack(
                        f"Confirmed, but the action failed while executing: "
                        f"{outcome.record.preview}. It was not retried — please "
                        "check the result and request it again if needed."
                    )
                # digest_mismatch | no_executor: never execute a stale or
                # unresolvable action — ask for a fresh confirmation cycle.
                return _ack(
                    f"Could not safely confirm: {outcome.record.preview}. "
                    "The action may have changed — please request it again."
                )
            # Not a confirm, not a cancel: the user moved on. Drop the
            # pending so a casual later "yes" can never fire it (Sheriff
            # behavior), then fall through to the normal agent path.
            self.store.drop(user_id, **scope)

        token = _pending_run_context.set(
            _PendingRunContext(
                store=self.store,
                user_id=user_id,
                channel_id=channel_id,
                conversation_id=conversation_id,
            )
        )
        try:
            return self.agent.run(message)
        finally:
            _pending_run_context.reset(token)
