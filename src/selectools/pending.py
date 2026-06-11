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
from .stability import beta

if TYPE_CHECKING:
    from .agent.core import Agent
from .types import AgentResult, Message, Role

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TTL_SECONDS",
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
    """Result of a confirmed ``pop_if_confirmed`` claim.

    ``status`` values:

    - ``"executed"``: all guards passed; ``result`` holds the executor output.
    - ``"expired"``: the "yes" arrived after the TTL. Nothing executed.
    - ``"digest_mismatch"``: the action changed after the preview. Nothing
      executed; a fresh confirmation cycle is required.
    - ``"no_executor"``: no closure and no factory could run the action
      (e.g. the confirming webhook landed on a fresh process). Nothing
      executed; a fresh confirmation cycle is required.

    In every non-``executed`` case the pending action has been removed —
    a stale destructive action is never left armed.
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
# and must never fire a pending destructive action. Spanish bare "si" is the
# conditional "if" mid-sentence, so it only confirms as the entire message;
# the accented "sí" is unambiguous anywhere at the start.
_CONFIRM_RE = re.compile(
    r"^(?:"
    r"(?:sim|sí|confirmo|confirma(?:r|do)?)\b"
    r"|si\s*[.!]?\s*$"
    r"|pode\s+(?:apagar|deletar|cancelar|remover)\b"
    r"|puedes?\s+(?:borrar|eliminar|cancelar)\b"
    r"|(?:yes|yep|yeah|confirm(?:ed)?)\b"
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
    """

    version: str = "regex-v1:pt-en-es"

    def is_confirm(self, msg: str) -> bool:
        stripped = (msg or "").strip()
        return bool(stripped and _CONFIRM_RE.match(stripped))

    def is_cancel(self, msg: str) -> bool:
        stripped = (msg or "").strip()
        if not stripped or _CONFIRM_RE.match(stripped):
            return False
        return bool(_CANCEL_RE.match(stripped))


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
    ) -> Optional[ConfirmOutcome]:
        """Atomically claim and execute the pending action iff the message
        confirms AND every guard passes. ``None`` means: no pending for this
        scope, the message is not a confirmation, or a twin request won the
        claim — in all three cases the caller must NOT execute anything.
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
    result = executor()
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
        return _guarded_execute(record, executor, args_digest)

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
    trips) and Lua (heavier dependency surface for fakes/tests) because the
    desired semantics — exactly-once claim — map onto one command.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "selectools:pending:",
        parser: Optional[ConfirmParser] = None,
        default_ttl_seconds: float = DEFAULT_TTL_SECONDS,
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
        self._lock = threading.RLock()
        # Process-local. See class docstring — closures are never persisted.
        self._executors: Dict[str, Executor] = {}
        self._factories: Dict[str, ExecutorFactory] = {}

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
        existing_raw = self._client.get(key)
        if existing_raw is not None:
            existing = PendingAction.from_dict(json.loads(existing_raw))
            if not existing.is_expired():
                raise PendingActionExistsError(
                    f"scope for user {user_id!r} already has an unexpired pending "
                    f"{existing.kind!r}; refusing to replace with {kind!r}"
                )
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
        # Server-side TTL is the primary expiry; expires_at in the record is
        # the precise sub-second guard checked at claim time.
        ttl = max(1, math.ceil(record.expires_at - record.requested_at))
        self._client.setex(key, ttl, json.dumps(record.to_dict(), ensure_ascii=False))
        with self._lock:
            self._executors[record.pending_action_id] = executor
        return record

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
            executor: Optional[Executor] = self._executors.pop(record.pending_action_id, None)
            factory = self._factories.get(record.kind)
        if executor is None and factory is not None:
            executor = factory(record)
        return _guarded_execute(record, executor, args_digest)

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
        parser: Confirm parser. Defaults to the store's parser.
    """

    def __init__(
        self,
        agent: "Agent",
        store: Optional[PendingActionStore] = None,
        parser: Optional[ConfirmParser] = None,
    ) -> None:
        self.agent = agent
        self.store: PendingActionStore = store if store is not None else InMemoryPendingStore()
        self.parser: ConfirmParser = parser if parser is not None else self.store.parser

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
                    user_id, message, args_digest=args_digest, **scope
                )
                if outcome is None:
                    # Twin webhook won the claim between get() and the pop.
                    return _ack("No pending action to confirm.")
                if outcome.executed:
                    return _ack(outcome.result or "")
                if outcome.status == "expired":
                    return _ack(
                        f"That confirmation expired: {outcome.record.preview}. Please start over."
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
