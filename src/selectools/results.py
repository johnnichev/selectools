"""
Typed tool results and the artifact side-channel (issue #59).

Two small conventions that every tool-using agent otherwise re-invents:

1. :class:`ToolResult` — a frozen dataclass base for typed, JSON-serializable
   tool returns the next LLM turn can reason over. Subclasses declare a
   ``kind`` discriminator as a ``ClassVar`` (class-level, not caller-supplied
   instance state).

   .. warning::
      ``ClassVar`` annotations are excluded from ``dataclasses.fields()`` by
      design, so ``dataclasses.asdict()`` silently drops ``kind``. The fix
      lives in the serializer, not the model: ``Tool._serialize_result``
      re-injects ``kind`` explicitly for ``ToolResult`` instances.

2. :class:`Artifact` + :func:`emit_artifact` — a side-channel for tools that
   produce files (charts, PDFs, audio, exports). Stuffing a URL into the
   reply string is LLM-hostile; instead tools call ``emit_artifact(...)``
   during execution and the agent drains the per-run collector into
   ``AgentResult.artifacts``.

   The artifact shape is deliberately richer than a bare URL: URLs rot and
   signed links expire, so ``sha256`` and ``size`` let consumers identify an
   artifact without storing its body.

Example::

    from selectools import Agent, NotFound, emit_artifact, tool

    @tool()
    def render_chart(title: str) -> str:
        url = make_chart(title)
        emit_artifact(url, mime_type="image/png", role="primary")
        return "chart rendered"

    @tool()
    def find_customer(query: str):
        rows = db.search(query)
        if not rows:
            return NotFound(entity="customer", query=query)
        ...
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

from .stability import beta

__all__ = [
    "ToolResult",
    "Ambiguous",
    "NotFound",
    "Artifact",
    "emit_artifact",
]


@beta
@dataclass(frozen=True)
class ToolResult:
    """Base class for typed tool returns.

    Subclasses set ``kind`` as a ``ClassVar`` discriminator::

        @dataclass(frozen=True)
        class RateLimited(ToolResult):
            kind: ClassVar[str] = "rate_limited"
            retry_after: int = 0

    The base is intentionally small: ``kind`` is the only convention.
    Subclasses that need provenance are encouraged to add their own
    ``source`` or evidence-reference field — typed results describe what a
    tool observed, not ground truth (see :class:`NotFound`).

    Note:
        ``kind`` is a ``ClassVar``, so ``dataclasses.asdict()`` does NOT
        include it. ``Tool._serialize_result`` re-injects it explicitly when
        building the JSON the LLM sees.
    """

    kind: ClassVar[str] = ""


@beta
@dataclass(frozen=True)
class Ambiguous(ToolResult):
    """A lookup matched more than one candidate.

    ``ambiguous`` is an observation, not a truth claim: it means "this tool
    found multiple plausible matches from this source at this time". The
    ``matches`` payload gives the next LLM turn something concrete to
    disambiguate against.

    Attributes:
        entity: What was being looked up (e.g. ``"customer"``).
        query: The query string that produced multiple matches.
        matches: Candidate matches, as small JSON-serializable dicts.
    """

    kind: ClassVar[str] = "ambiguous"

    entity: str
    query: str
    matches: List[Dict]


@beta
@dataclass(frozen=True)
class NotFound(ToolResult):
    """A lookup matched nothing.

    ``not_found`` means "this tool observed no match from this source at
    this time" — it does NOT mean "the entity does not exist". Downstream
    consumers (and the next LLM turn) should treat it as an observation
    scoped to one source and one moment, not as a truth claim.

    Attributes:
        entity: What was being looked up (e.g. ``"invoice"``).
        query: The query string that produced no match.
    """

    kind: ClassVar[str] = "not_found"

    entity: str
    query: str


@beta
@dataclass(frozen=True)
class Artifact:
    """A file produced by a tool, delivered out-of-band from the reply text.

    The shape is deliberately more than a URL: URLs rot and signed links
    expire, and some consumers must never store the body. ``sha256`` and
    ``size`` let them identify which artifact was produced anyway.

    Attributes:
        url: Location of the artifact (signed URL, object-store path, ...).
        mime_type: MIME type, e.g. ``"image/png"``.
        filename: Suggested filename for download/display.
        sha256: Hex digest of the artifact body, for identity across URL rot.
        size: Body size in bytes.
        role: Consumer hint, e.g. ``"primary"`` or ``"preview"``.
        retention: Retention hint, e.g. ``"30d"`` or ``"ephemeral"``.
    """

    url: str
    mime_type: Optional[str] = None
    filename: Optional[str] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    role: Optional[str] = None
    retention: Optional[str] = None


# Per-run artifact collector. The agent sets a fresh list at run start
# (Agent._prepare_run) and drains it into AgentResult.artifacts. Tools append
# via emit_artifact(). A ContextVar keeps concurrent runs (asyncio tasks,
# batch() clones in threads) isolated; thread-pool tool execution sites copy
# the caller's context so the SAME list object is visible in worker threads
# (see pitfall #28 / BUG-32).
_artifact_collector: contextvars.ContextVar[Optional[List[Artifact]]] = contextvars.ContextVar(
    "selectools_artifact_collector", default=None
)


def _begin_artifact_collection() -> List[Artifact]:
    """Install a fresh per-run collector list and return it (internal)."""
    artifacts: List[Artifact] = []
    _artifact_collector.set(artifacts)
    return artifacts


@beta
def emit_artifact(
    url: str,
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    sha256: Optional[str] = None,
    size: Optional[int] = None,
    role: Optional[str] = None,
    retention: Optional[str] = None,
) -> Artifact:
    """Attach an :class:`Artifact` to the current agent run.

    Call this from inside a tool function while the agent is executing it.
    The artifact is collected per-run and surfaced on
    ``AgentResult.artifacts`` — keep the reply string for the LLM, and let
    channel layers (chat, email, Slack, ...) deliver the artifact.

    Outside an agent run (no active collector) this is a no-op that still
    returns the constructed :class:`Artifact`, so tools remain directly
    callable in tests and scripts.

    Args:
        url: Location of the artifact.
        mime_type: MIME type, e.g. ``"image/png"``.
        filename: Suggested filename.
        sha256: Hex digest of the body.
        size: Body size in bytes.
        role: Consumer hint, e.g. ``"primary"`` or ``"preview"``.
        retention: Retention hint, e.g. ``"30d"``.

    Returns:
        The constructed :class:`Artifact`.
    """
    artifact = Artifact(
        url=url,
        mime_type=mime_type,
        filename=filename,
        sha256=sha256,
        size=size,
        role=role,
        retention=retention,
    )
    collector = _artifact_collector.get()
    if collector is not None:
        collector.append(artifact)
    return artifact
