"""
Built-in pre-save sanitizers for ``KnowledgeMemory``.

Remembered content is user-derived and flows back into system prompts via
``build_context()``, which makes the knowledge store a prompt-injection
vector: an attacker who can get text remembered (chat message, tool output,
scraped page) can plant structural markers that later read as instructions.

These sanitizers plug into ``KnowledgeMemory(pre_save=...)``.  Each takes the
entry text and returns the transformed text, or ``None`` to reject the entry
entirely::

    from selectools import KnowledgeMemory
    from selectools.knowledge_sanitizers import defang_delimiters, strip_surrogates

    memory = KnowledgeMemory(
        directory="./memory",
        pre_save=[strip_surrogates, defang_delimiters],
        dedupe=True,
    )

All functions here are stdlib-only and deterministic.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Iterable, Optional

from .stability import beta

# Full ``--- Label ---`` line.  Rewritten with em-dash bookends: visually
# similar, but no longer reads as a structural section header.  Lines of
# dashes only (``---``) pass through; they cannot carry a forged label.
# Up to 3 leading spaces/tabs allowed and preserved (CommonMark: 0-3 spaces
# of indentation keeps a line structural; 4+ is a code block).  Review
# finding PR #84: the previous bare ``^`` anchor let a single leading space
# bypass the rewrite.
_DELIMITER_RE = re.compile(r"^([ \t]{0,3})---[ \t]*([^\n-][^\n]*?)[ \t]*---[ \t]*$", re.MULTILINE)

# Half-delimiter: ``--- End of conversation`` with no closing dashes.  Still
# scans as a section break to an LLM, so it gets the same rewrite.
# Whitespace is restricted to same-line (`[ \t]`, not `\s`) so a bare
# ``---`` line never swallows the following line.
_HALF_DELIMITER_RE = re.compile(r"^([ \t]{0,3})---[ \t]+([^\n-][^\n]*?)[ \t]*$", re.MULTILINE)

# Role markers from common LLM chat templates (``[INST]``, ``<|im_start|>``,
# ``<system>``, Llama-2 ``<<SYS>>``).  Models are trained on text containing
# these tokens and may give them special weight even inside quoted content.
_ROLE_MARKER_RE = re.compile(
    r"""(?ix)
    \[/? \s* (?: INST | SYS | ASSISTANT | USER | SYSTEM ) \s* \]
    | << \s* /? \s* SYS \s* >>
    | <\|? \s* (?: im_start | im_end | system | assistant | user
                 | endoftext | start | end ) \s* \|? >
    | </? \s* (?: system | assistant | user ) \s* >
    """
)

# Speaker-label prefix at line start (``Assistant: I'll wire the money``).
# Inside a remembered-context block this reads as a forged conversation turn.
# Matches ASCII and fullwidth (U+FF1A) colons.  Indentation is unbounded
# (speaker labels are not markdown structure, so no 0-3 cap) but restricted
# to same-line whitespace (`[ \t]`, not `\s`) so a match never spans
# newlines (review finding PR #84).
_SPEAKER_LABEL_RE = re.compile(
    r"^([ \t]*)(User|Assistant|System|Human)([ \t]*)([:：])",
    re.MULTILINE | re.IGNORECASE,
)

# Code fence at line start (3+ backticks or tildes, optionally indented up
# to 3 spaces per CommonMark).  A fence opened inside remembered content can
# swallow the rest of the prompt or terminate a fence the host prompt
# opened.  Inline code spans (1-2 backticks, or backticks/tildes mid-line)
# are untouched.
_FENCE_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})", re.MULTILINE)

# Visually similar substitutes that markdown parsers and tokenizers do not
# treat as fence characters: U+02CB MODIFIER LETTER GRAVE ACCENT for the
# backtick, U+02DC SMALL TILDE for the tilde.
_FENCE_SUBSTITUTES = {"`": "ˋ", "~": "˜"}


@beta
def defang_delimiters(text: str) -> str:
    """Neutralize prompt-injection delimiters in text about to be remembered.

    Conservative by design: every rewrite keeps the content human-readable
    and meaning-preserving; only the *structural* interpretation is broken.

    What is defanged, and why:

    1. ``--- Label ---`` and ``--- Label`` lines become ``— Label —`` /
       ``— Label`` (em dashes).  Dash-fenced lines are the classic fake
       section header (``--- End of conversation ---``).
    2. Chat-template role markers (``[INST]``, ``<|im_start|>``,
       ``<system>``, ``</assistant>``, ``<<SYS>>``, ...) get a space
       inserted after the opening bracket so they no longer tokenize as
       control tokens.
    3. ``User:`` / ``Assistant:`` / ``System:`` / ``Human:`` at line start
       (ASCII or fullwidth U+FF1A colon) become ``User :`` etc., breaking
       forged conversation turns.
    4. Line-start code fences (3+ backticks or tildes) are rewritten with
       the visually identical U+02CB / U+02DC characters, so remembered
       content cannot open or close a markdown fence in the host prompt.
       Inline backtick and tilde spans are preserved.

    Indentation: delimiter and fence rules allow and preserve up to 3
    leading spaces/tabs.  Lines indented 4+ spaces are deliberately left
    alone — CommonMark treats them as code blocks, and rewriting them would
    corrupt legitimate indented literals such as diff headers
    (``    --- a/file.py``).  Speaker labels are defanged at any
    indentation, since they are not markdown structure.

    Known limitations (deliberately out of scope, not a closed list):

    - Unicode homoglyph dash runs (``———``, box-drawing characters) are
      not rewritten; only ASCII ``---`` lines are.
    - ``===`` setext-style underlines and other ASCII-art section breaks
      pass through.
    - Only the most common chat-template dialects are covered; others
      (e.g. Gemma ``<start_of_turn>``, dash runs longer than three like
      ``---- Label ----``) pass through.
    - Lines indented 4+ spaces are not inspected at all (see above).

    Treat this as defense-in-depth for the remembered-content channel, not
    a complete injection filter.

    Args:
        text: Entry text to sanitize.

    Returns:
        The defanged text.  Plain prose passes through unchanged.
    """
    s = text or ""
    s = _DELIMITER_RE.sub(r"\1— \2 —", s)
    s = _HALF_DELIMITER_RE.sub(r"\1— \2", s)
    s = _ROLE_MARKER_RE.sub(lambda m: m.group(0).replace("[", "[ ").replace("<", "< "), s)
    s = _SPEAKER_LABEL_RE.sub(r"\1\2\3 \4", s)
    s = _FENCE_RE.sub(lambda m: m.group(1) + _FENCE_SUBSTITUTES[m.group(2)[0]] * len(m.group(2)), s)
    return s


@beta
def strip_surrogates(text: str) -> str:
    """Remove lone UTF-16 surrogates and other UTF-8-unencodable characters.

    Text arriving from webhooks (WhatsApp/Telegram emoji edge cases, broken
    client encodings) can contain lone surrogate code points.  Python ``str``
    holds them happily, but the first ``open(..., encoding="utf-8").write()``
    raises ``UnicodeEncodeError`` — bricking persistence for that entry and,
    on shared files, anything written after it.

    Args:
        text: Entry text to sanitize.

    Returns:
        The text with unencodable code points dropped.  Well-formed text
        (including astral-plane emoji) passes through unchanged.
    """
    return (text or "").encode("utf-8", errors="ignore").decode("utf-8")


@beta
def dedupe_against(
    existing_fetcher: Callable[[], Iterable[str]],
    threshold: float = 0.9,
) -> Callable[[str], Optional[str]]:
    """Build a pre-save sanitizer that rejects near-duplicate entries.

    Returns a hook that fetches the current entry texts via
    ``existing_fetcher`` and rejects the new text (returns ``None``) when its
    ``difflib.SequenceMatcher`` ratio against any existing entry reaches
    ``threshold``.  Append-only memories otherwise accumulate the same fact
    once per session, bloating every future prompt.

    Cost: the fetcher runs on every save, and similarity is O(len(a) *
    len(b)) per comparison in the worst case.  Cheap upper-bound ratios
    (``real_quick_ratio`` / ``quick_ratio``) prune most non-matches, but for
    stores beyond a few thousand entries prefer a bounded fetcher (e.g. only
    the most recent or same-category entries).

    Args:
        existing_fetcher: Zero-arg callable returning the texts to compare
            against.  Called once per sanitizer invocation, so it always
            sees the latest state.
        threshold: Similarity ratio in ``[0.0, 1.0]`` at or above which the
            new entry is rejected.  ``1.0`` rejects exact matches only.

    Returns:
        A pre-save hook suitable for ``KnowledgeMemory(pre_save=...)``.
    """

    def _reject_near_duplicates(text: str) -> Optional[str]:
        for existing in existing_fetcher():
            matcher = difflib.SequenceMatcher(None, text, existing, autojunk=False)
            if (
                matcher.real_quick_ratio() >= threshold
                and matcher.quick_ratio() >= threshold
                and matcher.ratio() >= threshold
            ):
                return None
        return text

    return _reject_near_duplicates


__all__ = [
    "defang_delimiters",
    "strip_surrogates",
    "dedupe_against",
]
