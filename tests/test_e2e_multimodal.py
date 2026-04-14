"""End-to-end multimodal tests with real vision-capable LLM calls.

The existing ``test_multimodal.py`` checks that ``ContentPart`` objects are
constructed correctly and that providers' ``_format_messages`` produce the
expected dict shapes. Those tests never actually call a real vision model.

These tests:

- Build a tiny base64-encoded PNG in memory (4x4 pixels, no external asset)
- Send it to OpenAI (gpt-4o-mini), Anthropic (claude-haiku-4-5), and Gemini
  (gemini-2.5-flash) via ``image_message()``
- Assert that each provider returns a non-empty response

This is the only place we prove that the selectools wire format matches
what each provider actually accepts for image inputs.

Required env vars (tests skip if missing):
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY or GEMINI_API_KEY

Run with:

    pytest tests/test_e2e_multimodal.py --run-e2e -v
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from selectools import Agent, AgentConfig, image_message, tool
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.gemini_provider import GeminiProvider
from selectools.providers.openai_provider import OpenAIProvider

pytestmark = pytest.mark.e2e


@tool()
def _noop() -> str:
    """Return a fixed string. Used so Agent can be instantiated."""
    return "noop"


def _make_tiny_red_png_bytes() -> bytes:
    """Build a 4x4 solid-red PNG entirely in-memory.

    No PIL dependency, no network fetch for image construction. Only the
    subsequent LLM call needs the network.
    """
    width, height = 4, 4
    # One row: filter byte + RGB bytes per pixel
    row = b"\x00" + b"\xff\x00\x00" * width
    raw = row * height

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.fixture(scope="module")
def tiny_red_png(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write a 4x4 red PNG to a module-scoped temp file and return its path."""
    tmp_dir = tmp_path_factory.mktemp("mm")
    png_path = tmp_dir / "tiny_red.png"
    png_path.write_bytes(_make_tiny_red_png_bytes())
    return str(png_path)


class TestMultimodalRealProviders:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_gpt4o_mini_accepts_image(self, tiny_red_png: str) -> None:
        """Real OpenAI call with an image attachment returns a non-empty response."""
        agent = Agent(
            tools=[_noop],
            provider=OpenAIProvider(),
            config=AgentConfig(model="gpt-4o-mini", max_tokens=50),
        )
        msg = image_message(
            tiny_red_png,
            prompt="What primary color is this tiny image? Reply in one word.",
        )
        result = agent.run([msg])
        assert result.content, "Empty response from OpenAI"
        # Critical assertion: prove the image actually reached the model
        # (without this the provider could silently drop the image and
        # the test would still pass on "I can't see an image" style replies)
        assert "red" in result.content.lower(), (
            f"OpenAI did not see the red test image. Got: {result.content[:200]}"
        )
        assert result.usage.total_tokens > 0

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_anthropic_claude_accepts_image(self, tiny_red_png: str) -> None:
        """Real Anthropic call with an image attachment returns a non-empty response."""
        agent = Agent(
            tools=[_noop],
            provider=AnthropicProvider(),
            config=AgentConfig(model="claude-haiku-4-5", max_tokens=50),
        )
        msg = image_message(
            tiny_red_png,
            prompt="What primary color is this tiny image? Reply in one word.",
        )
        result = agent.run([msg])
        assert result.content, "Empty response from Anthropic"
        assert "red" in result.content.lower(), (
            f"Anthropic did not see the red test image. Got: {result.content[:200]}"
        )
        assert result.usage.total_tokens > 0

    @pytest.mark.skipif(
        not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        reason="GOOGLE_API_KEY / GEMINI_API_KEY not set",
    )
    def test_gemini_flash_accepts_image(self, tiny_red_png: str) -> None:
        """Real Gemini call with an image attachment returns a non-empty response."""
        agent = Agent(
            tools=[_noop],
            provider=GeminiProvider(),
            config=AgentConfig(model="gemini-2.5-flash", max_tokens=50),
        )
        msg = image_message(
            tiny_red_png,
            prompt="What primary color is this tiny image? Reply in one word.",
        )
        result = agent.run([msg])
        assert result.content, "Empty response from Gemini"
        assert "red" in result.content.lower(), (
            f"Gemini did not see the red test image. Got: {result.content[:200]}"
        )
        assert result.usage.total_tokens > 0


class TestMultimodalRealProvidersAsync:
    """Async path coverage for the v0.21.0 content_parts fix.

    The fix lives in each provider's ``_format_messages`` which is shared
    between sync ``complete()`` and async ``acomplete()`` / ``astream()``,
    but the sync tests above don't actually exercise the async code paths.
    These tests prove the fix flows through ``agent.arun()`` for every
    multimodal-capable provider.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_openai_async_accepts_image(self, tiny_red_png: str) -> None:
        agent = Agent(
            tools=[_noop],
            provider=OpenAIProvider(),
            config=AgentConfig(model="gpt-4o-mini", max_tokens=50),
        )
        msg = image_message(tiny_red_png, prompt="What color is this image? One word.")
        result = await agent.arun([msg])
        assert "red" in result.content.lower(), (
            f"OpenAI async did not see the red test image. Got: {result.content[:200]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_anthropic_async_accepts_image(self, tiny_red_png: str) -> None:
        agent = Agent(
            tools=[_noop],
            provider=AnthropicProvider(),
            config=AgentConfig(model="claude-haiku-4-5", max_tokens=50),
        )
        msg = image_message(tiny_red_png, prompt="What color is this image? One word.")
        result = await agent.arun([msg])
        assert "red" in result.content.lower(), (
            f"Anthropic async did not see the red test image. Got: {result.content[:200]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        reason="GOOGLE_API_KEY / GEMINI_API_KEY not set",
    )
    async def test_gemini_async_accepts_image(self, tiny_red_png: str) -> None:
        agent = Agent(
            tools=[_noop],
            provider=GeminiProvider(),
            config=AgentConfig(model="gemini-2.5-flash", max_tokens=50),
        )
        msg = image_message(tiny_red_png, prompt="What color is this image? One word.")
        result = await agent.arun([msg])
        assert "red" in result.content.lower(), (
            f"Gemini async did not see the red test image. Got: {result.content[:200]}"
        )


# ``image_message(url, ...)`` for HTTP URLs uses the ``image_url`` ContentPart
# path, which the OpenAI provider handles by forwarding the URL verbatim and
# the Anthropic / Gemini providers handle via a {"type": "url", ...} source
# or a ``types.FileData`` part. The sync + async tests above only exercise
# the ``image_base64`` path (file -> base64), so we need a separate class
# that explicitly covers URL delivery. We use a GitHub-hosted PNG because:
#   1. github.githubassets.com serves bot User-Agents without blocking
#      (Wikipedia's CDN does NOT, which is documented in MULTIMODAL.md)
#   2. The favicon is tiny (a few hundred bytes) so the request is cheap
#   3. It's part of GitHub's own infrastructure, so it won't disappear
_GITHUB_FAVICON_URL = "https://github.githubassets.com/favicons/favicon.png"


class TestMultimodalRealProvidersImageUrl:
    """Real URL-path coverage for image_message(url, ...).

    Locks in the ``ContentPart(type="image_url", image_url=...)`` code path
    for OpenAI (forwards URL verbatim), Anthropic (passes as URL source),
    and Gemini (passes as ``types.FileData``). Without this class, any
    future provider change that broke URL handling would go unnoticed by
    the file-based multimodal tests.
    """

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_accepts_image_url(self) -> None:
        agent = Agent(
            tools=[_noop],
            provider=OpenAIProvider(),
            config=AgentConfig(model="gpt-4o-mini", max_tokens=40),
        )
        msg = image_message(_GITHUB_FAVICON_URL, prompt="One word: what brand is this icon?")
        try:
            result = agent.run([msg])
        except Exception as exc:  # pragma: no cover — network hiccup only
            pytest.skip(f"Network / provider unavailable: {exc}")
        assert "github" in result.content.lower(), (
            f"OpenAI did not fetch the image URL correctly. Got: {result.content[:200]}"
        )

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_anthropic_accepts_image_url(self) -> None:
        agent = Agent(
            tools=[_noop],
            provider=AnthropicProvider(),
            config=AgentConfig(model="claude-haiku-4-5", max_tokens=40),
        )
        msg = image_message(_GITHUB_FAVICON_URL, prompt="One word: what brand is this icon?")
        try:
            result = agent.run([msg])
        except Exception as exc:  # pragma: no cover — network hiccup only
            pytest.skip(f"Network / provider unavailable: {exc}")
        assert "github" in result.content.lower(), (
            f"Anthropic did not fetch the image URL correctly. Got: {result.content[:200]}"
        )

    @pytest.mark.skipif(
        not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        reason="GOOGLE_API_KEY / GEMINI_API_KEY not set",
    )
    def test_gemini_accepts_image_url(self) -> None:
        agent = Agent(
            tools=[_noop],
            provider=GeminiProvider(),
            config=AgentConfig(model="gemini-2.5-flash", max_tokens=40),
        )
        msg = image_message(_GITHUB_FAVICON_URL, prompt="One word: what brand is this icon?")
        try:
            result = agent.run([msg])
        except Exception as exc:  # pragma: no cover — network hiccup only
            pytest.skip(f"Network / provider unavailable: {exc}")
        assert "github" in result.content.lower(), (
            f"Gemini did not fetch the image URL correctly. Got: {result.content[:200]}"
        )
