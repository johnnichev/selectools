"""
Image generation tools -- create images from text prompts.

Uses the OpenAI Images API via the ``openai`` library (a core selectools
dependency; the import is still lazy so the toolbox loads without it).

Authentication uses an API key passed as a parameter or via the
``OPENAI_API_KEY`` environment variable -- the same key the
``OpenAIProvider`` uses. Keys are never echoed in tool output or error
messages.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from ..stability import beta
from ..tools import tool

_MISSING_DEP_ERROR = "Error: 'openai' library not installed. Run: pip install openai"
_MISSING_KEY_ERROR = (
    "Error: No OpenAI API key provided. Pass api_key or set the OPENAI_API_KEY env var."
)
_DEFAULT_MODEL = "gpt-image-1"
_DEFAULT_SIZE = "1024x1024"


def _resolve_key(api_key: Optional[str]) -> str:
    return api_key or os.environ.get("OPENAI_API_KEY", "")


@beta
@tool(description="Generate an image from a text prompt")
def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = _DEFAULT_MODEL,
    size: str = _DEFAULT_SIZE,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate an image from a text prompt via the OpenAI Images API.

    Depending on the model, the API returns either base64 image data
    (``gpt-image-1``) or a hosted URL (``dall-e-3``). Base64 payloads are
    saved as PNG to ``output_path`` (or a temporary file when omitted)
    and the file path is returned; hosted results return the URL.

    Args:
        prompt: Text description of the image to generate.
        output_path: Destination file path for the PNG (parent
            directories are created if missing). Ignored when the API
            returns a URL instead of image data.
        model: Image model to use (default: ``"gpt-image-1"``).
        size: Image dimensions, e.g. ``"1024x1024"``, ``"1536x1024"``.
        api_key: OpenAI API key (falls back to ``OPENAI_API_KEY``). Never
            included in output or errors.

    Returns:
        The saved image path or the hosted image URL, or a readable
        error string.
    """
    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    if not prompt or not prompt.strip():
        return "Error: No prompt provided."

    try:
        client = OpenAI(api_key=key)
        response = client.images.generate(model=model, prompt=prompt, size=size, n=1)
        data = response.data[0] if getattr(response, "data", None) else None
        if data is None:
            return "Error: OpenAI returned no image data."

        url = getattr(data, "url", None)
        b64 = getattr(data, "b64_json", None)

        if b64:
            if output_path and output_path.strip():
                destination = Path(output_path.strip())
                destination.parent.mkdir(parents=True, exist_ok=True)
            else:
                handle = tempfile.NamedTemporaryFile(
                    suffix=".png", prefix="selectools_image_", delete=False
                )
                handle.close()
                destination = Path(handle.name)
            destination.write_bytes(base64.b64decode(b64))
            return f"Image generated with {model} and saved to {destination}"
        if url:
            return f"Image generated with {model}: {url}"
        return "Error: OpenAI response contained neither image data nor a URL."
    except OpenAIError as exc:
        return f"Error: OpenAI Images API call failed: {type(exc).__name__}"
    except OSError as exc:
        return f"Error: Could not write image file: {type(exc).__name__}"
    except Exception as exc:
        return f"Error generating image: {type(exc).__name__}"


__stability__ = "beta"

__all__ = [
    "generate_image",
]
