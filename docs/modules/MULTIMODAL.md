---
description: "Multimodal messages — pass images and other content parts to vision-capable LLMs"
tags:
  - core
  - messages
  - multimodal
  - vision
---

# Multimodal Messages

**Import:** `from selectools import ContentPart, image_message, Message`
**Stability:** beta
**Added in:** v0.21.0

`Message.content` now accepts a list of `ContentPart` objects in addition to a plain
string. This unlocks vision and other multimodal inputs across every provider that
supports them: GPT-4o, Claude 3.5/3.7, Gemini, and Ollama vision models.

```python title="multimodal_quick.py"
from selectools import Agent, OpenAIProvider, image_message

agent = Agent(provider=OpenAIProvider(model="gpt-4o"))

# Helper for the common "image + prompt" case
result = agent.run([
    image_message("https://example.com/diagram.png", "What does this diagram show?")
])
print(result.content)
```

!!! tip "See Also"
    - [Providers](PROVIDERS.md) - Which providers support multimodal input
    - [Models](MODELS.md) - Vision-capable model identifiers

---

## ContentPart Anatomy

```python
from selectools import ContentPart, Message, Role

msg = Message(
    role=Role.USER,
    content=[
        ContentPart(type="text", text="Compare these two screenshots."),
        ContentPart(type="image_url", image_url="https://example.com/before.png"),
        ContentPart(type="image_url", image_url="https://example.com/after.png"),
    ],
)
```

| Field | Used when |
|---|---|
| `type` | One of `"text"`, `"image_url"`, `"image_base64"`, `"audio"` |
| `text` | Set when `type == "text"` |
| `image_url` | Public URL for an image (most providers) |
| `image_base64` | Inline base64 payload for an image |
| `media_type` | MIME type, e.g. `"image/png"` or `"audio/wav"` |

---

## Helper: `image_message`

For the common "single image + prompt" case, use the `image_message` helper:

```python
from selectools import image_message

# From a URL
msg = image_message("https://example.com/photo.jpg", "Describe what you see.")

# From a local file path (auto-encoded as base64)
msg = image_message("./screenshots/error.png", "What's the error in this UI?")
```

The helper detects whether the input is a URL or a local path and chooses the
right `ContentPart.type` (`image_url` vs `image_base64`).

!!! warning "URL reachability"
    When you pass an `http://` / `https://` URL, **the provider's backend fetches
    the image**, not selectools. OpenAI, Anthropic Claude, and Google Gemini each
    download the URL server-side. Some hosts block bot User-Agents (Wikimedia
    Commons, many corporate CDNs) and will return 400 / 403 errors. If you hit
    "Unable to download the file" or "Cannot fetch content from the provided URL",
    download the image locally and pass a file path instead — that triggers the
    base64 path which is host-independent.

---

## Provider Compatibility

| Provider | Format used internally |
|---|---|
| OpenAI | `[{"type": "text", ...}, {"type": "image_url", "image_url": {"url": ...}}]` |
| Anthropic | `[{"type": "text", ...}, {"type": "image", "source": {"type": "base64", ...}}]` |
| Gemini | `types.Part` objects with `inline_data` |
| Ollama | `images` parameter (list of base64 strings) |

You don't need to format any of this yourself — selectools handles the conversion
in each provider's `_format_messages()`.

---

## Backward Compatibility

`Message(role=..., content="plain text")` continues to work everywhere. The
`list[ContentPart]` path is opt-in and existing code is unaffected.

```python
# Still works exactly as before
msg = Message(role=Role.USER, content="What is 2 + 2?")
```

---

## API Reference

| Symbol | Description |
|---|---|
| `ContentPart` | Dataclass for a single part of a multimodal message |
| `Message.content` | Now `str \| list[ContentPart]` |
| `image_message(image, prompt)` | Convenience constructor for image + text |
| `text_content(message)` | Extract concatenated text from a (possibly multimodal) Message |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 81 | [`81_multimodal_messages.py`](https://github.com/johnnichev/selectools/blob/main/examples/81_multimodal_messages.py) | Image input with `image_message` and raw `ContentPart` |
