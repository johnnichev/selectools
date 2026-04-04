#!/usr/bin/env python3
"""
Multimodal Messages -- send images to your agent.

No API key needed for this demo. Shows how to create messages with images
for vision-enabled models (GPT-4o, Claude 3.5, Gemini).

Run: python examples/81_multimodal_messages.py
"""

from selectools.types import ContentPart, Message, Role, image_message, text_content

print("=== Multimodal Messages Example ===\n")

# 1. Simple image message from URL
msg = image_message("https://example.com/photo.jpg", "What do you see in this image?")
print(f"Image URL message: {len(msg.content_parts)} parts")
print(f"  Text: {msg.content_parts[0].text!r}")
print(f"  Image: {msg.content_parts[1].image_url!r}")

# 2. Multiple images in one message
msg_multi = Message(
    role=Role.USER,
    content="Compare these two images",
    content_parts=[
        ContentPart(type="text", text="Compare these two product photos"),
        ContentPart(type="image_url", image_url="https://example.com/product_a.jpg"),
        ContentPart(type="image_url", image_url="https://example.com/product_b.jpg"),
    ],
)
print(f"\nMulti-image message: {len(msg_multi.content_parts)} parts")

# 3. Extract text from multimodal message
extracted = text_content(msg_multi)
print(f"Extracted text: {extracted!r}")

# 4. Backward compatibility -- str content still works
plain = Message(role=Role.USER, content="Just plain text, no images")
print(f"\nPlain text: {text_content(plain)!r}")
print(f"content_parts is None: {plain.content_parts is None}")

# 5. Usage with an agent (pattern)
print(
    """
# With a vision-enabled model:
from selectools import Agent
from selectools.providers import OpenAIProvider

agent = Agent(tools=[], provider=OpenAIProvider())
msg = image_message("photo.jpg", "Describe this image")
result = agent.run(msg)
print(result.content)
"""
)

print("Done!")
