"""
Bounding-box detection tool example using OpenAI Vision.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont

from ..tools import Tool, ToolParameter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = PROJECT_ROOT / "assets"
BBOX_MOCK_ENV = "SELECTOOLS_BBOX_MOCK_JSON"


def _resolve_image_path(image_path: str) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        asset_candidate = ASSETS_DIR / candidate
        if asset_candidate.exists():
            candidate = asset_candidate
    return candidate.resolve()


def _load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required for bounding-box detection.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run bounding-box detection.")
    return OpenAI(api_key=api_key)


def detect_bounding_box_impl(target_object: str, image_path: str) -> str:
    """
    Detect a target object in an image and draw a bounding box.

    Returns a JSON string containing success status, coordinates, and output path.
    """
    resolved_path = _resolve_image_path(image_path)
    if not resolved_path.exists():
        return json.dumps(
            {
                "success": False,
                "message": f"Image file not found: {resolved_path}",
                "coordinates": None,
                "output_path": None,
            }
        )

    mock_json = os.getenv(BBOX_MOCK_ENV)
    if mock_json:
        detection_data = _load_mock_detection(Path(mock_json))
    else:
        detection_data = _call_openai_vision(target_object=target_object, image_path=resolved_path)

    if not detection_data:
        return json.dumps(
            {
                "success": False,
                "message": "No detection data returned.",
                "coordinates": None,
                "output_path": None,
            }
        )

    if not detection_data.get("found"):
        return json.dumps(
            {
                "success": False,
                "message": f"Could not find {target_object}: {detection_data.get('description', '')}",
                "coordinates": None,
                "output_path": None,
            }
        )

    x_min = float(detection_data["x_min"])
    y_min = float(detection_data["y_min"])
    x_max = float(detection_data["x_max"])
    y_max = float(detection_data["y_max"])

    if not _coordinates_valid(x_min, y_min, x_max, y_max):
        return json.dumps(
            {
                "success": False,
                "message": f"Invalid coordinates returned (must be between 0 and 1): {detection_data}",
                "coordinates": None,
                "output_path": None,
            }
        )

    output_path, pixel_coordinates = _draw_box(resolved_path, target_object, x_min, y_min, x_max, y_max)

    return json.dumps(
        {
            "success": True,
            "message": f"Detected {target_object}; output saved to {output_path}",
            "coordinates": {
                "normalized": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
                "pixels": pixel_coordinates,
            },
            "output_path": str(output_path),
            "confidence": detection_data.get("confidence", "unknown"),
            "description": detection_data.get("description", ""),
        },
        indent=2,
    )


def _parse_detection_response(response_text: str) -> Dict[str, str]:
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return json.loads(response_text)


def _load_mock_detection(mock_path: Path) -> Dict[str, str]:
    """Load a deterministic mock response for offline testing."""
    if not mock_path.exists():
        return {"found": False, "description": f"Mock file not found at {mock_path}"}
    try:
        return json.loads(mock_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "description": f"Failed to read mock file: {exc}"}


def _call_openai_vision(target_object: str, image_path: Path) -> Optional[Dict[str, str]]:
    client = _load_openai_client()
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    prompt = f"""Analyze this image and locate the {target_object}.

Return ONLY a JSON object with the bounding box coordinates in normalized format (0.0 to 1.0):

{{
    "found": true/false,
    "x_min": 0.0-1.0,
    "y_min": 0.0-1.0,
    "x_max": 0.0-1.0,
    "y_max": 0.0-1.0,
    "confidence": "high/medium/low",
    "description": "brief description of what you found"
}}

If the {target_object} is not found, set "found" to false and explain why in the description.
Coordinates should be normalized (0.0 = left/top edge, 1.0 = right/bottom edge).
Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=500,
        )
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "description": f"Vision API error: {exc}"}

    response_text = response.choices[0].message.content
    return _parse_detection_response(response_text)


def _coordinates_valid(x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
    return 0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1


def _draw_box(image_path: Path, target_object: str, x_min: float, y_min: float, x_max: float, y_max: float):
    image = Image.open(image_path)
    width, height = image.size

    pixel_coordinates = {
        "x_min": int(x_min * width),
        "y_min": int(y_min * height),
        "x_max": int(x_max * width),
        "y_max": int(y_max * height),
    }

    draw = ImageDraw.Draw(image)
    thickness = max(3, int(min(width, height) * 0.005))
    for offset in range(thickness):
        draw.rectangle(
            [
                pixel_coordinates["x_min"] - offset,
                pixel_coordinates["y_min"] - offset,
                pixel_coordinates["x_max"] + offset,
                pixel_coordinates["y_max"] + offset,
            ],
            outline="red",
            width=1,
        )

    label = target_object.upper()
    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, int(height * 0.03)))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    label_x = pixel_coordinates["x_min"]
    label_y = pixel_coordinates["y_min"] - text_height - 10
    if label_y < 0:
        label_y = pixel_coordinates["y_min"] + 5

    padding = 5
    draw.rectangle(
        [
            label_x - padding,
            label_y - padding,
            label_x + text_width + padding,
            label_y + text_height + padding,
        ],
        fill="red",
    )
    draw.text((label_x, label_y), label, fill="white", font=font)

    output_path = image_path.parent / f"{image_path.stem}_with_bbox.png"
    image.save(output_path)
    return output_path, pixel_coordinates


def create_bounding_box_tool() -> Tool:
    """Factory for the bounding-box detection tool."""
    return Tool(
        name="detect_bounding_box",
        description=(
            "Detects and draws a bounding box around a specific object in an image. "
            "Returns normalized and pixel coordinates plus the output image path."
        ),
        parameters=[
            ToolParameter(
                name="target_object",
                param_type=str,
                description="The object to locate in the image.",
                required=True,
            ),
            ToolParameter(
                name="image_path",
                param_type=str,
                description="Path to the image file (absolute or relative to assets/).",
                required=True,
            ),
        ],
        function=detect_bounding_box_impl,
    )


__all__ = ["create_bounding_box_tool", "detect_bounding_box_impl"]
