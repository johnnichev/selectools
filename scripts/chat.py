"""
Bounding Box Detection Demo

This module demonstrates the AI tool calling framework with a practical use case:
detecting objects in images and drawing bounding boxes around them.

The demo uses OpenAI's GPT-4 Vision API to identify object locations and Pillow
to draw the bounding boxes on the images.

Usage:
    python chat.py

This will:
1. Load the dog.png image
2. Create an agent with bounding box detection capability
3. Ask the agent to detect and draw a bounding box around the dog
4. Output the result as dog_with_bbox.png
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent, Tool, Message, Role, ToolParameter
from PIL import Image, ImageDraw, ImageFont
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"


def detect_bounding_box_impl(target_object: str, image_path: str) -> str:
    """
    Detect and draw a bounding box around a target object in an image.
    
    This function uses OpenAI's Vision API to identify the location of an object
    in an image, then uses Pillow to draw a bounding box around it.
    
    Args:
        target_object: Description of the object to find (e.g., "dog", "cat", "person")
        image_path: Path to the image file to analyze
    
    Returns:
        JSON string containing:
        - success: Whether the detection was successful
        - message: Human-readable status message
        - coordinates: Bounding box coordinates (if successful)
        - output_path: Path to the output image with bounding box
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If the API call or image processing fails
    
    Example:
        >>> result = detect_bounding_box_impl("dog", "dog.png")
        >>> data = json.loads(result)
        >>> print(data["message"])
        Successfully detected dog and drew bounding box
    """
    print(f"\n{'='*60}")
    print(f"Bounding Box Detection Tool")
    print(f"{'='*60}")
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        candidate = ASSETS_DIR / resolved_path
        if candidate.exists():
            resolved_path = candidate

    resolved_path = resolved_path.resolve()

    print(f"Target Object: {target_object}")
    print(f"Image Path: {resolved_path}")
    
    if not resolved_path.exists():
        error_result = {
            "success": False,
            "message": f"Image file not found: {resolved_path}",
            "coordinates": None,
            "output_path": None
        }
        return json.dumps(error_result)
    
    try:
        from openai import OpenAI
    except ImportError:
        error_result = {
            "success": False,
            "message": "OpenAI library is required. Install with: pip install openai",
            "coordinates": None,
            "output_path": None
        }
        return json.dumps(error_result)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_result = {
            "success": False,
            "message": "OPENAI_API_KEY environment variable not set",
            "coordinates": None,
            "output_path": None
        }
        return json.dumps(error_result)
    
    client = OpenAI(api_key=api_key)
    
    import base64
    with open(resolved_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    print("\nQuerying OpenAI Vision API for object location...")
    
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        print(f"\nVision API Response:\n{response_text}")
        
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            detection_data = json.loads(json_match.group())
        else:
            detection_data = json.loads(response_text)
        
        if not detection_data.get("found", False):
            result = {
                "success": False,
                "message": f"Could not find {target_object} in the image. {detection_data.get('description', '')}",
                "coordinates": None,
                "output_path": None
            }
            return json.dumps(result)
        
        x_min = float(detection_data["x_min"])
        y_min = float(detection_data["y_min"])
        x_max = float(detection_data["x_max"])
        y_max = float(detection_data["y_max"])
        
        if not (0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1):
            result = {
                "success": False,
                "message": f"Invalid coordinates returned (must be between 0 and 1): {detection_data}",
                "coordinates": None,
                "output_path": None
            }
            return json.dumps(result)
        
        print(f"\nDetection successful!")
        print(f"  Normalized coordinates: ({x_min:.3f}, {y_min:.3f}) to ({x_max:.3f}, {y_max:.3f})")
        print(f"  Confidence: {detection_data.get('confidence', 'unknown')}")
        print(f"  Description: {detection_data.get('description', 'N/A')}")
        
        image = Image.open(resolved_path)
        width, height = image.size
        
        pixel_x_min = int(x_min * width)
        pixel_y_min = int(y_min * height)
        pixel_x_max = int(x_max * width)
        pixel_y_max = int(y_max * height)
        
        print(f"  Pixel coordinates: ({pixel_x_min}, {pixel_y_min}) to ({pixel_x_max}, {pixel_y_max})")
        
        draw = ImageDraw.Draw(image)
        
        box_thickness = max(3, int(min(width, height) * 0.005))
        
        for offset in range(box_thickness):
            draw.rectangle(
                [
                    pixel_x_min - offset,
                    pixel_y_min - offset,
                    pixel_x_max + offset,
                    pixel_y_max + offset
                ],
                outline="red",
                width=1
            )
        
        label = target_object.upper()
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(20, int(height * 0.03)))
        except:
            try:
                font = ImageFont.truetype("arial.ttf", size=max(20, int(height * 0.03)))
            except:
                font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        label_x = pixel_x_min
        label_y = pixel_y_min - text_height - 10
        if label_y < 0:
            label_y = pixel_y_min + 5
        
        padding = 5
        draw.rectangle(
            [
                label_x - padding,
                label_y - padding,
                label_x + text_width + padding,
                label_y + text_height + padding
            ],
            fill="red"
        )
        
        draw.text((label_x, label_y), label, fill="white", font=font)
        
        base_name = resolved_path.stem
        output_dir = resolved_path.parent
        output_path = output_dir / f"{base_name}_with_bbox.png"
        image.save(output_path)
        output_path_str = str(output_path)
        
        print(f"\nOutput saved to: {output_path}")
        
        result = {
            "success": True,
            "message": f"Successfully detected {target_object} and drew bounding box. Output saved to {output_path}",
            "coordinates": {
                "normalized": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                },
                "pixels": {
                    "x_min": pixel_x_min,
                    "y_min": pixel_y_min,
                    "x_max": pixel_x_max,
                    "y_max": pixel_y_max
                }
            },
            "output_path": output_path_str,
            "confidence": detection_data.get("confidence", "unknown"),
            "description": detection_data.get("description", "")
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as error:
        result = {
            "success": False,
            "message": f"Failed to parse Vision API response as JSON: {str(error)}",
            "coordinates": None,
            "output_path": None
        }
        return json.dumps(result)
    
    except Exception as error:
        result = {
            "success": False,
            "message": f"Error during detection: {str(error)}",
            "coordinates": None,
            "output_path": None
        }
        return json.dumps(result)


def create_bounding_box_tool() -> Tool:
    """
    Create the bounding box detection tool.
    
    Returns:
        Tool instance configured for bounding box detection
    
    Example:
        >>> tool = create_bounding_box_tool()
        >>> tool.name
        'detect_bounding_box'
    """
    return Tool(
        name="detect_bounding_box",
        description="Detects and draws a bounding box around a specific object in an image. Uses computer vision to identify the object location and creates a new image with the bounding box overlaid. Returns the coordinates and path to the output image.",
        parameters=[
            ToolParameter(
                name="target_object",
                param_type=str,
                description="The object to find in the image (e.g., 'dog', 'cat', 'person', 'car'). Be specific if there are multiple similar objects.",
                required=True
            ),
            ToolParameter(
                name="image_path",
                param_type=str,
                description="Path to the image file to analyze. Must be a valid image file (jpg, png, etc.).",
                required=True
            )
        ],
        function=detect_bounding_box_impl
    )


def demo_simple_detection():
    """
    Run a simple demonstration of bounding box detection.
    
    This demo:
    1. Creates an agent with the bounding box detection tool
    2. Asks it to detect a dog in dog.png
    3. Outputs the result to dog_with_bbox.png
    
    Example:
        >>> demo_simple_detection()
        Running Bounding Box Detection Demo...
        ...
        Demo completed successfully!
    """
    print("\n" + "="*60)
    print("BOUNDING BOX DETECTION DEMO")
    print("="*60)
    
    default_image = ASSETS_DIR / "dog.png"

    if not default_image.exists():
        print(f"\nError: dog.png not found at {default_image}")
        print("Please ensure assets/dog.png exists before running this demo.")
        return
    
    print("\nCreating agent with bounding box detection tool...")
    bbox_tool = create_bounding_box_tool()
    agent = Agent(tools=[bbox_tool], max_iterations=5)
    
    print("\nRunning detection task...")
    messages = [
        Message(
            role=Role.USER,
            content=f"Please detect the dog in the image and draw a bounding box around it. The image is located at {default_image}",
            image_path=str(default_image)
        )
    ]
    
    response = agent.run(messages=messages)
    
    print("\n" + "="*60)
    print("FINAL RESPONSE")
    print("="*60)
    print(response.content)
    print("\n" + "="*60)
    
    expected_output = ASSETS_DIR / "dog_with_bbox.png"
    if expected_output.exists():
        print(f"\n✓ Success! Bounding box image created: {expected_output}")
    else:
        print("\n✗ Warning: Output image was not created")


def demo_interactive_chat():
    """
    Run an interactive chat session with the agent.
    
    This allows users to have a conversation with the agent and request
    bounding box detection on various images.
    
    Type 'exit' to quit the chat.
    
    Example:
        >>> demo_interactive_chat()
        Interactive Chat with Bounding Box Detection Agent
        ...
        You: Find the dog in dog.png
        AI: I'll detect the dog for you...
    """
    print("\n" + "="*60)
    print("INTERACTIVE CHAT WITH BOUNDING BOX DETECTION")
    print("="*60)
    print("\nYou can ask the agent to detect objects in images.")
    print("Example: 'Find the dog in dog.png and draw a bounding box'")
    print("\nType 'exit' to quit.\n")
    
    bbox_tool = create_bounding_box_tool()
    agent = Agent(tools=[bbox_tool], max_iterations=5)
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        messages.append(Message(role=Role.USER, content=user_input))
        
        try:
            response = agent.run(messages=messages)
            messages.append(response)
            print(f"\nAI: {response.content}\n")
        except Exception as error:
            print(f"\nError: {str(error)}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        demo_interactive_chat()
    else:
        demo_simple_detection()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nTo run in interactive mode, use:")
        print("  python chat.py --interactive")
