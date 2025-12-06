"""
Bounding Box Detection Demo using the toolcalling library.

- Default run: processes assets/environment.png and writes environment_with_bbox.png.
- Interactive mode: `python scripts/chat.py --interactive`
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent, AgentConfig, Message, Role  # noqa: E402
from toolcalling.examples.bbox import create_bounding_box_tool  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_MODEL = "gpt-4o"


def run_detection(image_path: Path, target: str = "dog") -> None:
    if not image_path.exists():
        print(f"Image not found at {image_path}")
        return

    bbox_tool = create_bounding_box_tool()
    agent = Agent(tools=[bbox_tool], config=AgentConfig(max_iterations=5, model=DEFAULT_MODEL))

    messages = [
        Message(
            role=Role.USER,
            content=f"Please find the {target} in this image and draw a bounding box.",
            image_path=str(image_path),
        )
    ]

    response = agent.run(messages=messages)
    print("\nFINAL RESPONSE\n" + "=" * 40)
    print(response.content)

    output = image_path.parent / f"{image_path.stem}_with_bbox.png"
    if output.exists():
        print(f"\n✓ Bounding box image created at {output}")
    else:
        print("\nNo output image detected (check logs above).")


def interactive_chat() -> None:
    print("\nInteractive bounding-box chat. Type 'exit' to quit.\n")
    bbox_tool = create_bounding_box_tool()
    agent = Agent(tools=[bbox_tool], config=AgentConfig(max_iterations=6, model=DEFAULT_MODEL))
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append(Message(role=Role.USER, content=user_input))
        response = agent.run(messages=history)
        history.append(response)
        print(f"\nAI: {response.content}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bounding box detection demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    parser.add_argument("--image", help="Path to an image; defaults to assets/environment.png")
    args = parser.parse_args()

    if args.interactive:
        interactive_chat()
    else:
        image = Path(args.image) if args.image else ASSETS_DIR / "environment.png"
        run_detection(image)


if __name__ == "__main__":
    main()
