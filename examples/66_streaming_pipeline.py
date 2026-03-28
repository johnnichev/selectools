"""
Example 66: Streaming Pipeline with astream()

Demonstrates pipeline streaming (v0.19.0):
- Build a multi-step pipeline with @step
- Use astream() to stream the last step's output
- Earlier steps run to completion; the final step yields chunks
- Works with both sync generators and async generators

Run:
    python examples/66_streaming_pipeline.py
"""

import asyncio
from typing import Dict

from selectools.pipeline import Pipeline, step

# --- Pipeline steps ---


@step
def preprocess(text: str) -> str:
    """Normalize and clean the input text."""
    cleaned = text.strip().lower()
    return cleaned


@step
def analyze(text: str) -> Dict[str, int]:
    """Count words and characters."""
    words = text.split()
    return {
        "text": text,
        "word_count": len(words),
        "char_count": len(text),
        "unique_words": len(set(words)),
    }


@step
def format_report(data: Dict[str, int]) -> str:
    """Format the analysis into a readable report."""
    return (
        f"Text Analysis Report\n"
        f"====================\n"
        f"Words: {data['word_count']}\n"
        f"Characters: {data['char_count']}\n"
        f"Unique words: {data['unique_words']}\n"
        f"Original: {data['text'][:50]}...\n"
    )


# A generator step for streaming output character by character
def stream_chars(report: str):
    """Yield the report one character at a time (simulates streaming)."""
    for char in report:
        yield char


async def demo_sync_generator_stream():
    """Stream a pipeline where the last step is a sync generator."""
    print("=== Streaming with sync generator ===")

    pipeline = preprocess | analyze | format_report | stream_chars

    print(f"Pipeline: {pipeline}")
    print(f"Steps: {len(pipeline.steps)}")
    print()

    collected = []
    async for chunk in pipeline.astream("  The quick brown fox jumps over the lazy dog  "):
        collected.append(chunk)
        print(chunk, end="", flush=True)

    print(f"\n\nStreamed {len(collected)} chunks")


async def demo_single_output_stream():
    """Stream a pipeline where the last step returns a single value."""
    print("\n=== Streaming with single-output last step ===")

    pipeline = preprocess | analyze | format_report

    print(f"Pipeline: {pipeline}")
    print()

    async for chunk in pipeline.astream("composable pipelines are powerful and simple"):
        # Only one chunk since format_report returns a single string
        print(chunk)

    print("(Single chunk yielded)")


async def demo_async_generator_stream():
    """Stream a pipeline where the last step is an async generator."""
    print("\n=== Streaming with async generator ===")

    async def stream_words(report: str):
        """Yield the report one word at a time with simulated delay."""
        for word in report.split():
            await asyncio.sleep(0.01)  # simulate network latency
            yield word + " "

    pipeline = preprocess | analyze | format_report | stream_words

    print(f"Pipeline: {pipeline}")
    print()

    word_count = 0
    async for chunk in pipeline.astream("streaming makes user interfaces feel responsive"):
        print(chunk, end="", flush=True)
        word_count += 1

    print(f"\n\nStreamed {word_count} words")


async def main() -> None:
    print("=" * 60)
    print("Streaming Pipeline Demo")
    print("=" * 60)

    await demo_sync_generator_stream()
    await demo_single_output_stream()
    await demo_async_generator_stream()

    print("\nDone! pipeline.astream() streams chunks from the final step.")


if __name__ == "__main__":
    asyncio.run(main())
