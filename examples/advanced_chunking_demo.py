"""
Advanced Chunking Demo: SemanticChunker and ContextualChunker

This example demonstrates:
- TextSplitter (fixed-size) vs RecursiveTextSplitter vs SemanticChunker
- ContextualChunker wrapping RecursiveTextSplitter
- SemanticChunker + ContextualChunker composition
- Chunk count and quality comparison
- Full pipeline: contextual chunks → vector store → search

Requirements:
    pip install selectools
"""

import os
from typing import List

from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import (
    ContextualChunker,
    Document,
    RecursiveTextSplitter,
    SemanticChunker,
    TextSplitter,
    VectorStore,
)

# Long multi-topic document for chunking comparison
MULTI_TOPIC_DOCUMENT = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labelled datasets to train algorithms that classify data or predict outcomes.
Common algorithms include linear regression, decision trees, and neural networks.
Deep learning extends neural networks with many layers to model complex patterns.

The Python programming language is widely used in data science and web development.
Python was created by Guido van Rossum and first released in 1991.
Its syntax emphasises readability and simplicity, making it ideal for beginners.
Popular frameworks include Django for web apps and NumPy for scientific computing.

Climate change refers to long-term shifts in global temperatures and weather patterns.
Human activities, particularly burning fossil fuels, have been the main driver since the 1800s.
The Paris Agreement aims to limit global warming to 1.5 degrees Celsius above pre-industrial levels.
Renewable energy sources like solar and wind are critical for reducing carbon emissions.

Quantum computing uses qubits that can exist in superposition of states.
Unlike classical bits which are 0 or 1, qubits can represent both simultaneously.
This enables quantum computers to solve certain problems exponentially faster.
Major players include IBM, Google, and IonQ in the quantum hardware space.

Italian cuisine is known for its regional diversity and emphasis on fresh ingredients.
Pasta carbonara from Rome uses eggs, pecorino, guanciale, and black pepper.
Pizza originated in Naples, with Margherita being a classic preparation.
Espresso and gelato are central to Italian food culture.
"""


def main() -> None:
    """Run the advanced chunking demonstration."""

    print("=" * 80)
    print("✂️  Advanced Chunking Demo: SemanticChunker & ContextualChunker")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    doc = Document(text=MULTI_TOPIC_DOCUMENT.strip(), metadata={"source": "demo.txt"})
    docs = [doc]

    # -------------------------------------------------------------------------
    # Step 1: Create a long multi-topic document
    # -------------------------------------------------------------------------
    print("\n📄 Step 1: Multi-topic document")
    print("-" * 80)
    print(f"   Document length: {len(doc.text)} characters")
    print(f"   Topics: ML, Python, Climate, Quantum, Italian cooking")
    print(f"   Preview: {doc.text[:100]}...")

    # -------------------------------------------------------------------------
    # Step 2: TextSplitter (fixed-size) result
    # -------------------------------------------------------------------------
    print("\n📏 Step 2: TextSplitter (fixed-size)")
    print("-" * 80)
    fixed = TextSplitter(chunk_size=300, chunk_overlap=50)
    fixed_chunks = fixed.split_documents(docs)
    print(f"   Chunks: {len(fixed_chunks)}")
    for i, c in enumerate(fixed_chunks[:2], 1):
        print(f"   [{i}] {c.text[:80]}...")
    if len(fixed_chunks) > 2:
        print(f"   ... and {len(fixed_chunks) - 2} more")

    # -------------------------------------------------------------------------
    # Step 3: RecursiveTextSplitter result
    # -------------------------------------------------------------------------
    print("\n🔄 Step 3: RecursiveTextSplitter")
    print("-" * 80)
    recursive = RecursiveTextSplitter(chunk_size=300, chunk_overlap=50)
    recursive_chunks = recursive.split_documents(docs)
    print(f"   Chunks: {len(recursive_chunks)}")
    for i, c in enumerate(recursive_chunks[:2], 1):
        print(f"   [{i}] {c.text[:80]}...")
    if len(recursive_chunks) > 2:
        print(f"   ... and {len(recursive_chunks) - 2} more")

    # -------------------------------------------------------------------------
    # Step 4: SemanticChunker result (with real embeddings)
    # -------------------------------------------------------------------------
    print("\n🧠 Step 4: SemanticChunker (topic-boundary splitting)")
    print("-" * 80)
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
    semantic = SemanticChunker(embedder, similarity_threshold=0.70)
    try:
        semantic_chunks = semantic.split_documents(docs)
        print(f"   Chunks: {len(semantic_chunks)}")
        for i, c in enumerate(semantic_chunks[:4], 1):
            topic = c.metadata.get("chunker", "semantic")
            print(f"   [{i}] ({topic}) {c.text[:70]}...")
        if len(semantic_chunks) > 4:
            print(f"   ... and {len(semantic_chunks) - 4} more")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # -------------------------------------------------------------------------
    # Step 5: ContextualChunker wrapping RecursiveTextSplitter
    # -------------------------------------------------------------------------
    print("\n📝 Step 5: ContextualChunker (LLM-enriched chunks)")
    print("-" * 80)
    provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
    base = RecursiveTextSplitter(chunk_size=400, chunk_overlap=80)
    contextual = ContextualChunker(base_chunker=base, provider=provider, model="gpt-4o-mini")
    try:
        contextual_chunks = contextual.split_documents(docs)
        print(f"   Chunks: {len(contextual_chunks)}")
        for i, c in enumerate(contextual_chunks[:2], 1):
            ctx = c.metadata.get("context", "")[:80]
            print(f"   [{i}] Context: {ctx}...")
            print(f"       Text starts with: {c.text[:60]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # -------------------------------------------------------------------------
    # Step 6: SemanticChunker + ContextualChunker composition
    # -------------------------------------------------------------------------
    print("\n🔗 Step 6: SemanticChunker + ContextualChunker composition")
    print("-" * 80)
    semantic_base = SemanticChunker(embedder, similarity_threshold=0.70)
    composed = ContextualChunker(
        base_chunker=semantic_base,
        provider=provider,
        model="gpt-4o-mini",
    )
    try:
        composed_chunks = composed.split_documents(docs)
        print(f"   Chunks: {len(composed_chunks)}")
        for i, c in enumerate(composed_chunks[:2], 1):
            print(f"   [{i}] {c.text[:90]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # -------------------------------------------------------------------------
    # Step 7: Compare chunk counts and quality
    # -------------------------------------------------------------------------
    print("\n📊 Step 7: Chunk count comparison")
    print("-" * 80)
    print("   | Chunker                    | Chunks | Notes                      |")
    print("   |----------------------------|--------|----------------------------|")
    print(
        f"   | TextSplitter (300)        | {len(fixed_chunks):6} | Fixed size, may cut mid-sent |"
    )
    print(
        f"   | RecursiveTextSplitter     | {len(recursive_chunks):6} | Natural boundaries         |"
    )
    try:
        print(
            f"   | SemanticChunker          | {len(semantic_chunks):6} | Topic boundaries           |"
        )
    except NameError:
        print("   | SemanticChunker          |   N/A  | (skipped)                  |")
    try:
        print(
            f"   | ContextualChunker        | {len(contextual_chunks):6} | LLM-enriched               |"
        )
    except NameError:
        print("   | ContextualChunker        |   N/A  | (skipped)                  |")
    try:
        print(
            f"   | Semantic+Contextual      | {len(composed_chunks):6} | Best of both               |"
        )
    except NameError:
        print("   | Semantic+Contextual      |   N/A  | (skipped)                  |")

    # -------------------------------------------------------------------------
    # Step 8: Full pipeline — contextual chunks → vector store → search
    # -------------------------------------------------------------------------
    print("\n🔍 Step 8: Full pipeline — contextual chunks → vector store → search")
    print("-" * 80)
    try:
        pipeline_chunker = ContextualChunker(
            base_chunker=RecursiveTextSplitter(chunk_size=350, chunk_overlap=70),
            provider=provider,
            model="gpt-4o-mini",
        )
        enriched = pipeline_chunker.split_documents(docs)
        store = VectorStore.create("memory", embedder=embedder)
        store.add_documents(enriched)
        query_emb = embedder.embed_query("Who created Python and when?")
        results = store.search(query_emb, top_k=2)
        print(f"   Indexed {len(enriched)} enriched chunks")
        print(f'   Query: "Who created Python and when?"')
        for i, r in enumerate(results, 1):
            print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:80]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 80)
    print("✅ Advanced Chunking Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
