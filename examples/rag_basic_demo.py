"""
Basic RAG (Retrieval-Augmented Generation) Demo

This example demonstrates how to use selectools to create an AI agent
that can answer questions about your documents using RAG.

Requirements:
    pip install selectools numpy

Optional (for other vector stores):
    pip install selectools[rag]  # Includes chromadb, pinecone, etc.
"""

from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import Document, DocumentLoader, RAGAgent, VectorStore


def main() -> None:
    print("\n" + "=" * 70)
    print("🤖 RAG Demo: Question Answering with Document Knowledge")
    print("=" * 70 + "\n")

    # Step 1: Create some sample documents
    print("📄 Creating sample documents...")
    documents = [
        Document(
            text=(
                "Selectools is a Python library for building AI agents that can call your custom "
                "Python functions. It supports multiple LLM providers including OpenAI, Anthropic, "
                "Gemini, and Ollama."
            ),
            metadata={"source": "intro.txt", "section": "overview"},
        ),
        Document(
            text=(
                "To install selectools, run: pip install selectools. "
                "For RAG features, install with: pip install selectools[rag]. "
                "This includes support for embeddings and vector stores."
            ),
            metadata={"source": "install.txt", "section": "installation"},
        ),
        Document(
            text=(
                "Selectools v0.8.0 introduces RAG (Retrieval-Augmented Generation) support. "
                "This includes 4 embedding providers (OpenAI, Anthropic, Gemini, Cohere) and "
                "4 vector store backends (in-memory, SQLite, Chroma, Pinecone). You can now "
                "build agents that answer questions about your documents."
            ),
            metadata={"source": "features.txt", "section": "rag"},
        ),
        Document(
            text=(
                "The RAGAgent API provides convenient methods like from_documents(), "
                "from_directory(), and from_files() to quickly create document-aware agents. "
                "These agents automatically embed your documents, store them in a vector database, "
                "and search for relevant information when answering questions."
            ),
            metadata={"source": "api.txt", "section": "usage"},
        ),
    ]
    print(f"✅ Created {len(documents)} documents\n")

    # Step 2: Set up embedding provider and vector store
    print("🔧 Setting up components...")
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
    vector_store = VectorStore.create("memory", embedder=embedder)
    print(f"✅ Using {embedder.model} for embeddings")
    print(f"✅ Using in-memory vector store\n")

    # Step 3: Create RAG agent
    print("🤖 Creating RAG agent...")
    agent = RAGAgent.from_documents(
        documents=documents,
        provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
        vector_store=vector_store,
        chunk_size=500,  # Split documents into chunks
        chunk_overlap=50,  # Overlap for context
        top_k=2,  # Retrieve top 2 most relevant chunks
    )
    print("✅ RAG agent created and ready!\n")

    # Step 4: Ask questions
    questions = [
        "What is selectools?",
        "How do I install it?",
        "What's new in version 0.8.0?",
        "Tell me about the RAGAgent API",
    ]

    print("=" * 70)
    print("💬 Asking questions about the documents...")
    print("=" * 70 + "\n")

    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        print("-" * 70)

        try:
            from selectools import Message, Role

            response = agent.run([Message(role=Role.USER, content=question)])
            print(f"A{i}: {response.content}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

    # Step 5: Show usage statistics
    print("=" * 70)
    print("📊 Usage Statistics")
    print("=" * 70)
    print(agent.usage)

    # Alternative: Load from files
    print("\n" + "=" * 70)
    print("📁 Alternative: Loading from files")
    print("=" * 70 + "\n")

    print("You can also create a RAG agent from files or directories:")
    print()
    print("# From a directory:")
    print("agent = RAGAgent.from_directory(")
    print('    directory="./docs",')
    print('    glob_pattern="**/*.md",')
    print("    provider=OpenAIProvider(),")
    print("    vector_store=vector_store")
    print(")")
    print()
    print("# From specific files:")
    print("agent = RAGAgent.from_files(")
    print('    file_paths=["doc1.txt", "doc2.pdf", "manual.md"],')
    print("    provider=OpenAIProvider(),")
    print("    vector_store=vector_store")
    print(")")
    print()

    # Show different vector store options
    print("=" * 70)
    print("🗄️  Vector Store Options")
    print("=" * 70 + "\n")

    print("1. In-Memory (default, great for prototyping):")
    print('   store = VectorStore.create("memory", embedder=embedder)')
    print()
    print("2. SQLite (persistent local storage):")
    print('   store = VectorStore.create("sqlite", embedder=embedder, db_path="my_docs.db")')
    print()
    print("3. Chroma (advanced features, requires chromadb):")
    print(
        '   store = VectorStore.create("chroma", embedder=embedder, persist_directory="./chroma")'
    )
    print()
    print("4. Pinecone (cloud-hosted, requires pinecone-client):")
    print('   store = VectorStore.create("pinecone", embedder=embedder, index_name="my-index")')
    print()

    print("\n✨ Demo complete! Check out the other examples for more advanced usage.\n")


if __name__ == "__main__":
    main()
