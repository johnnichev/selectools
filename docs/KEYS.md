# Environment Variables & Keys

The `selectools` library uses the following keys, organized by capability.

## Core LLM Providers

1.  **OpenAI**
    - `OPENAI_API_KEY`: Required for `OpenAIProvider`.

2.  **Anthropic**
    - `ANTHROPIC_API_KEY`: Required for `AnthropicProvider`.

3.  **Gemini (Google)**
    - `GEMINI_API_KEY`: Primary key for `GeminiProvider`.
    - `GOOGLE_API_KEY`: Fallback key if `GEMINI_API_KEY` is not set.

4.  **Ollama (Local)**
    - No key required. Connects to `localhost:11434` by default.

## Embeddings & Vector Stores

1.  **Cohere**
    - `COHERE_API_KEY`: Required for `CohereEmbeddingProvider`.

2.  **Voyage AI (Anthropic Partner)**
    - `VOYAGE_API_KEY`: Required for `AnthropicEmbeddingProvider` (uses Voyage models).

3.  **Pinecone**
    - `PINECONE_API_KEY`: Required for `PineconeVectorStore`.
    - `PINECONE_ENVIRONMENT`: Required for `PineconeVectorStore` (legacy, optional for serverless).

## Example `.env`

See `.env.example` in the project root for a template.
