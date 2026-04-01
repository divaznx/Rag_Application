"""
embedder.py  —  OpenAI Embedding Function
==========================================
Uses OpenAI's text-embedding-3-small model.
Compatible with ChromaDB's custom embedding interface.

Set your API key in a .env file or as an environment variable:
    OPENAI_API_KEY=sk-...
"""

import os
from openai import OpenAI

# Load from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to a .env file or export it:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dims, cheap & fast


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI.
    Batches up to 100 texts per API call automatically.
    """
    client = _get_client()
    all_embeddings = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


class OpenAIEmbeddingFunction:
    """Drop-in ChromaDB-compatible embedding function using OpenAI."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return embed_texts(input)

    def name(self) -> str:
        return "openai-text-embedding-3-small"


def get_embedding_function() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction()