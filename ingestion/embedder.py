"""
ingestion/embedder.py
Converts text chunks into embedding vectors.

Supports:
  - OpenAI  text-embedding-3-small / large  (default)
  - Local   sentence-transformers           (no API key needed)

Batching is handled automatically to stay within API rate limits.
"""
from __future__ import annotations

import os
import time
from typing import List, Literal

from config.loader import EmbeddingConfig
from ingestion.chunker import Chunk

class BaseEmbedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    """
    Calls OpenAI's embedding endpoint in batches.
    Requires OPENAI_API_KEY in environment (or .env).
    """

    def __init__(self, cfg: EmbeddingConfig):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )

        self.client = OpenAI(api_key=api_key)
        self.cfg = cfg

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, batching to respect API limits."""
        all_embeddings: List[List[float]] = []
        batch_size = self.cfg.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(2):
                try:
                    response = self.client.embeddings.create(
                        model=self.cfg.model,
                        input=batch,
                        dimensions=self.cfg.dimensions,
                    )
                    all_embeddings.extend(
                        [item.embedding for item in response.data]
                    )
                    break
                except Exception as e:
                    if attempt == 0 and "rate" in str(e).lower():
                        time.sleep(5)
                    else:
                        raise

        return all_embeddings


class LocalEmbedder(BaseEmbedder):
    """
    Uses a local sentence-transformers model.
    No API key required — good for offline / low-cost usage.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, cfg: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        model_name = cfg.model if cfg.provider == "local" else self.DEFAULT_MODEL
        self.model = SentenceTransformer(model_name)
        self.batch_size = cfg.batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    """Return the correct embedder based on config."""
    if cfg.provider == "openai":
        return OpenAIEmbedder(cfg)
    elif cfg.provider == "local":
        return LocalEmbedder(cfg)
    else:
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")
