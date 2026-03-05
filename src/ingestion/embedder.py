"""
src/ingestion/embedder.py
─────────────────────────
Batch embedding of Chunk objects using Azure OpenAI.

Design decisions:
  - Batches of `batch_size` (default 100) to stay within API rate limits
  - Uses tenacity for automatic retry on transient errors (rate limits, 5xx)
  - Skips empty text chunks rather than sending blanks to the API
  - Returns chunks with `.embedding` field populated in-place
"""

from __future__ import annotations

import time
from typing import Sequence

from openai import AzureOpenAI, RateLimitError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

from src.config import settings
from .chunking.base_chunker import Chunk


class Embedder:
    """
    Wraps Azure OpenAI embeddings with batching and retry logic.
    """

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size or settings.embedding_batch_size
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
        )

    def embed_chunks(self, chunks: list[Chunk], show_progress: bool = True) -> list[Chunk]:
        """
        Embed all chunks in-place. Returns the same list with `.embedding` populated.

        Chunks with empty text are skipped (embedding remains None).
        """
        # Filter to only chunks that need embedding
        to_embed = [c for c in chunks if c.text.strip() and c.embedding is None]

        if not to_embed:
            return chunks

        # Process in batches
        batches = [
            to_embed[i : i + self.batch_size]
            for i in range(0, len(to_embed), self.batch_size)
        ]

        with tqdm(
            total=len(to_embed),
            desc="  Embedding",
            unit="chunks",
            disable=not show_progress,
        ) as pbar:
            for batch in batches:
                texts = [c.text for c in batch]
                embeddings = self._embed_batch(texts)
                for chunk, emb in zip(batch, embeddings):
                    chunk.embedding = emb
                pbar.update(len(batch))

        return chunks

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one batch to Azure OpenAI Embeddings API."""
        # API rejects empty strings; replace with a single space
        cleaned = [t if t.strip() else " " for t in texts]
        response = self._client.embeddings.create(
            input=cleaned,
            model=settings.azure_openai_embedding_deployment,
        )
        return [item.embedding for item in response.data]
