"""
src/ingestion/chunking/semantic_chunker.py
───────────────────────────────────────────
Strategy 2: Embedding-similarity semantic chunker.

How it works:
  1. Split all prose into sentences using a simple sentence splitter
  2. Embed each sentence (or sliding window of 3 sentences for context)
     using Azure OpenAI text-embedding-3-small
  3. Compute cosine similarity between adjacent sentence-window embeddings
  4. Insert a chunk boundary where similarity drops below the threshold (default: 0.75)
     — this signals a topic change
  5. Merge micro-segments to enforce a minimum token floor
  6. Tables, captions, and footnotes are handled as atomic chunks (not split)

Strengths:
  - Respects topical coherence — avoids putting unrelated paragraphs in the same chunk
  - Excellent for dense policy prose (Basel III numbered regulations, ECB macro analysis)
  - Each chunk covers exactly one conceptual topic

Weaknesses:
  - ~2x more embedding API calls during ingestion compared to fixed-size
  - The similarity threshold is a hyperparameter that may need tuning per corpus
  - No section-path metadata (unlike hierarchical)

Interview talking point:
  "Semantic chunking is the right choice for policy-dense documents like Basel III,
   where a paragraph about 'credit risk-weighted assets' and the next about 'leverage
   ratio calculation' should never be in the same retrieval chunk. A fixed-size chunker
   would put them together if they happen to span a 512-token window."
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from openai import AzureOpenAI

from ..pdf_parser import ParsedDocument, PageBlock
from .base_chunker import BaseChunker, Chunk, count_tokens
from src.config import settings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter for financial text.
    Avoids spaCy dependency for the chunking step (spaCy is optional).
    Falls back to period-based splitting with common abbreviation guards.
    """
    import re

    # Protect common financial abbreviations from false sentence splits
    text = re.sub(r"\b(Corp|Inc|Ltd|LLC|Co|p\.l\.c|U\.S|e\.g|i\.e|vs|est|approx|no|No|para|Par|Art|Sec|Fig|fig|Tab|Ref)\.", r"\1<DOT>", text)
    # Protect decimal numbers
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", text)

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"])", text)

    # Restore protected dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """
    Embedding-based semantic boundary detection.

    Parameters:
        similarity_threshold — cosine similarity drop below this triggers a chunk boundary
        window_size          — number of sentences per embedding window (for context)
        min_tokens           — minimum tokens per chunk (prevents micro-chunks)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        window_size: int = 3,
        min_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.min_tokens = min_tokens
        self._client: Optional[AzureOpenAI] = None

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def _get_client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
            )
        return self._client

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Azure OpenAI."""
        if not texts:
            return []
        client = self._get_client()
        # Clean texts (API rejects empty strings)
        cleaned = [t if t.strip() else " " for t in texts]
        response = client.embeddings.create(
            input=cleaned,
            model=settings.azure_openai_embedding_deployment,
        )
        return [item.embedding for item in response.data]

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        # ── Handle non-prose blocks as atomic chunks ───────────────────────────
        prose_runs: list[list[PageBlock]] = []
        current_run: list[PageBlock] = []

        for block in parsed_doc.blocks:
            if block.block_type in ("table", "caption", "footnote"):
                # Flush current prose run
                if current_run:
                    prose_runs.append(current_run)
                    current_run = []
                # Emit atomic chunk
                chunk = self._make_chunk(
                    text=block.text,
                    parsed_doc=parsed_doc,
                    page_start=block.page_num,
                    page_end=block.page_num,
                    section_path=[],
                    chunk_type=block.block_type,
                )
                chunks.append(chunk)
            else:
                current_run.append(block)

        if current_run:
            prose_runs.append(current_run)

        # ── Process each prose run with semantic boundary detection ────────────
        for run in prose_runs:
            run_chunks = self._process_prose_run(run, parsed_doc)
            chunks.extend(run_chunks)

        return chunks

    def _process_prose_run(
        self, blocks: list[PageBlock], parsed_doc: ParsedDocument
    ) -> list[Chunk]:
        """Apply semantic chunking to a contiguous run of prose/heading blocks."""
        if not blocks:
            return []

        # Collect sentences with their source block info
        sentences: list[str] = []
        sentence_pages: list[int] = []
        sentence_heading_levels: list[int] = []

        for block in blocks:
            block_sentences = _split_sentences(block.text)
            for s in block_sentences:
                sentences.append(s)
                sentence_pages.append(block.page_num)
                sentence_heading_levels.append(block.heading_level)

        if not sentences:
            return []

        # ── Create sliding windows for embedding ──────────────────────────────
        # Window of `window_size` sentences gives local context for embeddings
        windows: list[str] = []
        for i in range(len(sentences)):
            start = max(0, i - self.window_size // 2)
            end = min(len(sentences), i + self.window_size // 2 + 1)
            windows.append(" ".join(sentences[start:end]))

        # ── Embed all windows (batched for efficiency) ────────────────────────
        embeddings = self._embed_texts(windows)

        # ── Compute similarities between adjacent windows ─────────────────────
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # ── Detect chunk boundaries ────────────────────────────────────────────
        # Boundary = similarity drop below threshold OR heading encountered
        boundary_indices: list[int] = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold or sentence_heading_levels[i + 1] > 0:
                boundary_indices.append(i + 1)
        boundary_indices.append(len(sentences))

        # ── Build chunks from boundaries ──────────────────────────────────────
        chunks: list[Chunk] = []
        current_section: list[str] = []  # Track heading context

        for seg_start, seg_end in zip(boundary_indices, boundary_indices[1:]):
            segment_sentences = sentences[seg_start:seg_end]
            if not segment_sentences:
                continue

            # Update heading context if segment starts with a heading
            for sent, hlevel in zip(
                sentences[seg_start:seg_end],
                sentence_heading_levels[seg_start:seg_end],
            ):
                if hlevel > 0:
                    # Trim section path to this level
                    current_section = current_section[: hlevel - 1] + [sent]

            segment_text = " ".join(segment_sentences)

            # Merge micro-segments with previous chunk if too small
            if chunks and count_tokens(segment_text) < self.min_tokens:
                # Append to last chunk's text
                last = chunks[-1]
                merged_text = last.text + " " + segment_text
                if count_tokens(merged_text) <= self.max_tokens:
                    last.text = merged_text
                    last.token_count = count_tokens(merged_text)
                    last.page_end = sentence_pages[min(seg_end - 1, len(sentence_pages) - 1)]
                    continue

            # Split oversized segments at max_tokens boundary
            sub_chunks = self._split_at_max_tokens(
                segment_text,
                parsed_doc=parsed_doc,
                page_start=sentence_pages[seg_start],
                page_end=sentence_pages[min(seg_end - 1, len(sentence_pages) - 1)],
                section_path=list(current_section),
            )
            chunks.extend(sub_chunks)

        return chunks

    def _split_at_max_tokens(
        self,
        text: str,
        parsed_doc: ParsedDocument,
        page_start: int,
        page_end: int,
        section_path: list[str],
    ) -> list[Chunk]:
        """Recursively split text at max_tokens if needed."""
        if count_tokens(text) <= self.max_tokens:
            return [
                self._make_chunk(
                    text=text,
                    parsed_doc=parsed_doc,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                )
            ]

        # Split at sentence boundary near midpoint
        sentences = _split_sentences(text)

        # Base case: can't split further (single sentence exceeds max_tokens) — emit as-is
        if len(sentences) <= 1:
            return [
                self._make_chunk(
                    text=text,
                    parsed_doc=parsed_doc,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                )
            ]

        mid = len(sentences) // 2
        first_half = " ".join(sentences[:mid])
        second_half = " ".join(sentences[mid:])

        result = []
        if first_half.strip():
            result.extend(
                self._split_at_max_tokens(first_half, parsed_doc, page_start, page_end, section_path)
            )
        if second_half.strip():
            result.extend(
                self._split_at_max_tokens(second_half, parsed_doc, page_start, page_end, section_path)
            )
        return result
