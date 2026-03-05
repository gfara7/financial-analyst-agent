"""
src/ingestion/chunking/fixed_size_chunker.py
─────────────────────────────────────────────
Strategy 1: Naive fixed-size chunker (BASELINE).

How it works:
  - Flatten all page text into a single token stream (ignoring block structure)
  - Slide a window of max_tokens with overlap_tokens stride
  - Zero awareness of section boundaries, tables, or footnotes

Why include it?
  This is the "wrong way" to chunk financial PDFs. It is here SPECIFICALLY to
  demonstrate failure modes during the compare_chunkers.py benchmark:
    - Tables are split mid-row → retrieved rows are meaningless fragments
    - Section context is lost → a chunk about "credit risk" has no section header
    - Footnotes appear randomly inside prose chunks

Interview point:
  "I included the fixed-size baseline to benchmark against. On the JPMorgan 10-K,
   it produces 0% table integrity (every table is split) vs 100% for hierarchical.
   This is why production RAG systems never use fixed-size chunking on structured docs."
"""

from __future__ import annotations

import tiktoken

from ..pdf_parser import ParsedDocument
from .base_chunker import BaseChunker, Chunk, count_tokens

_ENCODER = tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker(BaseChunker):
    """
    Naive sliding token-window chunker.

    Parameters:
        max_tokens     — maximum tokens per chunk (default: 512)
        overlap_tokens — tokens of overlap between adjacent chunks (default: 64)
    """

    @property
    def strategy_name(self) -> str:
        return "fixed_size"

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        # ── Flatten all blocks into one text stream with page tracking ─────────
        # We keep (text, page_num) pairs so we can recover page metadata later
        page_segments: list[tuple[str, int]] = []

        for block in parsed_doc.blocks:
            if block.text.strip():
                page_segments.append((block.text.strip(), block.page_num))

        if not page_segments:
            return []

        # ── Tokenize the full stream ───────────────────────────────────────────
        # Build flat token list alongside token-to-page mapping
        full_tokens: list[int] = []
        token_pages: list[int] = []

        for text, page_num in page_segments:
            tokens = _ENCODER.encode(text)
            full_tokens.extend(tokens)
            token_pages.extend([page_num] * len(tokens))

        # ── Sliding window ─────────────────────────────────────────────────────
        chunks: list[Chunk] = []
        step = max(1, self.max_tokens - self.overlap_tokens)

        for start in range(0, len(full_tokens), step):
            end = min(start + self.max_tokens, len(full_tokens))
            window_tokens = full_tokens[start:end]
            if not window_tokens:
                break

            text = _ENCODER.decode(window_tokens)
            page_start = token_pages[start]
            page_end = token_pages[end - 1]

            chunk = self._make_chunk(
                text=text,
                parsed_doc=parsed_doc,
                page_start=page_start,
                page_end=page_end,
                section_path=[],        # No section awareness
                chunk_type="prose",
            )
            chunks.append(chunk)

            if end >= len(full_tokens):
                break

        return chunks
