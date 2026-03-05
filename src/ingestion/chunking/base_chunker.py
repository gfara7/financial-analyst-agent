"""
src/ingestion/chunking/base_chunker.py
────────────────────────────────────────
Abstract base class and shared Chunk dataclass.

The Chunk dataclass is the contract between the chunking layer and the
embedding/indexing layer. Rich metadata enables:
  - Filtered retrieval (document_type, chunk_type, section)
  - Precise citations in the final report (page, section_path)
  - Strategy comparison benchmarks (chunking_strategy field)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from ..pdf_parser import ParsedDocument

# Token encoder — cl100k_base is used by both GPT-4 and text-embedding-3-small
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Chunk:
    """
    A unit of text ready for embedding and indexing.

    All fields are stored verbatim in Azure AI Search.
    The `embedding` field is populated later by embedder.py.
    """

    # Identity
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_title: str = ""
    document_type: str = ""             # "10k" | "fsr" | "basel3" | "ecb_bulletin"

    # Content
    text: str = ""
    token_count: int = 0

    # Location
    page_start: int = 0
    page_end: int = 0

    # Structure
    section_path: list[str] = field(default_factory=list)
    section_path_str: str = ""          # " > ".join(section_path) for indexing

    # Classification
    chunk_type: str = "prose"           # "prose" | "table" | "caption" | "footnote"
    table_context: str = ""             # Preceding prose paragraph for table chunks
    footnote_refs: list[str] = field(default_factory=list)

    # Provenance
    chunking_strategy: str = ""         # "fixed_size" | "semantic" | "hierarchical"

    # Populated by embedder.py
    embedding: Optional[list[float]] = None

    def to_search_document(self) -> dict:
        """Serialize to the Azure AI Search index document format."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_type": self.document_type,
            "text": self.text,
            "token_count": self.token_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path_str": self.section_path_str,
            "chunk_type": self.chunk_type,
            "table_context": self.table_context,
            "chunking_strategy": self.chunking_strategy,
            "embedding": self.embedding,
        }


class BaseChunker(ABC):
    """
    Abstract base for all chunking strategies.

    All subclasses receive a ParsedDocument and return List[Chunk].
    The ParsedDocument contains structured blocks (heading/prose/table/caption/footnote)
    already extracted by PDFParser.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Identifier stored in Chunk.chunking_strategy."""
        ...

    @abstractmethod
    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        """Convert a parsed document into a list of Chunks."""
        ...

    def _make_chunk(
        self,
        text: str,
        parsed_doc: ParsedDocument,
        page_start: int,
        page_end: int,
        section_path: list[str],
        chunk_type: str = "prose",
        table_context: str = "",
        footnote_refs: list[str] | None = None,
    ) -> Chunk:
        """Factory helper to create a well-formed Chunk."""
        section_path_str = " > ".join(section_path) if section_path else ""
        return Chunk(
            document_id=parsed_doc.document_id,
            document_title=parsed_doc.document_title,
            document_type="",  # Set by pipeline.py from document registry
            text=text,
            token_count=count_tokens(text),
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            section_path_str=section_path_str,
            chunk_type=chunk_type,
            table_context=table_context,
            footnote_refs=footnote_refs or [],
            chunking_strategy=self.strategy_name,
        )
