"""
src/ingestion/pipeline.py
─────────────────────────
End-to-end ingestion orchestrator: PDF → chunks → embeddings → Azure AI Search.

The pipeline also:
  - Uploads the source PDF to Azure Blob Storage (audit trail)
  - Writes an ingestion receipt to data/ingested/{document_id}.json
  - Assigns the correct chunking strategy per document type

Usage (from scripts/run_ingestion.py):
    pipeline = IngestionPipeline()
    result = pipeline.ingest(
        pdf_path="data/pdfs/jpmorgan_10k_2023.pdf",
        document_type="10k",
        document_id="jpmorgan_10k_2023",
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from azure.storage.blob import BlobServiceClient
from rich.console import Console

from src.config import settings
from .pdf_parser import PDFParser
from .chunking import (
    HierarchicalChunker,
    SemanticChunker,
    FixedSizeChunker,
)
from .chunking.base_chunker import Chunk
from .embedder import Embedder
from .indexer import Indexer

console = Console()

# ── Per-document-type chunking strategy assignment ─────────────────────────────
#
# The strategy is chosen based on the structural characteristics of each document:
#
#   10k          → Hierarchical: deep section hierarchy (Note 14.3.2) + multi-page
#                  financial statement tables. Table fusion is critical.
#
#   fsr          → Semantic: Federal Reserve reports are dense policy prose.
#                  Topic coherence matters more than section tracking.
#
#   basel3       → Hierarchical: numbered regulatory paragraphs (52.a.iii) and
#                  capital ratio tables. Section path enables filtered retrieval
#                  by regulation number.
#
#   ecb_bulletin → Semantic: macro economic analysis with chart-heavy sections.
#                  Semantic boundaries align better than structural headings.
#
CHUNKING_STRATEGIES = {
    "10k":          lambda: HierarchicalChunker(max_tokens=512, overlap_tokens=64),
    "fsr":          lambda: SemanticChunker(similarity_threshold=0.75, max_tokens=512),
    "basel3":       lambda: HierarchicalChunker(max_tokens=512, overlap_tokens=64),
    "ecb_bulletin": lambda: SemanticChunker(similarity_threshold=0.72, max_tokens=512),
}

# Default fallback
_DEFAULT_STRATEGY = lambda: HierarchicalChunker(max_tokens=512, overlap_tokens=64)


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow for a single PDF document.
    """

    def __init__(self):
        self._parser = PDFParser()
        self._embedder = Embedder()
        self._indexer = Indexer()
        self._blob_client: BlobServiceClient | None = None

    def ingest(
        self,
        pdf_path: str | Path,
        document_type: str,
        document_id: str,
        delete_existing: bool = True,
    ) -> dict:
        """
        Full pipeline: parse → chunk → embed → index.

        Args:
            pdf_path:        Path to the source PDF
            document_type:   One of "10k", "fsr", "basel3", "ecb_bulletin"
            document_id:     Unique identifier (used as primary key prefix)
            delete_existing: If True, delete existing chunks before re-ingesting

        Returns:
            Receipt dict with timing, chunk counts, and token totals.
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()

        console.print(f"\n[bold cyan]Ingesting:[/bold cyan] {pdf_path.name}")
        console.print(f"  Document ID   : {document_id}")
        console.print(f"  Document type : {document_type}")

        # ── 0. Ensure index exists ─────────────────────────────────────────────
        console.print("  Ensuring index exists...")
        self._indexer.ensure_index_exists()

        # ── 1. Upload PDF to Blob Storage ──────────────────────────────────────
        try:
            self._upload_to_blob(pdf_path, document_id)
        except Exception as e:
            console.print(f"  [yellow]Blob upload skipped:[/yellow] {e}")

        # ── 2. Delete existing chunks for this document ────────────────────────
        if delete_existing:
            deleted = self._indexer.delete_document_chunks(document_id)
            if deleted > 0:
                console.print(f"  Deleted {deleted} existing chunks")

        # ── 3. Parse PDF ───────────────────────────────────────────────────────
        console.print("  Parsing PDF...")
        t_parse_start = time.time()
        parsed_doc = self._parser.parse(pdf_path, document_id)
        t_parse = time.time() - t_parse_start
        console.print(
            f"  Parsed {parsed_doc.total_pages} pages, "
            f"{len(parsed_doc.blocks)} blocks ({t_parse:.1f}s)"
        )

        # ── 4. Chunk ───────────────────────────────────────────────────────────
        console.print("  Chunking...")
        t_chunk_start = time.time()
        strategy_factory = CHUNKING_STRATEGIES.get(document_type, _DEFAULT_STRATEGY)
        chunker = strategy_factory()

        chunks: list[Chunk] = chunker.chunk(parsed_doc)

        # Assign document_type (not available in parser)
        for chunk in chunks:
            chunk.document_type = document_type

        t_chunk = time.time() - t_chunk_start
        total_tokens = sum(c.token_count for c in chunks)
        console.print(
            f"  Created {len(chunks)} chunks, "
            f"{total_tokens:,} tokens, "
            f"strategy={chunker.strategy_name} ({t_chunk:.1f}s)"
        )

        # Chunk type breakdown
        type_counts: dict[str, int] = {}
        for chunk in chunks:
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
        for ctype, count in sorted(type_counts.items()):
            console.print(f"    {ctype:12s}: {count:5d}")

        # ── 5. Embed ───────────────────────────────────────────────────────────
        console.print("  Embedding...")
        t_embed_start = time.time()
        self._embedder.embed_chunks(chunks)
        t_embed = time.time() - t_embed_start
        embedded_count = sum(1 for c in chunks if c.embedding is not None)
        console.print(f"  Embedded {embedded_count}/{len(chunks)} chunks ({t_embed:.1f}s)")

        # ── 6. Index ───────────────────────────────────────────────────────────
        console.print("  Uploading to Azure AI Search...")
        t_index_start = time.time()
        index_result = self._indexer.upload_chunks(chunks)
        t_index = time.time() - t_index_start
        console.print(
            f"  Uploaded {index_result['uploaded']} chunks "
            f"({index_result['failed']} failed) ({t_index:.1f}s)"
        )

        total_time = time.time() - start_time

        # ── 7. Write ingestion receipt ─────────────────────────────────────────
        receipt = {
            "document_id": document_id,
            "document_type": document_type,
            "document_title": parsed_doc.document_title,
            "pdf_path": str(pdf_path),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "total_pages": parsed_doc.total_pages,
            "chunks_created": len(chunks),
            "chunks_uploaded": index_result["uploaded"],
            "chunks_failed": index_result["failed"],
            "total_tokens": total_tokens,
            "chunking_strategy": chunker.strategy_name,
            "chunk_type_breakdown": type_counts,
            "timing": {
                "parse_s": round(t_parse, 2),
                "chunk_s": round(t_chunk, 2),
                "embed_s": round(t_embed, 2),
                "index_s": round(t_index, 2),
                "total_s": round(total_time, 2),
            },
        }
        self._write_receipt(receipt)

        console.print(f"\n  [green]✓[/green] Done in {total_time:.1f}s")
        return receipt

    def _upload_to_blob(self, pdf_path: Path, document_id: str) -> None:
        """Upload the source PDF to Azure Blob Storage."""
        if self._blob_client is None:
            self._blob_client = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            )
        blob_name = f"{document_id}.pdf"
        container_client = self._blob_client.get_container_client(
            settings.azure_storage_container
        )
        with open(pdf_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        console.print(f"  Uploaded to Blob Storage: {blob_name}")

    def _write_receipt(self, receipt: dict) -> None:
        """Save ingestion receipt to data/ingested/{document_id}.json."""
        receipt_dir = (
            Path(__file__).parent.parent.parent / "data" / "ingested"
        )
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{receipt['document_id']}.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
        console.print(f"  Receipt: {receipt_path.name}")
